import glob
import logging
import os
import tempfile

import torch
from rdkit import Chem
from diffpack.torchdrug import data
from diffpack.torchdrug.core import Registry as R
from diffpack.torchdrug.layers import functional
from tqdm import tqdm

from diffpack import rotamer, repack
from diffpack.rotamer import get_chi_mask, atom_name_vocab, bb_atom_name

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


@R.register("datasets.SideChainDataset")
class SideChainDataset(data.ProteinDataset):
    processed_file = None
    exclude_pdb_files = []

    def __init__(self, path=None, pdb_files=None, verbose=1, **kwargs):
        center_residues = kwargs.pop("center_residues", None)
        repack_radius = kwargs.pop("repack_radius", None)
        center_residues = center_residues or []
        self.center_residue_selectors = repack.parse_center_residue_selectors(center_residues)
        self.repack_radius = repack_radius
        self.hetero_policy = kwargs.pop("hetero_policy", "exclude")
        if self.hetero_policy not in {"exclude", "context_only", "error"}:
            raise ValueError("`hetero_policy` must be one of: exclude, context_only, error")
        self._temp_pdb_dir = tempfile.mkdtemp(prefix="diffpack_pdb_")

        if self.repack_radius is not None and len(self.center_residue_selectors) == 0:
            raise ValueError("`center_residues` must be provided when `repack_radius` is set.")
        if self.repack_radius is None and len(self.center_residue_selectors) > 0:
            raise ValueError("`repack_radius` must be provided when `center_residues` is set.")
        if self.repack_radius is not None and self.repack_radius <= 0:
            raise ValueError(f"`repack_radius` must be > 0, got {self.repack_radius}.")

        if path is not None:
            logger.info("Loading dataset from folder %s" % path)
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.path = path
            pkl_file = os.path.join(path, self.processed_file)

            if os.path.exists(pkl_file):
                logger.info("Found existing pickle file %s" % pkl_file
                            + ". Loading from pickle file (this may take a while)")
                self.load_pickle(pkl_file, verbose=verbose, **kwargs)
            else:
                logger.info("No pickle file found. Loading from pdb files (this may take a while)"
                            + " and save to pickle file %s" % pkl_file)
                pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
                self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
                self.save_pickle(pkl_file, verbose=verbose)
        elif pdb_files is not None:
            logger.info("Loading dataset from pdb files")
            pdb_files = [os.path.expanduser(pdb_file) for pdb_file in pdb_files]
            pdb_files = [pdb_file for pdb_file in pdb_files if pdb_file.endswith(".pdb")]
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)

        # Filter out proteins with no residues
        indexes = [i for i, (protein, pdb_file) in enumerate(zip(self.data, self.pdb_files))
                   if (protein.num_residue > 0).all() and os.path.basename(pdb_file) not in self.exclude_pdb_files]
        self.data = [self.data[i] for i in indexes]
        self.sequences = [self.sequences[i] for i in indexes]
        self.pdb_files = [self.pdb_files[i] for i in indexes]
        if hasattr(self, "parsed_pdb_files"):
            self.parsed_pdb_files = [self.parsed_pdb_files[i] for i in indexes]

    def _prepare_pdb_file(self, pdb_file: str) -> str:
        with open(pdb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        has_hetatm = any(line.startswith("HETATM") for line in lines)
        if self.hetero_policy == "error" and has_hetatm:
            raise ValueError(
                f"HETATM records found in `{pdb_file}` while `hetero_policy=error`."
            )
        if self.hetero_policy in {"exclude", "context_only"} and has_hetatm:
            if self.hetero_policy == "context_only":
                logger.warning(
                    "`hetero_policy=context_only` is currently mapped to ATOM-only parsing in this runtime path."
                )
            filtered = []
            for line in lines:
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("MODEL") \
                        or line.startswith("ENDMDL") or line.startswith("END"):
                    filtered.append(line)
            temp_path = os.path.join(self._temp_pdb_dir, os.path.basename(pdb_file))
            with open(temp_path, "w", encoding="utf-8") as f:
                f.writelines(filtered)
            return temp_path
        return pdb_file

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, sanitize=True, removeHs=True, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.parsed_pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            parse_pdb_file = self._prepare_pdb_file(pdb_file)
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(parse_pdb_file, sanitize=sanitize, removeHs=removeHs)
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.parsed_pdb_files.append(parse_pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.parsed_pdb_files[index], **self.kwargs)
        else:
            protein = self.data[index].clone()
        protein = protein.subgraph(protein.atom_name < 37)

        with protein.atom():
            # Init atom14 index map
            protein.atom14index = rotamer.restype_atom14_index_map[
                protein.residue_type[protein.atom2residue], protein.atom_name
            ]  # [num_atom, 14]

        with protein.residue():
            # Init residue features
            protein.residue_feature = functional.one_hot(protein.residue_type, 21)  # [num_residue, 21]

            # Init residue masks
            chi_mask = get_chi_mask(protein)
            chi_1pi_periodic_mask = torch.tensor(rotamer.chi_pi_periodic)[protein.residue_type]
            chi_2pi_periodic_mask = ~chi_1pi_periodic_mask
            protein.chi_mask = chi_mask
            protein.chi_1pi_periodic_mask = torch.logical_and(chi_mask, chi_1pi_periodic_mask)  # [num_residue, 4]
            protein.chi_2pi_periodic_mask = torch.logical_and(chi_mask, chi_2pi_periodic_mask)  # [num_residue, 4]

            # Init atom37 features
            protein.atom37_mask = torch.zeros(protein.num_residue, len(atom_name_vocab), device=protein.device,
                                              dtype=torch.bool)  # [num_residue, 37]
            protein.atom37_mask[protein.atom2residue, protein.atom_name] = True
            protein.sidechain37_mask = protein.atom37_mask.clone()  # [num_residue, 37]
            protein.sidechain37_mask[:, bb_atom_name] = False

            if self.repack_radius is None:
                protein.repack_residue_mask = torch.ones(protein.num_residue, dtype=torch.bool, device=protein.device)
            else:
                try:
                    residue_identifiers = repack.load_residue_identifiers_from_protein(protein)
                except Exception:
                    parsed_pdb_files = getattr(self, "parsed_pdb_files", self.pdb_files)
                    residue_identifiers = repack.load_residue_identifiers_from_pdb(
                        parsed_pdb_files[index], allowed_residue_names=rotamer.residue_list
                    )
                repack_residue_mask, _ = repack.select_residues_by_radius(
                    atom_positions=protein.node_position,
                    atom2residue=protein.atom2residue,
                    num_residue=protein.num_residue,
                    residue_identifiers=residue_identifiers,
                    center_selectors=self.center_residue_selectors,
                    radius=self.repack_radius,
                )
                protein.repack_residue_mask = repack_residue_mask
                logger.info(
                    "Selected %d / %d residues for repacking in %s",
                    int(repack_residue_mask.sum().item()),
                    int(protein.num_residue),
                    os.path.basename(self.pdb_files[index]),
                )
        item = {"graph": protein}

        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def from_pdb_files(pdb_files, verbose=1, **kwargs):
        return SideChainDataset(pdb_files, verbose=verbose, **kwargs)

    def __repr__(self):
        lines = ["#sample: %d" % len(self)]
        return "%s(  %s)" % (self.__class__.__name__, "\n  ".join(lines))
