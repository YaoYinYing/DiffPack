import os
import logging

import torch
from diffpack.torchdrug import core, data
from diffpack.torchdrug.utils import comm

from diffpack.device import move_to_device
from diffpack.pdb_connectivity import append_conect_records_inplace

logger = logging.getLogger(__name__)

class DiffusionEngine(core.Engine):
    @torch.no_grad()
    def generate(self, test_set, path):
        if comm.get_rank() == 0:
            logger.warning(f"Test on {test_set}")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        logger.warning(path)
        dataloader = data.DataLoader(test_set, self.batch_size, shuffle=False)
        self.model.eval()
        sample_id = 0
        output_files = []
        metric_summary = {}
        for batch in dataloader:
            batch = move_to_device(batch, self.device)
            true_proteins = batch["graph"].clone()
            pred_proteins = self.model.generate(batch)["graph"]
            evaluation_metric = self.model.get_metric(pred_proteins, true_proteins, {})
            metric_summary = {
                "atom_rmsd_per_residue": float(evaluation_metric["atom_rmsd_per_residue"].mean().item()),
                "chi_0_mae_deg": float(evaluation_metric["chi_0_ae_deg"].mean().item()),
                "chi_1_mae_deg": float(evaluation_metric["chi_1_ae_deg"].mean().item()),
                "chi_2_mae_deg": float(evaluation_metric["chi_2_ae_deg"].mean().item()),
                "chi_3_mae_deg": float(evaluation_metric["chi_3_ae_deg"].mean().item()),
            }
            print(f"atom_rmsd_per_residue: {evaluation_metric['atom_rmsd_per_residue'].mean():<20}"
                  f"chi_0_mae_deg: {evaluation_metric['chi_0_ae_deg'].mean():<20}"
                  f"chi_1_mae_deg: {evaluation_metric['chi_1_ae_deg'].mean():<20}"
                  f"chi_2_mae_deg: {evaluation_metric['chi_2_ae_deg'].mean():<20}"
                  f"chi_3_mae_deg: {evaluation_metric['chi_3_ae_deg'].mean():<20}")
            for p in pred_proteins.unpack():
                pdb_file = os.path.basename(test_set.pdb_files[sample_id])
                protein = p.cpu()
                output_path = os.path.join(path, pdb_file)
                protein.to_pdb(output_path)
                append_conect_records_inplace(output_path)
                output_files.append(output_path)
                sample_id += 1

        return {
            "metrics": metric_summary,
            "output_files": output_files,
        }
