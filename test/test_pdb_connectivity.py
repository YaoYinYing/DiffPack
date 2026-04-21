from pathlib import Path

from diffpack.mutation import normalize_mutations, preprocess_pdb_files_with_mutations
from diffpack.pdb_connectivity import append_conect_records_inplace


def _parse_atom_serials_by_residue(path: Path) -> dict[tuple[str, int, str], int]:
    mapping: dict[tuple[str, int, str], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        serial = int(line[6:11])
        atom = line[12:16].strip()
        chain = line[21].strip()
        resseq = int(line[22:26])
        mapping[(chain, resseq, atom)] = serial
    return mapping


def _parse_conect(path: Path) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("CONECT"):
            continue
        src = int(line[6:11])
        for start in (11, 16, 21, 26):
            token = line[start : start + 5].strip()
            if not token:
                continue
            dst = int(token)
            edge = (min(src, dst), max(src, dst))
            edges.add(edge)
    return edges


def test_append_conect_records_inplace_writes_topology(tmp_path):
    src = Path("1ubq.pdb")
    dst = tmp_path / "1ubq.copy.pdb"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    assert append_conect_records_inplace(str(dst)) is True
    text = dst.read_text(encoding="utf-8")
    assert "CONECT" in text
    assert "MASTER" not in text


def test_proline_mutation_has_n_cd_conect(tmp_path):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")

    mutations = normalize_mutations("AA46P")
    out_files, _ = preprocess_pdb_files_with_mutations(
        [str(pdb_in)],
        mutations=mutations,
        output_dir=str(tmp_path),
    )
    out_pdb = Path(out_files[0])

    serials = _parse_atom_serials_by_residue(out_pdb)
    edges = _parse_conect(out_pdb)
    n_serial = serials[("A", 46, "N")]
    cd_serial = serials[("A", 46, "CD")]
    assert (min(n_serial, cd_serial), max(n_serial, cd_serial)) in edges
