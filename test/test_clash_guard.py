from __future__ import annotations

import torch

from diffpack.clash_guard import mitigate_severe_clashes


class _DummyProtein:
    def __init__(self, coords: torch.Tensor):
        self.node_position = coords.clone()
        self.atom2residue = torch.tensor([0, 1], dtype=torch.long)
        self.residue_chain = ["A", "A"]
        self.residue_number = [1, 2]
        self.num_node = 2

    @property
    def device(self):
        return self.node_position.device

    def cpu(self):
        self.node_position = self.node_position.cpu()
        self.atom2residue = self.atom2residue.cpu()
        return self

    def to_pdb(self, path: str):
        lines = []
        for i, (x, y, z) in enumerate(self.node_position.tolist(), start=1):
            res_num = i
            lines.append(
                f"ATOM  {i:5d} {'CA':>4s} {'ALA':>3s} {'A':1s}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {'C':>2s}"
            )
        lines.extend(["TER", "END"])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def test_clash_guard_reverts_clashing_residues():
    # Atom pair at 0.5A is a severe clash under threshold 1.0.
    pred = _DummyProtein(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32))
    ref = _DummyProtein(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32))
    fixed, summary = mitigate_severe_clashes(pred, ref, threshold=1.0, max_iter=2)
    assert summary["clash_guard_applied"] is True
    assert summary["clashes_before"] > 0
    assert summary["clashes_after"] == 0
    assert summary["reverted_residues"] >= 1
    assert torch.allclose(fixed.node_position, ref.node_position)
