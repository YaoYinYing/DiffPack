from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from typing import Any

from diffpack.backends.base import InferenceRequest
from diffpack.mutation import normalize_mutations, preprocess_pdb_files_with_mutations


def _radius_mode(radius: float) -> str:
    if radius == -1:
        return "mutated_only"
    if radius == 0:
        return "full_repack"
    return "local_repack"


def prepare_mutation_request(request: InferenceRequest) -> tuple[InferenceRequest, dict[str, Any]]:
    specs = normalize_mutations(request.mutations)
    if not specs:
        return request, {
            "mutations_requested": [],
            "mutations_applied": [],
            "mutation_validation_errors": [],
            "mutation_preprocessed_pdb_files": [],
        }

    if request.repack_radius is None:
        raise ValueError(
            "`repack_radius` must be set when `mutations` are provided. "
            "Use `-1` (mutated residues only), `0` (full repack), or `>0` (local repack radius)."
        )
    if request.repack_radius < -1:
        raise ValueError(f"`repack_radius` must be one of -1, 0, or >0. Got {request.repack_radius}.")

    mutated_pdb_files, mutation_sites = preprocess_pdb_files_with_mutations(
        pdb_files=request.pdb_files,
        mutations=specs,
        output_dir=request.output_dir,
        pro_remodel_window=request.pro_remodel_window,
        pro_remodel_max_steps=request.pro_remodel_max_steps,
    )

    centers = list(request.center_residues)
    if request.repack_radius > 0:
        centers = sorted(set(centers).union(mutation_sites))
    frozen_sites = sorted({spec.selector for spec in specs if spec.new_res_3 == "PRO"})

    updated_request = replace(
        request,
        pdb_files=mutated_pdb_files,
        center_residues=centers,
        mutation_residues=mutation_sites,
        frozen_residues=frozen_sites,
        mutations=[spec.to_dict() for spec in specs],
    )
    metadata = {
        "mutations_requested": [spec.compact for spec in specs],
        "mutations_applied": [spec.to_dict() for spec in specs],
        "mutation_validation_errors": [],
        "mutation_preprocessed_pdb_files": mutated_pdb_files,
        "radius_mode": _radius_mode(float(request.repack_radius)),
        "mutation_site_count": len(mutation_sites),
        "effective_center_count": len(centers),
        "effective_center_residues": centers,
        "pro_remodel_window": request.pro_remodel_window,
        "pro_remodel_max_steps": request.pro_remodel_max_steps,
        "frozen_residues": frozen_sites,
    }
    diagnostics_path = Path(request.output_dir).resolve() / "_mutation_inputs" / "mutation_diagnostics.json"
    if diagnostics_path.exists():
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            diagnostics = {}
        metadata.update(
            {
                "pro_solver_mode": diagnostics.get("pro_solver_mode"),
                "pro_solver_window": diagnostics.get("pro_solver_window"),
                "pro_solver_iterations": diagnostics.get("pro_solver_iterations"),
                "pro_solver_final_cost": diagnostics.get("pro_solver_final_cost"),
                "pro_applied_remodel_residues": diagnostics.get("pro_applied_remodel_residues", []),
            }
        )
    return updated_request, metadata
