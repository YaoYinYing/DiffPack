from __future__ import annotations

import os
import pprint
import time
from contextlib import nullcontext

import numpy as np
import torch

from diffpack import util
from diffpack.device import choose_torch_device
from diffpack.schedule_cache import resolve_cache_root, validate_required_schedule_caches


class TorchDrugCompatibleRunner:
    def _configure_runtime_environment(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(output_dir, ".torch_extensions"))
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(output_dir, ".cache"))
        os.environ.setdefault("TORCHDRUG_ENABLE_OPENMP", "0")
        os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

    def _import_runtime(self):
        from diffpack.torchdrug import core
        from diffpack.torchdrug.utils import comm
        from diffpack import dataset, layer, schedule, task  # noqa: F401
        return core, comm

    def _set_seed(self, seed: int, comm):
        torch.manual_seed(seed + comm.get_rank())
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_solver(self, cfg, logger, core):
        from diffpack.engine import DiffusionEngine

        task = core.Configurable.load_config_dict(cfg.task)
        solver = DiffusionEngine(task, None, None, None, None, None, **cfg.engine)
        if "checkpoint" in cfg:
            solver.load(cfg.checkpoint, load_optimizer=cfg.get("load_optimizer", False))
        if "model_checkpoint" in cfg:
            model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            model_dict = torch.load(model_checkpoint, map_location=torch.device("cpu"))["model"]
            task.load_state_dict(model_dict, strict=False)
        if cfg.get("_override_device"):
            solver.device = cfg._override_device
            solver.model = solver.model.to(solver.device)
        if core.Configurable is not None:
            logger.warning("#parameter: %d", sum(p.numel() for p in task.parameters() if p.requires_grad))
        return solver

    def run(self, request, *, backend_requested: str, backend_effective: str, backend_mode: str, fallback_reason: str | None):
        self._configure_runtime_environment(request.output_dir)
        core, comm = self._import_runtime()

        cfg = util.load_config(os.path.realpath(request.config))
        cfg.test_set.pdb_files = request.pdb_files
        cfg.test_set.center_residues = request.center_residues
        cfg.test_set.repack_radius = request.repack_radius
        cfg.test_set.mutation_residues = request.mutation_residues or []
        cfg.test_set.frozen_residues = request.frozen_residues or []
        cfg.test_set.hetero_policy = request.hetero_policy
        cfg.backend = backend_effective
        cfg.cache = cfg.get("cache", {})
        cfg.cache["root"] = resolve_cache_root(request.cache_root or cfg.cache.get("root"))
        cfg.cache["mode"] = "read_only"
        if not request.cache_read_only:
            raise RuntimeError("Inference enforces read-only cache mode.")
        for skey in ("schedule_1pi_periodic", "schedule_2pi_periodic"):
            if skey in cfg.task:
                cfg.task[skey]["cache_folder"] = cfg.cache["root"]
                cfg.task[skey]["cache_read_only"] = True

        device = choose_torch_device(request.device)
        cfg._override_device = device
        if request.fast and getattr(cfg.task, "class", "") == "ConfidencePrediction" and "num_sample" in cfg.task:
            cfg.task.num_sample = max(1, min(int(cfg.task.num_sample), 2))

        self._set_seed(request.seed, comm)
        logger = util.get_root_logger(file=False)
        if comm.get_rank() == 0:
            logger.warning("Backend requested: %s", backend_requested)
            logger.warning("Backend effective: %s (%s)", backend_effective, backend_mode)
            logger.warning("Device: %s", device)
            logger.warning("Config file: %s", request.config)
            logger.warning(pprint.pformat(cfg))
            logger.warning("Output dir: %s", request.output_dir)
            logger.warning("Cache root: %s", cfg.cache["root"])
            logger.warning("Cache mode: %s", cfg.cache["mode"])
            logger.warning("Cache preflight validation: start")
            cache_validation = validate_required_schedule_caches(cfg.cache["root"])
            if cache_validation["errors"]:
                raise RuntimeError(
                    "Read-only cache validation failed. "
                    f"cache_root={cfg.cache['root']} errors={cache_validation['errors']}. "
                    f"Run `diffpack-prepare-cache --cache_root {cfg.cache['root']}` and retry."
                )
            logger.warning("Cache preflight validation: ok")
        else:
            cache_validation = validate_required_schedule_caches(cfg.cache["root"])
            if cache_validation["errors"]:
                raise RuntimeError(
                    "Read-only cache validation failed. "
                    f"cache_root={cfg.cache['root']} errors={cache_validation['errors']}. "
                    f"Run `diffpack-prepare-cache --cache_root {cfg.cache['root']}` and retry."
                )

        solver = self._build_solver(cfg, logger, core)
        test_set = core.Configurable.load_config_dict(cfg.test_set)

        profile_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) if request.profile else nullcontext()

        start = time.perf_counter()
        with profile_ctx as prof:
            run_summary = solver.generate(test_set=test_set, path=request.output_dir)
        elapsed = time.perf_counter() - start

        profile_path = None
        if request.profile and prof is not None:
            profile_path = os.path.join(request.output_dir, "torch_profile.txt")
            os.makedirs(request.output_dir, exist_ok=True)
            with open(profile_path, "w", encoding="utf-8") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total"))

        metadata = {
            "backend": backend_effective,
            "backend_requested": backend_requested,
            "backend_effective": backend_effective,
            "backend_mode": backend_mode,
            "fallback_reason": fallback_reason,
            "device": str(device),
            "elapsed_sec": elapsed,
            "profile_path": profile_path,
            "memory_mode": request.memory_mode,
            "device_allocated_peak_bytes": None,
            "device_reserved_peak_bytes": None,
            "memory_phase_peaks": {},
        }
        if isinstance(run_summary, dict):
            metadata.update(run_summary)
        metadata.update(
            {
                "cache_root": cfg.cache["root"],
                "cache_mode": cfg.cache["mode"],
                "cache_validation_status": "pass",
                "cache_validation_errors": [],
                "cache_keys": cache_validation["keys"],
            }
        )
        return metadata
