from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
try:
    from platformdirs import user_cache_dir  # type: ignore
except Exception:  # pragma: no cover
    def user_cache_dir(appname: str) -> str:
        return os.path.join(os.path.expanduser("~"), ".cache", appname)


CACHE_APP_NAME = "DiffPackCache"
CACHE_VERSION = "v1"
X_MIN, X_N = 1e-5, 5000
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000


def resolve_cache_root(cache_root: str | None) -> str:
    if cache_root:
        return os.path.realpath(os.path.expanduser(cache_root))
    return os.path.realpath(user_cache_dir(CACHE_APP_NAME))


def required_schedule_pis() -> list[float]:
    return [np.pi, 0.5 * np.pi]


def _schedule_prefix(pi: float) -> str:
    return (
        f"Periodic.{pi:.3f}."
        f"ver-{CACHE_VERSION}.x-{X_N}.sigma-{SIGMA_N}."
        f"sigmamin-{SIGMA_MIN:g}.sigmamax-{SIGMA_MAX:g}"
    )


def schedule_cache_paths(cache_root: str, pi: float) -> dict[str, str]:
    root = Path(resolve_cache_root(cache_root)) / "schedules"
    prefix = _schedule_prefix(pi)
    return {
        "root": str(root),
        "p": str(root / f"{prefix}.p.npy"),
        "score": str(root / f"{prefix}.score.npy"),
        "score_norm": str(root / f"{prefix}.score_norm.npy"),
        "meta": str(root / f"{prefix}.meta.json"),
    }


def _periodic_p(x: np.ndarray, sigma: np.ndarray, n: int = 100, pi: float = np.pi) -> np.ndarray:
    out = 0
    for i in range(-n, n + 1):
        out += np.exp(-((x + 2 * pi * i) ** 2) / (2 * sigma**2))
    return out


def _periodic_grad(x: np.ndarray, sigma: np.ndarray, n: int = 100, pi: float = np.pi) -> np.ndarray:
    out = 0
    for i in range(-n, n + 1):
        out += (x + 2 * pi * i) / (sigma**2) * np.exp(-((x + 2 * pi * i) ** 2) / (2 * sigma**2))
    return out


def _periodic_sample(sigma: np.ndarray, pi: float = np.pi, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = sigma * rng.standard_normal(size=sigma.shape)
    out = (out + pi) % (2 * pi) - pi
    return out


def _atomic_save_npy(path: str, arr: np.ndarray):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, target)


def _atomic_write_json(path: str, payload: dict):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def validate_schedule_cache(cache_root: str, pi: float) -> list[str]:
    paths = schedule_cache_paths(cache_root, pi)
    errors: list[str] = []
    required = ("p", "score", "score_norm", "meta")
    for key in required:
        if not Path(paths[key]).exists():
            errors.append(f"missing:{key}:{paths[key]}")
    if errors:
        return errors

    try:
        p = np.load(paths["p"])
    except Exception as exc:
        errors.append(f"load_error:p:{exc}")
        p = None
    try:
        score = np.load(paths["score"])
    except Exception as exc:
        errors.append(f"load_error:score:{exc}")
        score = None
    try:
        score_norm = np.load(paths["score_norm"])
    except Exception as exc:
        errors.append(f"load_error:score_norm:{exc}")
        score_norm = None

    try:
        meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"load_error:meta:{exc}")
        meta = None

    expected_shape = (SIGMA_N + 1, X_N + 1)
    expected_norm_shape = (SIGMA_N + 1,)
    if p is not None and p.shape != expected_shape:
        errors.append(f"shape_error:p:{p.shape}!={expected_shape}")
    if score is not None and score.shape != expected_shape:
        errors.append(f"shape_error:score:{score.shape}!={expected_shape}")
    if score_norm is not None and score_norm.shape != expected_norm_shape:
        errors.append(f"shape_error:score_norm:{score_norm.shape}!={expected_norm_shape}")

    if score_norm is not None and not np.isfinite(score_norm).all():
        errors.append("non_finite:score_norm")
    if p is not None:
        sample = p[::500, ::500]
        if not np.isfinite(sample).all():
            errors.append("non_finite_sample:p")
    if score is not None:
        sample = score[::500, X_N // 2]
        if not np.isfinite(sample).all():
            errors.append("non_finite_sample:score_center")

    if meta is not None:
        if meta.get("cache_version") != CACHE_VERSION:
            errors.append(f"meta_version:{meta.get('cache_version')}!={CACHE_VERSION}")
        if abs(float(meta.get("pi", 0.0)) - float(pi)) > 1e-6:
            errors.append(f"meta_pi:{meta.get('pi')}!={pi}")
        if tuple(meta.get("shape", [])) != expected_shape:
            errors.append(f"meta_shape:{meta.get('shape')}!={expected_shape}")
        if tuple(meta.get("score_norm_shape", [])) != expected_norm_shape:
            errors.append(f"meta_norm_shape:{meta.get('score_norm_shape')}!={expected_norm_shape}")
    return errors


def load_schedule_tables_readonly(cache_root: str, pi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = validate_schedule_cache(cache_root, pi)
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"Invalid schedule cache for PI={pi:.3f} at `{resolve_cache_root(cache_root)}`: {joined}")
    paths = schedule_cache_paths(cache_root, pi)
    return np.load(paths["p"]), np.load(paths["score"]), np.load(paths["score_norm"])


def prepare_schedule_cache(cache_root: str, pi: float, force: bool = False) -> dict:
    root = resolve_cache_root(cache_root)
    paths = schedule_cache_paths(root, pi)
    if not force:
        errors = validate_schedule_cache(root, pi)
        if not errors:
            return {"pi": pi, "status": "reused", "paths": paths}

    x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * pi
    sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * pi
    p = _periodic_p(x, sigma[:, None], n=100, pi=pi)
    score = _periodic_grad(x, sigma[:, None], n=100, pi=pi) / p
    sampled = _periodic_sample(sigma[None].repeat(10000, 0).flatten(), pi=pi, seed=0)
    score_norm_table = _lookup_score(score, sampled, sigma[None].repeat(10000, 0).flatten(), pi).reshape(10000, -1)
    score_norm = (score_norm_table**2).mean(0)

    _atomic_save_npy(paths["p"], p)
    _atomic_save_npy(paths["score"], score)
    _atomic_save_npy(paths["score_norm"], score_norm)
    _atomic_write_json(
        paths["meta"],
        {
            "cache_version": CACHE_VERSION,
            "pi": float(pi),
            "shape": [SIGMA_N + 1, X_N + 1],
            "score_norm_shape": [SIGMA_N + 1],
            "sigma_min": SIGMA_MIN,
            "sigma_max": SIGMA_MAX,
            "x_min": X_MIN,
        },
    )
    errors = validate_schedule_cache(root, pi)
    if errors:
        raise RuntimeError(f"Prepared cache failed validation for PI={pi:.3f}: {'; '.join(errors)}")
    return {"pi": pi, "status": "built", "paths": paths}


def _lookup_score(score_table: np.ndarray, x: np.ndarray, sigma: np.ndarray, pi: float) -> np.ndarray:
    x = (x + pi) % (2 * pi) - pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / pi + 1e-10)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_table[sigma, x]


def validate_required_schedule_caches(cache_root: str) -> dict:
    root = resolve_cache_root(cache_root)
    result = {"cache_root": root, "keys": [], "errors": []}
    for pi in required_schedule_pis():
        errs = validate_schedule_cache(root, pi)
        key = {"pi": float(pi), "status": "ok" if not errs else "invalid", "errors": errs}
        result["keys"].append(key)
        if errs:
            result["errors"].append({"pi": float(pi), "errors": errs})
    return result
