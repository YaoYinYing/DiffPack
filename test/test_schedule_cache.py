from __future__ import annotations

from pathlib import Path

import numpy as np

from diffpack import schedule_cache


def test_resolve_cache_root_uses_diffpackcache_name():
    root = schedule_cache.resolve_cache_root(None)
    assert "DiffPackCache" in root


def test_validate_schedule_cache_reports_missing(tmp_path: Path):
    errs = schedule_cache.validate_schedule_cache(str(tmp_path), np.pi)
    assert errs
    assert any(e.startswith("missing:") for e in errs)

