from __future__ import annotations

import argparse
import json

from diffpack.schedule_cache import prepare_schedule_cache, required_schedule_pis, resolve_cache_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DiffPack schedule caches")
    parser.add_argument("--cache_root", default=None, help="cache root override (default: platformdirs DiffPackCache)")
    parser.add_argument("--force", action="store_true", help="force rebuild even if cache validates")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    cache_root = resolve_cache_root(args.cache_root)
    events = [prepare_schedule_cache(cache_root, pi, force=args.force) for pi in required_schedule_pis()]
    print(json.dumps({"cache_root": cache_root, "events": events}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

