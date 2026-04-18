#!/usr/bin/env bash
set -euo pipefail

TORCHDRUG_FORK_PATH="${1:-/Users/yyy/Documents/protein_design/torchdrug}"

if [[ ! -d "$TORCHDRUG_FORK_PATH" ]]; then
  echo "TorchDrug fork path not found: $TORCHDRUG_FORK_PATH" >&2
  exit 1
fi

python -m pip install -e "$TORCHDRUG_FORK_PATH"
echo "Installed torchdrug fork from: $TORCHDRUG_FORK_PATH"

