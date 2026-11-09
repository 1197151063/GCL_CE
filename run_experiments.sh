#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
  python_bin="$repo_dir/.venv/bin/python"
else
  python_bin="python3"
fi
exec "$python_bin" -m rrec_experiments.datasets "$@"
