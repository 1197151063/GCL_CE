#!/usr/bin/env bash
set -euo pipefail

# Server checkout and existing Python environment. Never create a new venv.
project_dir="${GCL_PROJECT_DIR:-/home/lhb/cyj/GCL_CE}"
environment_name="${GCL_ENV_NAME:-cyj}"

if [[ ! -d "$project_dir/rrec_experiments" ]]; then
  echo "Project not found: $project_dir (set GCL_PROJECT_DIR to override)." >&2
  exit 1
fi
cd "$project_dir"
project_dir="$(pwd -P)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  runner=("$PYTHON_BIN")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$environment_name" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  runner=("$CONDA_PREFIX/bin/python")
elif [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV##*/}" == "$environment_name" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  runner=("$VIRTUAL_ENV/bin/python")
elif [[ "$(basename "$(dirname "$project_dir")")" == "$environment_name" && -x "$(dirname "$project_dir")/bin/python" ]]; then
  runner=("$(dirname "$project_dir")/bin/python")
else
  conda_bin="${CONDA_EXE:-}"
  if [[ ! -x "$conda_bin" ]]; then
    conda_bin="$(type -P conda || true)"
  fi
  if [[ -z "$conda_bin" || ! -x "$conda_bin" ]]; then
    echo "Cannot locate environment '$environment_name'. Activate it first, or set CONDA_EXE / PYTHON_BIN." >&2
    exit 1
  fi
  runner=("$conda_bin" run --no-capture-output -n "$environment_name" python)
fi

echo "Project: $project_dir"
echo "Python environment: $environment_name"
export PYTHONUNBUFFERED=1
exec "${runner[@]}" -u -m rrec_experiments.datasets \
  --data-root "$project_dir/data" --output-root "$project_dir/runs" \
  --device cuda "$@"
