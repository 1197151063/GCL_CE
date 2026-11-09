#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"
exec "$repo_dir/.venv/bin/python" -m rrec_experiments.suite \
  --data-dir "$repo_dir/../R-Rec/data/yelp2018" \
  --output-dir "$repo_dir/runs/yelp2018" "$@"
