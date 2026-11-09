#!/usr/bin/env bash
set -e
cd /home/lhb/cyj/GCL_CE
python -u run.py --device cuda "$@"
