#!/usr/bin/env bash
set -e
cd /home/lhb/cyj/GCL_CE
python -u run.py --datasets yelp2018 --device cuda "$@"
