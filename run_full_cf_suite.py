"""Run user-uniform baselines sequentially, preserving completed and interrupted cases."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys


ORDER = ("rgcf", "lightgcl", "lightccf", "sgcl", "adagcl", "nlgcl", "paac")


def run_one(method, args):
    home = args.output / method
    home.mkdir(exist_ok=True)
    for path in home.glob("attempt_*/result.json"):
        if "test" in json.loads(path.read_text()):
            return True
    number = 1
    while (home / f"attempt_{number}").exists():
        number += 1
    dest = home / f"attempt_{number}"
    dest.mkdir()
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("run_full_cf.py")),
        "--method",
        method,
        "--device",
        "cuda",
        "--dataset",
        args.dataset,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(dest),
    ]
    print("Starting", method, dest, flush=True)
    with (dest / "train.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = (dest / "train.log").read_text()[-5000:]
        raise RuntimeError(tail)
    return True


def suite(args):
    if args.wait_queue:
        print("Waiting for", args.wait_queue, flush=True)
        with (args.wait_queue / "queue.lock").open("r") as other:
            fcntl.flock(other, fcntl.LOCK_SH)
            if not (args.wait_queue / "best.json").is_file():
                raise RuntimeError("Preceding experiments are incomplete")
    for model in ORDER:
        run_one(model, args)
    summary = {}
    for model in ORDER:
        path = next((args.output / model).glob("attempt_*/result.json"))
        summary[model] = json.loads(path.read_text())
    tmp = args.output / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(args.output / "summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-queue", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite(args)
