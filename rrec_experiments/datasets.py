"""Run the same controlled sampling suite on one or more project datasets."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from .presets import PRESETS, preset_options


def commands(args, extra):
    datasets = list(PRESETS) if args.datasets == ["all"] else args.datasets
    if "all" in datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Use all alone, or list unique dataset names")
    reserved = {"--data-dir", "--output-dir", "--dataset"}
    if any(token.split("=", 1)[0] in reserved for token in extra):
        raise ValueError("Use --data-root/--output-root/--datasets with the multi-dataset launcher")
    result = []
    for dataset in datasets:
        directory = (args.data_root / dataset).resolve()
        if not all((directory / name).is_file() for name in ["train.txt", "test.txt"]):
            raise ValueError(f"Missing train.txt/test.txt in {directory}")
        result.append([sys.executable, "-m", "rrec_experiments.suite",
                       "--dataset", dataset, "--data-dir", str(directory),
                       "--output-dir", str((args.output_root / dataset).resolve()),
                       *preset_options(dataset), *extra,
                       *(["--dry-run"] if args.dry_run else [])])
    return result


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                epilog="Other options, e.g. --seeds, --variants, --device, are forwarded to every suite.")
    p.add_argument("--datasets", choices=["all", *PRESETS], nargs="+", default=["yelp2018"])
    p.add_argument("--data-root", type=Path, default=root / "data")
    p.add_argument("--output-root", type=Path, default=root / "runs" / "sgl_official_v2")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    try:
        planned = commands(args, extra)
    except ValueError as error:
        p.error(str(error))
    for command in planned:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
