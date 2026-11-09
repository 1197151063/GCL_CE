import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .train import VARIANTS, jsonable, parser as training_parser, validate, write_json
from .summarize import summarize
from .presets import PRESETS


def main():
    p = argparse.ArgumentParser(description="Run shared-batch baselines, 2x2 sampling matrix and no-CL controls", allow_abbrev=False,
                                epilog="Other training options are forwarded; use train --help to list them.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=["custom", *PRESETS], default="custom")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", choices=VARIANTS, nargs="+", default=["shared-fast", "shared-full", "A", "B", "C", "D", "bpr-fast", "bpr-full"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    args, extra = p.parse_known_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error("Seeds and variants must be unique")
    reserved = ("--seed", "--variant", "--data-dir", "--output-dir")
    if any(token.split("=", 1)[0] in reserved for token in extra):
        p.error("Use --seeds/--variants and suite paths instead of overriding run identity")
    if not (args.data_dir / "train.txt").is_file() or not (args.data_dir / "test.txt").is_file():
        p.error("data-dir must contain train.txt and test.txt")
    commands, parsed = [], []
    for variant in args.variants:
        for seed in args.seeds:
            output = (args.output_dir / variant / f"seed_{seed}").resolve()
            options = ["--dataset", args.dataset, "--data-dir", str(args.data_dir.resolve()), "--output-dir", str(output),
                       "--variant", variant, "--seed", str(seed), *extra]
            run_args = training_parser().parse_args(options)
            validate(run_args)
            if not VARIANTS[variant][3]:
                run_args.ssl_weight = 0.
            commands.append({"variant": variant, "seed": seed, "output_dir": str(output),
                             "command": [sys.executable, "-m", "rrec_experiments.train", *options]})
            parsed.append(run_args)
    plan = {"runs": commands, "data_dir": str(args.data_dir.resolve())}
    if args.dry_run:
        for item in commands:
            print(shlex.join(item["command"]))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "suite.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        p.error("Existing suite plan differs; use a new output directory")
    write_json(plan_path, plan)
    for item, run_args in zip(commands, parsed):
        output = Path(item["output_dir"])
        if args.skip_completed and (output / "result.json").exists():
            stored = json.loads((output / "config.json").read_text())
            expected = jsonable(vars(run_args))
            if any(stored.get(key) != value for key, value in expected.items()):
                p.error(f"Stored configuration differs: {output}")
            print(f"Skipping completed run: {output}", flush=True)
            continue
        print(f"Running {item['variant']} seed={item['seed']}", flush=True)
        subprocess.run(item["command"], check=True)
    summarize(args.output_dir)
    print(f"Completed {len(commands)} runs; summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
