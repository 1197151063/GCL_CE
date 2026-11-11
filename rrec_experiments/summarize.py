import argparse
import csv
import json
from pathlib import Path
import statistics


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def summarize(directory, allow_incomplete=False):
    directory = Path(directory)
    plan_path = directory / "suite.json"
    if plan_path.exists() and not allow_incomplete:
        plan = json.loads(plan_path.read_text())
        missing = [x["output_dir"] for x in plan["runs"] if not (Path(x["output_dir"]) / "result.json").exists()]
        if missing:
            raise ValueError(f"Incomplete suite: {len(missing)} runs missing results")
    paths = sorted(directory.glob("*/seed_*/result.json"))
    if not paths:
        raise ValueError(f"No completed suite runs under {directory}")
    groups, records, reference = {}, {}, None
    compare_fields = ("dataset", "data", "epochs", "batch_size", "dim", "layers", "init_std", "lr", "decay",
                      "temperature", "drop_ratio", "self_loops", "l2_source", "ssl_side", "score",
                      "split_seed", "data_protocol", "ks", "selection_k", "selection_split", "eval_every", "patience",
                      "strict_determinism", "repository_commit", "torch_version")
    compare_fields += ("experiment_protocol", "source_fingerprint", "init", "cl_sampler",
                       "cl_anchor_mode", "loss_reduction", "augmentation", "selection_objective")
    cl_weights = set()
    for path in paths:
        result = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        config.setdefault("dataset", "custom")
        config.setdefault("selection_split", "validation")
        config.setdefault("data_protocol", "legacy_validation_holdout")
        signature = {key: config.get(key) for key in compare_fields}
        if reference is not None and signature != reference:
            raise ValueError(f"Cannot pool runs with different controlled settings: {path}")
        reference = signature
        if not result["variant"].startswith("bpr-"):
            cl_weights.add(config["ssl_weight"])
        key = result["variant"], result["seed"]
        if key in records:
            raise ValueError(f"Duplicate variant/seed: {key}")
        records[key] = result
        numeric = flatten({**({"validation": result["validation"]} if "validation" in result else {}), "test": result["test"],
                           "best_epoch": result["best_epoch"], "stopped_epoch": result["stopped_epoch"],
                           "optimizer_steps": result["optimizer_steps"]})
        diagnostics_path = path.parent / "diagnostics.jsonl"
        if diagnostics_path.exists():
            diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
            best_diagnostics = [row for row in diagnostics if row.get("checkpoint") == "best"]
            if best_diagnostics:
                numeric.update(flatten(best_diagnostics[-1], "best_diagnostics"))
        groups.setdefault(key[0], []).append(numeric)
    if len(cl_weights) > 1:
        raise ValueError("Cannot pool runs with different CL weights")
    summary = {}
    for variant, rows in groups.items():
        metric_names = sorted(set.intersection(*(set(row) for row in rows)))
        summary[variant] = {name: {"mean": statistics.mean(row[name] for row in rows),
                                 "std": statistics.stdev(row[name] for row in rows) if len(rows) > 1 else 0.,
                                 "n": len(rows)} for name in metric_names}
    pairs = {}
    for left, right in [("fast", "full"), ("official", "fast"), ("official", "full"),
                        ("fast", "bpr-fast"), ("full", "bpr-full"), ("bpr-fast", "bpr-full"),
                        ("shared-fast", "shared-full"), ("A", "B"), ("C", "B"), ("D", "B")]:
        seeds = sorted({s for v, s in records if v == left} & {s for v, s in records if v == right})
        if not seeds:
            continue
        pair = {}
        for metric in sorted(set(flatten(records[(left, seeds[0])]["test"])) & set(flatten(records[(right, seeds[0])]["test"]))):
            if "recall" not in metric and "ndcg" not in metric:
                continue
            deltas = [flatten(records[(left, seed)]["test"])[metric] - flatten(records[(right, seed)]["test"])[metric] for seed in seeds]
            pair[metric] = {"mean_delta": statistics.mean(deltas), "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.,
                            "n": len(seeds), "seeds": seeds}
        pairs[f"{left}_minus_{right}"] = pair
    test_selection = reference["selection_split"] == "test"
    output = {"variants": summary, "paired_test_differences": pairs,
              "selection_split": reference["selection_split"], "test_used_for_selection": test_selection,
              "data_protocol": reference["data_protocol"],
              "experiment_protocol": reference["experiment_protocol"],
              "notes": ["std is sample standard deviation; n=1 has no uncertainty estimate.",
                        "Paired differences use matching training seeds; they do not establish statistical significance.",
                        ("Test data were used for checkpoint selection and early stopping; test scores are not an independent held-out estimate."
                         if test_selection else "Test comparisons are for reporting; do not select hyperparameters on these results.")]}
    (directory / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    columns = sorted({key for item in summary.values() for key in item})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "seeds"] + [f"{name}.{stat}" for name in columns for stat in ("mean", "std")])
        for variant, metrics in sorted(summary.items()):
            writer.writerow([variant, len(groups[variant])] + [metrics.get(name, {}).get(stat, "") for name in columns for stat in ("mean", "std")])
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate matched-seed experiment results")
    p.add_argument("directory", type=Path)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    result = summarize(args.directory, args.allow_incomplete)
    print(f"Summarized {len(result['variants'])} variants into {args.directory / 'summary.csv'}")
