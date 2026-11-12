"""One-seed Yelp attribution study; all treatments have identical Fast BPR."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .train import (EXPERIMENT_PROTOCOL, jsonable, parser as training_parser,
                    source_fingerprint, validate, write_json)

STUDY_PROTOCOL = "sampling_attribution_v1"
PROFILES = {
    "baseline": ["--variant", "fast"],
    "cl-fast": ["--variant", "fast", "--cl-sampler", "fast"],
    "l2-fast": ["--variant", "fast", "--l2-source", "bpr"],
    "cl-l2-fast": ["--variant", "fast", "--cl-sampler", "fast", "--l2-source", "bpr"],
    "shared-fast": ["--variant", "shared-fast"],
    "head-up": ["--variant", "fast", "--user-cl-head-target", "fast"],
    "head-down": ["--variant", "shared-fast", "--user-cl-head-target", "user"],
    "no-cl": ["--variant", "bpr-fast", "--ssl-weight", "0"],
}
CONTRASTS = {
    "cl_with_fixed_l2": ("cl-fast", "baseline", "CL distribution; official L2 fixed"),
    "cl_with_bpr_l2": ("cl-l2-fast", "l2-fast", "CL distribution; BPR L2 fixed"),
    "l2_with_user_cl": ("l2-fast", "baseline", "L2 triples; user CL fixed"),
    "l2_with_fast_cl": ("cl-l2-fast", "cl-fast", "L2 triples; independent Fast CL fixed"),
    "batch_coupling": ("shared-fast", "cl-l2-fast", "Shared vs independent Fast CL; BPR L2 fixed"),
    "raise_head_cl": ("head-up", "baseline", "Only user-head CL row weight raised; E[w]=1"),
    "lower_head_cl": ("head-down", "shared-fast", "Only user-head CL row weight lowered; E[w]=1"),
    "cl_vs_no_cl": ("baseline", "no-cl", "CL enabled; BPR and official L2 fixed"),
}


def metric_values(test):
    result = {"overall." + k: v for k, v in test["overall"].items() if k != "users"}
    for side in ("users", "items"):
        for group in ("head", "middle", "tail"):
            for metric, value in test[side][group].items():
                if "@" in metric and value is not None:
                    result[f"{side}.{group}.{metric}"] = value
    return result


def summarize_study(directory):
    """Works after download too: resolve runs relative to this study directory."""
    directory = Path(directory)
    plan = json.loads((directory / "study.json").read_text())
    records, epochs, configs = {}, {}, {}
    for item in plan["runs"]:
        name = item["name"]
        folder = directory / name / f"seed_{plan['seed']}"
        if not (folder / "result.json").exists():
            continue
        config = json.loads((folder / "config.json").read_text())
        if (config["source_fingerprint"] != plan["source_fingerprint"] or
                config["experiment_protocol"] != plan["experiment_protocol"] or
                any(config.get(k) != v for k, v in item["expected_config"].items())):
            raise ValueError(f"Configuration/source mismatch in {folder}")
        result = json.loads((folder / "result.json").read_text())
        if result["seed"] != plan["seed"] or result["variant"] != config["variant"]:
            raise ValueError(f"Run identity mismatch in {folder}")
        records[name], configs[name] = result, config
        epochs[name] = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()]
    if len({json.dumps(c["data"], sort_keys=True) for c in configs.values()}) > 1:
        raise ValueError("Study datasets differ")
    # Verify the streams that must stay fixed before attributing differences.
    stream_checks = {}
    checks = [("bpr_positives", "bpr_sample_fingerprint", list(PROFILES)),
              ("bpr_negatives", "bpr_negative_fingerprint", list(PROFILES)),
              ("user_cl", "cl_sample_fingerprint", ["baseline", "l2-fast", "head-up"]),
              ("independent_fast_cl", "cl_sample_fingerprint", ["cl-fast", "cl-l2-fast"]),
              ("shared_fast_cl", "cl_sample_fingerprint", ["shared-fast", "head-down"])]
    for label, members in [("official_l2", ["baseline", "cl-fast", "head-up", "no-cl"]),
                           ("bpr_l2", ["l2-fast", "cl-l2-fast", "shared-fast", "head-down"])]:
        for kind in ("sample", "negative"):
            checks.append((label + "_" + kind, "l2_" + kind + "_fingerprint", members))
    for label, key, members in checks:
        present = [name for name in members if name in epochs]
        common = min((len(epochs[name]) for name in present), default=0)
        for j in range(common):
            if len({epochs[name][j]["training"][key] for name in present}) != 1:
                raise ValueError(f"Unexpected stream change: {key} epoch={j+1}")
        stream_checks[label] = {"field": key, "runs": present, "common_epochs": common}
    effects = {}
    for name, (left, right, description) in CONTRASTS.items():
        if left not in records or right not in records:
            continue
        a, b = metric_values(records[left]["test"]), metric_values(records[right]["test"])
        deltas = {key: {"delta": a[key] - b[key],
                        "relative_percent": 100*(a[key]/b[key]-1) if b[key] else None} for key in a}
        common = min(len(epochs[left]), len(epochs[right]))
        matched = {}
        for j in range(common):
            x, y = metric_values(epochs[left][j]["test"]), metric_values(epochs[right][j]["test"])
            matched[str(j+1)] = {key: x[key]-y[key] for key in x}
        effects[name] = {"left": left, "right": right, "treatment": description,
                         "best_checkpoint_differences": deltas, "matched_epoch_deltas": matched}
    factorial = ["baseline", "cl-fast", "l2-fast", "cl-l2-fast"]
    interaction = None
    if all(name in records for name in factorial):
        a, b, c, d = [metric_values(records[name]["test"]) for name in factorial]
        interaction = {key: d[key]-b[key]-c[key]+a[key] for key in a}
    missing = [r["name"] for r in plan["runs"] if r["name"] not in records]
    output = {"study_protocol": STUDY_PROTOCOL, "source_fingerprint": plan["source_fingerprint"],
              "complete": not missing, "missing_runs": missing, "runs": records,
              "sampling": {name: {"policy": c["sampling_policy"], "weighting": c["user_cl_weighting"]}
                           for name, c in configs.items()},
              "stream_checks": stream_checks, "contrasts": effects, "cl_l2_interaction": interaction,
              "notes": ["One seed; no significance claim. Test Recall@20 selects checkpoints and early stopping.",
                        "Best-epoch contrasts include model-selection effects; inspect matched-epoch differences too.",
                        "CL reweighting changes whole InfoNCE rows, not positive attraction alone.",
                        "Component diagnostics use an unweighted fixed Fast probe, not each sampler's expected update."]}
    write_json(directory / "attribution.json", output)
    k = next(iter(configs.values()), {}).get("selection_k", 20)
    lines = ["# Yelp2018 归因实验", "", f"完成 {len(records)}/{len(plan['runs'])} 组；单seed={plan['seed']}，test Recall@{k}早停。", "",
             f"| 组别 | 最佳epoch | Recall@{k} | NDCG@{k} | 头部用户R@{k} |", "|---|---:|---:|---:|---:|"]
    for name, r in records.items():
        t = r["test"]
        lines.append(f"| {name} | {r['best_epoch']} | {t['overall'][f'recall@{k}']:.6f} | {t['overall'][f'ndcg@{k}']:.6f} | {t['users']['head'][f'recall@{k}']:.6f} |")
    lines += ["", "## 指定对照（左组减右组）", "", f"| 对照 | 整体R@{k}差值 | 头部用户R@{k}差值 |", "|---|---:|---:|"]
    for e in effects.values():
        m = e["best_checkpoint_differences"]
        lines.append(f"| {e['left']} − {e['right']} | {m[f'overall.recall@{k}']['delta']:+.6f} | {m[f'users.head.recall@{k}']['delta']:+.6f} |")
    lines += ["", "baseline/cl-fast/l2-fast/cl-l2-fast组成CL×L2的2×2对照；CL和L2的交互项见JSON。shared-fast与cl-l2-fast检验batch共享关系。head-up/head-down检验用户头部CL权重，整体用户CL期望权重固定为1，物品CL不变。baseline与no-cl检验CL相对纯BPR的影响。", "",
              "逐epoch匹配差值、采样指纹检查及实际权重见attribution.json；梯度分解见各组diagnostics.jsonl。log-partition含正例，不等于纯负例惩罚或几何uniformity。单seed与test选模不能证明统计显著性，也不能仅凭分数确认均匀性惩罚机制。"]
    (directory / "attribution.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-dir", type=Path, default=root / "data/yelp2018")
    p.add_argument("--output-dir", type=Path, default=root / "runs/attribution_v4/yelp2018")
    p.add_argument("--dataset", choices=("yelp2018", "custom"), default="yelp2018")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--runs", choices=PROFILES, nargs="+", default=list(PROFILES))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    args, extra = p.parse_known_args()
    reserved = {"--variant", "--cl-sampler", "--l2-source", "--user-cl-head-target", "--ssl-weight",
                "--cl-anchor-mode", "--ssl-side", "--decompose-cl", "--seeds", "--variants"}
    if any(token.split("=", 1)[0] in reserved for token in extra):
        p.error("Treatment flags are fixed by named runs; use --runs to select experiments")
    if len(set(args.runs)) != len(args.runs):
        p.error("Run names must be unique")
    if not all((args.data_dir / name).is_file() for name in ("train.txt", "test.txt")):
        p.error("data-dir must contain train.txt and test.txt")
    fingerprint = source_fingerprint()
    planned = []
    for name in args.runs:
        output = args.output_dir / name / f"seed_{args.seed}"
        options = ["--dataset", args.dataset, "--data-dir", str(args.data_dir.resolve()), "--output-dir", str(output.resolve()),
                   "--seed", str(args.seed), "--decompose-cl", *extra, *PROFILES[name]]
        training = training_parser().parse_args(options)
        validate(training)
        if training.diagnostics_every == 0:
            p.error("Attribution requires diagnostics; use --diagnostics-every >= 1")
        planned.append({"name": name, "command": [sys.executable, "-m", "rrec_experiments.train", *options],
                        "expected_config": jsonable(vars(training))})
    if args.dry_run:
        for item in planned:
            print(shlex.join(item["command"]))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"study_protocol": STUDY_PROTOCOL, "experiment_protocol": EXPERIMENT_PROTOCOL,
            "source_fingerprint": fingerprint, "seed": args.seed, "runs": planned}
    plan_path = args.output_dir / "study.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        p.error("Existing study plan differs; choose a new output directory")
    write_json(plan_path, plan)
    for item in planned:
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed during study; stopping to avoid mixed-version results")
        output = Path(item["expected_config"]["output_dir"])
        if args.skip_completed and (output / "result.json").exists():
            summarize_study(args.output_dir)  # Validate before accepting a saved run.
            print(f"Skipping {item['name']}", flush=True)
            continue
        print(f"Running attribution {item['name']} seed={args.seed}", flush=True)
        subprocess.run(item["command"], check=True)
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed during study; refusing mixed-version aggregation")
        summarize_study(args.output_dir)
    print(f"Completed {len(planned)} attribution runs: {args.output_dir / 'attribution.md'}", flush=True)


if __name__ == "__main__":
    main()
