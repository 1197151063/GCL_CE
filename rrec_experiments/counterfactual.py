"""Measure downstream effects of reallocating Fast InfoNCE anchor weight."""
import argparse
import copy
from itertools import permutations
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .data import GROUP_NAMES, RecData, canonical_edges, edge_codes, groups_by_degree, read_edges
from .metrics import evaluate_targets
from .model import SGL, normalized_graph
from .sampling import generator, sample_fingerprint
from .train import (append_json, epoch_samples, parser as training_parser, source_fingerprint,
                    source_revision, train_epoch, validate, write_json)

GROUPS = GROUP_NAMES[:3]
STREAMS = ("bpr_sample_fingerprint", "bpr_negative_fingerprint", "cl_sample_fingerprint",
           "l2_sample_fingerprint", "l2_negative_fingerprint")


def split_probe(path, fraction, seed):
    """Use train.txt alone. Keep at least one fit edge per active user and item."""
    if not 0 < fraction < 1:
        raise ValueError("Probe fraction must be in (0, 1)")
    edges = read_edges(path)
    nu, ni = (int(edges[i].max()) + 1 for i in (0, 1))
    edges = canonical_edges(edges, ni)
    order = torch.randperm(edges.size(1), generator=generator(seed, "counterfactual_probe"))
    by_item = order[torch.argsort(edges[1, order], stable=True)]
    sorted_items = edges[1, by_item]
    first = torch.cat([torch.ones(1, dtype=torch.bool), sorted_items[1:] != sorted_items[:-1]])
    protected = torch.zeros(edges.size(1), dtype=torch.bool)
    protected[by_item[first]] = True
    candidates = order[~protected[order]]
    candidates = candidates[torch.argsort(edges[0, candidates], stable=True)]
    users = edges[0, candidates]
    counts = torch.bincount(users, minlength=nu)
    rank = torch.arange(candidates.numel()) - (counts.cumsum(0) - counts)[users]
    original_degree = torch.bincount(edges[0], minlength=nu)
    quota = torch.minimum((original_degree.double() * fraction).long().clamp(min=1),
                          (original_degree - 1).clamp(min=0))
    holdout = torch.zeros_like(protected)
    holdout[candidates[rank < quota[users]]] = True
    train, probe = edges[:, ~holdout], edges[:, holdout]
    if not probe.size(1):
        raise ValueError("No eligible probe interactions after preserving graph coverage")
    ud = torch.bincount(train[0], minlength=nu)
    idg = torch.bincount(train[1], minlength=ni)
    codes = edge_codes(train, ni)
    fit = RecData(train, torch.empty((2, 0), dtype=torch.long), nu, ni, ud, idg,
                  groups_by_degree(ud), groups_by_degree(idg), codes, sample_fingerprint(codes))
    return fit, probe


class BudgetTransfer:
    """Move equal mass in either direction; keep each batch's side budget fixed."""

    def __init__(self, data, side, donor, recipient, fraction):
        if side not in ("user", "item") or donor not in range(3) or recipient not in range(3) or donor == recipient:
            raise ValueError("Transfer needs a side and two distinct active degree groups")
        if not 0 <= fraction < 1:
            raise ValueError("Transfer fraction must be in [0, 1)")
        self.side = int(side == "item")
        self.groups = data.item_group if self.side else data.user_group
        self.donor, self.recipient, self.fraction = donor, recipient, fraction
        self.reset()

    def reset(self):
        self.base = torch.zeros(3, dtype=torch.long)
        self.moved = 0.
        self.skipped_batches = 0

    def __call__(self, edges):
        group = self.groups[edges[self.side]]
        counts = torch.bincount(group, minlength=3)
        if counts.numel() != 3:
            raise ValueError("A cold node cannot occur in the fit CL stream")
        self.base += counts
        src, dst = int(counts[self.donor]), int(counts[self.recipient])
        weights = None
        if src and dst and self.fraction:
            moved = self.fraction * min(src, dst)
            table = torch.ones(3)
            table[self.donor] -= moved / src
            table[self.recipient] += moved / dst
            weights = table[group]
            self.moved += moved
        elif not (src and dst):
            self.skipped_batches += 1
        return (None, weights) if self.side else (weights, None)

    def report(self):
        effective = self.base.double()
        effective[self.donor] -= self.moved
        effective[self.recipient] += self.moved
        total = int(self.base.sum())
        return {"anchor_rows": total, "transferred_weight": self.moved,
                "transferred_share": self.moved / total if total else None,
                "batches_without_both_groups": self.skipped_batches,
                "original_mass": dict(zip(GROUPS, self.base.tolist())),
                "effective_mass": dict(zip(GROUPS, effective.tolist()))}


class HeadShare:
    """Set the user-head loss mass on an unchanged Fast stream."""

    def __init__(self, data, target):
        self.groups = data.user_group
        self.target = target
        self.fast_share = float((self.groups[data.train[0]] == 0).double().mean())
        if not 0 < self.fast_share < 1 or not 0 <= target < 1:
            raise ValueError("Head dose needs both head/non-head anchors and a target in [0, 1)")
        self.head_weight = target / self.fast_share
        self.other_weight = (1 - target) / (1 - self.fast_share)
        self.table = torch.tensor([self.head_weight, self.other_weight, self.other_weight])
        self.reset()

    def reset(self):
        self.counts = torch.zeros(3, dtype=torch.long)
        self.min_batch_mean = float("inf")
        self.max_batch_mean = 0.

    def __call__(self, edges):
        group = self.groups[edges[0]]
        self.counts += torch.bincount(group, minlength=3)
        weights = self.table[group]
        average = float(weights.mean())
        self.min_batch_mean = min(average, self.min_batch_mean)
        self.max_batch_mean = max(average, self.max_batch_mean)
        return weights, None

    def report(self):
        counts = self.counts.double()
        mass = counts * self.table.double()
        return {"target_head_share": self.target, "fast_epoch_head_share": self.fast_share,
                "achieved_head_share": float(mass[0] / mass.sum()) if mass.sum() else None,
                "mean_weight": float(mass.sum() / counts.sum()),
                "batch_mean_weight_range": [self.min_batch_mean, self.max_batch_mean],
                "head_weight": self.head_weight, "non_head_weight": self.other_weight,
                "original_mass": dict(zip(GROUPS, counts.tolist())),
                "effective_mass": dict(zip(GROUPS, mass.tolist()))}


class UserCLProbe:
    """Same views and anchor rows in all dose branches; no updates or test labels."""

    def __init__(self, data, args, device, seed, epoch):
        self.groups = data.user_group.to(device)
        self.n_users = data.num_users
        self.temperature = args.temperature
        self.chunk_size = args.anchor_chunk_size
        self.views = [normalized_graph(data, device, args.drop_ratio,
                                       generator(seed, epoch, tag), args.self_loops)
                      for tag in ("dose_probe_view1", "dose_probe_view2")]
        indices = torch.randperm(data.num_edges, generator=generator(seed, epoch, "dose_probe_anchors"))
        self.anchors = data.train[0, indices[:args.diagnostic_batch_size]].to(device)

    @torch.no_grad()
    def __call__(self, model):
        a, b = (F.normalize(model(view)[:self.n_users], dim=1) for view in self.views)
        cosine = (a * b).sum(1)
        incoming = torch.zeros_like(cosine)
        for ids in self.anchors.split(self.chunk_size):
            p = (a[ids] @ b.T / self.temperature).softmax(1)
            p.scatter_(1, ids[:, None], 0.)
            incoming += p.sum(0)
        incoming /= self.anchors.numel()
        return {name: {"positive_cosine": float(cosine[self.groups == i].mean()),
                       "negative_key_mass": float(incoming[self.groups == i].sum()),
                       "negative_key_mass_per_user": float(incoming[self.groups == i].mean())}
                for i, name in enumerate(GROUPS)}


def fork_state(model, optimizer):
    # A joint deepcopy remaps Adam's parameter keys to the copied model.
    return copy.deepcopy((model, optimizer))


def run_branch(model, optimizer, graph, data, probe, args, epoch, horizons, transfer, output, label,
               cl_probe=None):
    branch, adam = fork_state(model, optimizer)
    observations, streams, budgets, diagnostics = {}, [], {}, {}

    def assess(tag, current):
        observations[tag] = evaluate_targets(current, graph, data, probe, args.ks,
                                             args.eval_batch_size, args.score)
        if transfer is not None:
            budgets[tag] = transfer.report()
        if cl_probe is not None:
            diagnostics[tag] = cl_probe(current)

    for offset in range(1, max(horizons) + 1):
        if transfer is not None:
            transfer.reset()
        started = time.monotonic()
        log = train_epoch(branch, adam, graph, data, args, epoch + offset, anchor_reweighter=transfer,
                          after_first_step=(lambda current: assess("step_1", current)) if offset == 1 else None)
        streams.append({key: log[key] for key in STREAMS})
        if offset in horizons:
            assess(f"epoch_{offset}", branch)
        entry = {"fork_epoch": epoch, "branch": label, "epoch_offset": offset,
                 "training": log, "seconds": time.monotonic() - started}
        if transfer is not None:
            entry["budget"] = transfer.report()
        append_json(output / "branches.jsonl", entry)
        print(f"fork={epoch} branch={label} epoch_offset={offset}/{max(horizons)}", flush=True)
    return {"probe_metrics": observations, "sample_streams": streams, "budget": budgets,
            "user_cl_probe": diagnostics}


def recall_delta(branch, baseline, k):
    metric = f"recall@{k}"
    observed = {"overall": branch["overall"][metric], **{g: branch["users"][g][metric] for g in GROUPS}}
    reference = {"overall": baseline["overall"][metric], **{g: baseline["users"][g][metric] for g in GROUPS}}
    return {g: observed[g] - reference[g] if observed[g] is not None and reference[g] is not None else None
            for g in observed}


def write_summary(output, result, k):
    write_json(output / "influence.json", result)
    is_dose = result["study"] == "head-dose"
    lines = ["# 头部用户 CL 权重扫描" if is_dose else "# CL 权重转移实验", "",
             "差值为干预分支减同起点 Fast 对照。只用原训练集留出的探针交互，未读取 test.txt。"]
    lines += ["用户侧每整轮总权重不变，batch 平均权重可以波动；物品侧不变。" if is_dose else
              "每个 batch 的用户侧、物品侧总权重各自不变；实际转移比例见 JSON。", ""]
    for epoch, fork in result["forks"].items():
        for horizon, columns in fork["matrix"].items():
            if is_dose:
                lines += [f"## 第 {epoch} 轮分叉 · {horizon} · Recall@{k}", "",
                          "| 目标头部权重 | 整体 R | 头部用户 R | 中部用户 R | 尾部用户 R | 尾部物品条件 R |",
                          "|---|---:|---:|---:|---:|---:|"]
                metric = f"recall@{k}"
                for name, branch in [("Fast", fork["baseline"]), *fork["transfers"].items()]:
                    t = branch["probe_metrics"][horizon]
                    values = [t["overall"][metric], *(t["users"][g][metric] for g in GROUPS),
                              t["items"]["tail"][f"conditional_recall@{k}"]]
                    lines.append("| " + name + " | " + " | ".join("—" if v is None else f"{v:.6f}" for v in values) + " |")
                lines += ["", "| 目标头部权重 | 实际头部份额 | 头部正对余弦 | 头部负 key 概率质量 |",
                          "|---|---:|---:|---:|"]
                for name, branch in [("Fast", fork["baseline"]), *fork["transfers"].items()]:
                    d = branch["user_cl_probe"][horizon]["head"]
                    actual = branch["budget"].get(horizon, {}).get("achieved_head_share")
                    if name == "Fast":
                        actual = fork["baseline_head_share"][horizon]
                    formatted = f"{actual:.2%}" if actual is not None else "—"
                    lines.append(f"| {name} | {formatted} | {d['positive_cosine']:.6f} | {d['negative_key_mass']:.6f} |")
                lines.append("")
                continue
            lines += [f"## 第 {epoch} 轮分叉 · {horizon} · ΔRecall@{k}", "",
                      "| 评估用户 | " + " | ".join(columns) + " |",
                      "|---|" + "|".join("---:" for _ in columns) + "|"]
            for group in ("overall", *GROUPS):
                values = [v[group] for v in columns.values()]
                lines.append("| " + group + " | " + " | ".join("—" if v is None else f"{v:+.6f}" for v in values) + " |")
            lines.append("")
    path = output / "influence.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(settings, args):
    validate(args)
    if not args.ssl_weight or args.cl_anchor_mode != "repeated" or args.ssl_side != "both":
        raise ValueError("Counterfactual study requires active CL on both sides with repeated anchors")
    if args.variant != "fast" or args.cl_sampler != "fast" or args.user_cl_head_target != "none":
        raise ValueError("Counterfactual study fixes independent Fast CL/BPR and uses its own weights")
    if not 0 < settings.transfer_fraction < 1:
        raise ValueError("Transfer fraction must be in (0, 1)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.strict_determinism)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data, probe = split_probe(args.data_dir / "train.txt", settings.probe_fraction, settings.probe_seed)
    if max(args.ks) > data.num_items:
        raise ValueError("max(ks) exceeds the number of items")
    for side in (("user",) if settings.study == "head-dose" else settings.sides):
        groups = data.user_group if side == "user" else data.item_group
        if not all((groups == g).any() for g in range(3)):
            raise ValueError(f"Counterfactual matrix needs all three active {side} groups")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = SGL(data.num_users, data.num_items, args.dim, args.layers, args.init_std, args.init).to(device)
    graph = normalized_graph(data, device, self_loops=args.self_loops)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fingerprint = source_fingerprint()
    config = {"protocol": "cl_head_dose_v1" if settings.study == "head-dose" else "cl_budget_transfer_v1",
              "study": settings.study, "source_fingerprint": fingerprint,
              "repository_commit": source_revision(), "torch_version": torch.__version__, "device": device,
              "training": vars(args), "fork_epochs": settings.fork_epochs, "horizons": settings.horizons,
              "sides": settings.sides, "transfer_fraction": settings.transfer_fraction,
              "head_shares": settings.head_shares,
              "fast_user_head_share": float((data.user_group[data.train[0]] == 0).double().mean()),
              "probe_seed": settings.probe_seed, "probe_fraction": settings.probe_fraction,
              "probe_edges": probe.size(1), "probe_fingerprint": sample_fingerprint(edge_codes(probe, data.num_items)),
              "fit_edges": data.num_edges, "fit_fingerprint": data.fingerprint, "official_test_read": False}
    write_json(output / "config.json", config)
    result = {"study": settings.study, "complete": False, "source_fingerprint": fingerprint, "forks": {}}
    for epoch in range(max(settings.fork_epochs) + 1):
        if epoch:
            log = train_epoch(model, optimizer, graph, data, args, epoch)
            append_json(output / "warmup.jsonl", {"epoch": epoch, "training": log})
            print(f"warmup epoch={epoch}", flush=True)
        if epoch not in settings.fork_epochs:
            continue
        initial = evaluate_targets(model, graph, data, probe, args.ks, args.eval_batch_size, args.score)
        cl_probe = UserCLProbe(data, args, device, settings.probe_seed, epoch) if settings.study == "head-dose" else None
        baseline = run_branch(model, optimizer, graph, data, probe, args, epoch, settings.horizons,
                              None, output, "fast", cl_probe)
        fork = {"before_transfer": initial, "baseline": baseline, "transfers": {},
                "matrix": {key: {} for key in baseline["probe_metrics"]}}
        if cl_probe is not None:
            # For the step-1 report, use the same first batch rather than the epoch average.
            _, indices, _ = epoch_samples(data, args, epoch + 1)
            fork["baseline_head_share"] = {
                "step_1": float((data.user_group[data.train[0, indices[:args.batch_size]]] == 0).double().mean()),
                **{f"epoch_{offset}": config["fast_user_head_share"] for offset in settings.horizons},
            }
        result["forks"][str(epoch)] = fork
        if settings.study == "head-dose":
            branches = [(f"head={target}", HeadShare(data, target)) for target in settings.head_shares]
        else:
            branches = [(f"{side}:{GROUPS[donor]}→{GROUPS[recipient]}",
                         BudgetTransfer(data, side, donor, recipient, settings.transfer_fraction))
                        for side in settings.sides for donor, recipient in permutations(range(3), 2)]
        for name, transfer in branches:
            branch = run_branch(model, optimizer, graph, data, probe, args, epoch, settings.horizons,
                                transfer, output, name, cl_probe)
            if branch["sample_streams"] != baseline["sample_streams"]:
                raise RuntimeError(f"Sample streams differ from Fast in {name}")
            if source_fingerprint() != fingerprint:
                raise RuntimeError("Source changed during the counterfactual study")
            fork["transfers"][name] = branch
            for horizon, observed in branch["probe_metrics"].items():
                fork["matrix"][horizon][name] = recall_delta(observed, baseline["probe_metrics"][horizon], args.selection_k)
            write_summary(output, result, args.selection_k)
    result["complete"] = True
    write_summary(output, result, args.selection_k)
    return result


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                epilog="Training options such as --device, --batch-size and --ks are forwarded.")
    p.add_argument("--data-dir", type=Path, default=root / "data/yelp2018")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--study", choices=("transfer", "head-dose"), default="transfer")
    p.add_argument("--fork-epochs", type=int, nargs="+")
    p.add_argument("--head-shares", type=float, nargs="+")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--sides", choices=("user", "item"), nargs="+", default=["user", "item"])
    p.add_argument("--probe-fraction", type=float, default=.1)
    p.add_argument("--probe-seed", type=int, default=20261113)
    p.add_argument("--transfer-fraction", type=float, default=.25)
    p.add_argument("--dry-run", action="store_true")
    settings, extra = p.parse_known_args(argv)
    dose = settings.study == "head-dose"
    if settings.output_dir is None:
        settings.output_dir = root / "runs" / ("head_dose_v1" if dose else "counterfactual_v1") / "yelp2018"
    if settings.fork_epochs is None:
        settings.fork_epochs = [0, 10, 20] if dose else [10, 20]
    if settings.head_shares is not None and not dose:
        p.error("--head-shares requires --study head-dose")
    settings.head_shares = settings.head_shares if settings.head_shares is not None else (
        [0., .1, .2, .3, .4, .5, .6] if dose else [])
    if (len(set(settings.head_shares)) != len(settings.head_shares) or
            any(not 0 <= share < 1 for share in settings.head_shares)):
        p.error("head-shares must be distinct and in [0, 1)")
    settings.head_shares.sort()
    for name in ("fork_epochs", "horizons"):
        values = getattr(settings, name)
        if min(values) < (0 if name == "fork_epochs" else 1) or len(set(values)) != len(values):
            p.error(f"{name} must contain distinct valid epochs")
        values.sort()
    if len(set(settings.sides)) != len(settings.sides):
        p.error("sides must be distinct")
    fixed = {"--variant", "--cl-sampler", "--user-cl-head-target", "--ssl-side", "--cl-anchor-mode",
             "--decompose-cl", "--epochs", "--patience", "--eval-every", "--selection-objective", "--diagnostics-every"}
    if any(token.split("=", 1)[0] in fixed for token in extra):
        p.error("Use the counterfactual options; sampling, diagnostics and epoch schedule are fixed")
    args = training_parser().parse_args(["--data-dir", str(settings.data_dir), "--output-dir", str(settings.output_dir),
                                        "--variant", "fast", "--cl-sampler", "fast", "--diagnostics-every", "0",
                                        "--patience", "0", "--epochs", str(max(settings.fork_epochs) + max(settings.horizons)),
                                        *extra])
    validate(args)
    if settings.dry_run:
        print(f"Fork epochs {settings.fork_epochs}; evaluate after 1 step and epochs {settings.horizons}; "
              f"{1 + (len(settings.head_shares) if dose else 6 * len(settings.sides))} branches per fork; "
              f"read {args.data_dir / 'train.txt'} only")
        if dose:
            print(f"User head CL mass: {settings.head_shares}; item CL unchanged; Fast is the control")
        return
    run(settings, args)
