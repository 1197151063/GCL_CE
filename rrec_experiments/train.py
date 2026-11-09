import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import subprocess
import time

import torch

from .data import RecData
from .diagnostics import gradient_diagnostics
from .metrics import evaluate
from .model import SGL, normalized_graph, contrastive_loss
from .sampling import batch_occupancy, epoch_indices, generator, negative_items, num_batches
from .presets import PRESETS

VARIANTS = {
    "A": ("fast", "fast", False, True),
    "B": ("full", "full", False, True),
    "C": ("fast", "full", False, True),
    "D": ("full", "fast", False, True),
    "shared-fast": ("fast", "fast", True, True),
    "shared-full": ("full", "full", True, True),
    "bpr-fast": ("fast", "fast", False, False),
    "bpr-full": ("full", "full", False, False),
}


def parser():
    p = argparse.ArgumentParser(description="SGL sampling experiment; see README for controlled variants", allow_abbrev=False)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=["custom", *PRESETS], default="custom", help="Dataset label; profile applied by run_experiments.sh")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="A")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=20261109)
    p.add_argument("--validation-ratio", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--init-std", type=float, default=.01)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--decay", type=float, default=.0001)
    p.add_argument("--ssl-weight", type=float, default=.2)
    p.add_argument("--ssl-side", choices=("both", "user", "item"), default="both")
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--drop-ratio", type=float, default=.1)
    p.add_argument("--self-loops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-chunk-size", type=int, default=256)
    p.add_argument("--l2-source", choices=("independent", "bpr"), default="independent")
    p.add_argument("--ks", type=int, nargs="+", default=[20, 50])
    p.add_argument("--selection-k", type=int, default=20)
    p.add_argument("--score", choices=("dot", "cosine"), default="cosine")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=0, help="0: fixed epoch budget; otherwise validation checks without improvement")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--diagnostics-every", type=int, default=25, help="epochs; 0 disables gradient diagnostics")
    p.add_argument("--diagnostic-batch-size", type=int, default=256)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...; MPS sparse is not supported")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--strict-determinism", action="store_true")
    return p


def validate(args):
    for name in ("epochs", "batch_size", "dim", "init_std", "lr", "temperature", "anchor_chunk_size",
                 "eval_every", "eval_batch_size", "diagnostic_batch_size", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.layers < 0 or args.decay < 0 or args.ssl_weight < 0 or args.diagnostics_every < 0 or args.patience < 0:
        raise ValueError("layers, decay, ssl_weight, diagnostics_every and patience must be nonnegative")
    if not 0 <= args.drop_ratio < 1:
        raise ValueError("drop_ratio must be in [0, 1)")
    if not args.ks or min(args.ks) < 1 or args.selection_k not in args.ks:
        raise ValueError("Positive ks must contain selection_k")
    if args.device.startswith("mps"):
        raise ValueError("Use CPU or CUDA: sparse graph operations are not supported here on MPS")


def jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")


def make_batch(data, indices, start, size):
    return data.train[:, indices[start:start + size]]


def source_revision():
    root = Path(__file__).resolve().parent.parent
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() == root:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    recorded = root / "SOURCE_COMMIT"
    return recorded.read_text().strip() if recorded.is_file() else None


def train_epoch(model, optimizer, graph, data, args, epoch):
    model.train()
    device = model.embedding.weight.device
    bpr_sampler, cl_sampler, shared, _ = VARIANTS[args.variant]
    bpr_indices = epoch_indices(data, bpr_sampler, args.seed, epoch, "bpr")
    cl_indices = (bpr_indices if shared else epoch_indices(data, cl_sampler, args.seed, epoch, "cl")) if args.ssl_weight else None
    l2_indices = epoch_indices(data, "fast", args.seed, epoch, "l2") if args.l2_source == "independent" and args.decay else None
    if args.ssl_weight:
        view1 = normalized_graph(data, device, args.drop_ratio, generator(args.seed, epoch, "view1"), args.self_loops)
        view2 = normalized_graph(data, device, args.drop_ratio, generator(args.seed, epoch, "view2"), args.self_loops)
    totals = defaultdict(float)
    steps = num_batches(data, args.batch_size)
    for step, start in enumerate(range(0, data.num_edges, args.batch_size)):
        edges = make_batch(data, bpr_indices, start, args.batch_size)
        neg = negative_items(data, edges[0], generator(args.seed, epoch, step, "bpr_negatives"))
        cl_edges = make_batch(data, cl_indices, start, args.batch_size) if cl_indices is not None else None
        optimizer.zero_grad(set_to_none=True)
        bpr = model.bpr_terms(model(graph), edges.to(device), neg.to(device)).mean()
        reg = bpr.new_zeros(())
        if args.decay:
            if args.l2_source == "independent":
                le = make_batch(data, l2_indices, start, args.batch_size)
                ln = negative_items(data, le[0], generator(args.seed, epoch, step, "l2_negatives"))
                reg = args.decay * model.l2(le.to(device), ln.to(device))
            else:
                reg = args.decay * model.l2(edges.to(device), neg.to(device))
        # Backpropagate recommendation first to bound simultaneous graph memory.
        loss = bpr + reg
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite recommendation loss at epoch={epoch} step={step}")
        loss.backward()
        cl = bpr.new_zeros(())
        if args.ssl_weight:
            cl = args.ssl_weight * contrastive_loss(model, view1, view2, cl_edges.to(device),
                                                     args.temperature, args.anchor_chunk_size, args.ssl_side)
            if not torch.isfinite(cl):
                raise FloatingPointError(f"Nonfinite CL loss at epoch={epoch} step={step}")
            cl.backward()
        optimizer.step()
        totals["bpr_loss"] += float(bpr.detach())
        totals["l2_loss"] += float(reg.detach())
        totals["cl_loss"] += float(cl.detach())
        if cl_edges is not None:
            for key, value in batch_occupancy(data, edges, cl_edges).items():
                totals[key] += value
    result = {k: v / steps for k, v in totals.items()}
    result["optimizer_steps"] = steps
    return result


def run(args):
    validate(args)
    _, _, _, use_cl = VARIANTS[args.variant]
    if not use_cl:
        args.ssl_weight = 0.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.strict_determinism)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    data = RecData.load(args.data_dir, args.validation_ratio, args.split_seed)
    if max(args.ks) > data.num_items:
        raise ValueError("max(ks) exceeds number of dataset items")
    model = SGL(data.num_users, data.num_items, args.dim, args.layers, args.init_std).to(device)
    graph = normalized_graph(data, device, self_loops=args.self_loops)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    commit = source_revision()
    config = {**vars(args), "effective_device": str(device), "torch_version": torch.__version__,
              "repository_commit": commit, "data": data.summary(),
              "selection_metric": f"validation recall@{args.selection_k} + ndcg@{args.selection_k}"}
    write_json(args.output_dir / "config.json", config)
    best, best_epoch, failures, updates = -math.inf, 0, 0, 0
    begin = time.time()
    if args.diagnostics_every:
        append_json(args.output_dir / "diagnostics.jsonl", {"epoch": 0, **gradient_diagnostics(model, graph, data, args)})
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        training = train_epoch(model, optimizer, graph, data, args, epoch)
        updates += training["optimizer_steps"]
        record = {"epoch": epoch, "training": training, "cumulative_optimizer_steps": updates}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, graph, data, "validation", args.ks, args.eval_batch_size, args.score)
            record["validation"] = metrics
            score = metrics["overall"][f"recall@{args.selection_k}"] + metrics["overall"][f"ndcg@{args.selection_k}"]
            if score > best:
                best, best_epoch, failures = score, epoch, 0
                temporary = args.output_dir / "best.pt.tmp"
                torch.save({"model": model.state_dict(), "epoch": epoch, "selection_score": score}, temporary)
                temporary.replace(args.output_dir / "best.pt")
            else:
                failures += 1
        if args.diagnostics_every and epoch % args.diagnostics_every == 0:
            append_json(args.output_dir / "diagnostics.jsonl", {"epoch": epoch, **gradient_diagnostics(model, graph, data, args)})
        record["seconds"] = time.time() - started
        append_json(args.output_dir / "epochs.jsonl", record)
        message = f"{args.variant} seed={args.seed} epoch={epoch} bpr={training['bpr_loss']:.5f} cl={training['cl_loss']:.5f}"
        if "validation" in record:
            message += f" validation_score={score:.6f} best_epoch={best_epoch}"
        print(message, flush=True)
        if args.patience and failures >= args.patience:
            break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation = evaluate(model, graph, data, "validation", args.ks, args.eval_batch_size, args.score)
    test = evaluate(model, graph, data, "test", args.ks, args.eval_batch_size, args.score)
    if args.diagnostics_every:
        append_json(args.output_dir / "diagnostics.jsonl", {"epoch": best_epoch, "checkpoint": "best", **gradient_diagnostics(model, graph, data, args)})
    result = {"variant": args.variant, "seed": args.seed, "best_epoch": best_epoch,
              "stopped_epoch": epoch, "optimizer_steps": updates, "best_selection_score": best,
              "seconds": time.time() - begin, "validation": validation, "test": test,
              "data_fingerprint": data.fingerprint}
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"completed": str(args.output_dir), "test": test["overall"]}), flush=True)
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, FileExistsError, FloatingPointError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
