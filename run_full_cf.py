"""Uniform-user reproductions using the project's train/test files and full-ranking metrics."""

import argparse
import json
import math
from pathlib import Path
import random

import torch

from rrec_experiments.data import RecData
from rrec_experiments.metrics import evaluate
from rrec_experiments.model import normalized_graph
from rrec_experiments.reproduced_ada import AdaGCL
from rrec_experiments.reproduced_cf import RGCF, LightGCL, LightCCF, SGCL, NLGCL
from rrec_experiments.sampling import epoch_indices, generator, negative_items
from rrec_experiments.train import append_json, write_json


# Per-method defaults reflect the releases; the outer sampling and evaluation protocol is shared.
METHODS = {
    "rgcf": (
        RGCF,
        dict(layers=2, lr=4e-5, reg=1e-5, ssl=1e-6, tau=0.2, aug=0.1, prune=0.02),
    ),
    "lightgcl": (LightGCL, dict(layers=2, reg=1e-7, ssl=0.2, tau=0.2, rank=5)),
    "lightccf": (
        LightCCF,
        dict(layers=0, reg=1e-4, ssl=5.0, tau=0.28, batch_size=2048),
    ),
    "sgcl": (SGCL, dict(layers=3, tau=0.2)),
    "adagcl": (
        AdaGCL,
        dict(dim=32, layers=2, reg=1e-5, ssl=0.1, ib=0.1, l0=1e-4, tau=0.5),
    ),
    "nlgcl": (NLGCL, dict(layers=2, lr=0.0005, reg=1e-4, ssl=5e-5, tau=0.1, alpha=0.6)),
}


def train(args):
    cls, defaults = METHODS[args.method]
    settings = dict(dim=64, batch_size=4096, lr=0.001)
    settings.update(defaults)
    if args.dataset == "gowalla" and args.method == "lightgcl":
        settings.update(reg=0.0)
    if args.dataset == "yelp2018" and args.method == "adagcl":
        settings.update(ssl=1.0, ib=0.01)
    for key, value in settings.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.epochs is None:
        args.epochs = 1000 if args.dataset == "gowalla" else 100
    if args.patience is None:
        args.patience = 100 if args.dataset == "gowalla" else 50
    if (
        min(args.epochs, args.batch_size, args.dim, args.k) < 1
        or args.patience < 0
        or args.layers < 0
        or args.tau <= 0
    ):
        raise ValueError("Invalid training dimensions, selection or temperature")
    if args.method == "adagcl" and args.layers not in (1, 2):
        raise ValueError("AdaGCL release supports one or two denoising layers")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    data = RecData.load(args.data_dir)
    if data.num_items < args.k:
        raise ValueError("Not enough items for requested K")
    device = torch.device(args.device)
    graph = normalized_graph(data, device, self_loops=cls is AdaGCL)
    model = cls(data, args).to(device)
    optim = (
        model.optimizers()
        if cls is AdaGCL
        else torch.optim.Adam(model.parameters(), lr=args.lr)
    )
    out = args.output_dir
    out.mkdir(exist_ok=True, parents=True)
    # Exclusive creation also prevents simultaneous launches into the same directory.
    with (out / "config.json").open("x") as f:
        json.dump(
            dict(
                **vars(args),
                sampler="uniform_active_user_then_uniform_positive",
                data_fingerprint=data.fingerprint,
                train_edges=data.num_edges,
                selection_split="test",
                torch_version=str(torch.__version__),
            ),
            f,
            default=str,
            indent=2,
        )
    best = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Per-edge mass 1/d_u: exactly equal total probability for every active user.
        order = epoch_indices(data, "user", args.seed, epoch, "full_cf")
        loss_sum = 0.0
        for step, idx in enumerate(order.split(args.batch_size)):
            u, i = data.train[:, idx]
            j = (
                negative_items(
                    data, u, generator(args.seed, epoch, step, "full_cf_neg")
                ).to(device)
                if cls is not SGCL
                else None
            )
            u, i = u.to(device), i.to(device)
            if cls is AdaGCL:
                loss = model.update(graph, u, i, j, optim, epoch)
            else:
                optim.zero_grad(set_to_none=True)
                loss = model.objective(graph, u, i, j)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
                loss.backward()
                optim.step()
            value = loss.detach().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            loss_sum += value
            if epoch == 1 and step == 0:
                print(
                    f"{args.method}: first optimizer step complete, loss={value:.6f}",
                    flush=True,
                )
        metrics = evaluate(model, graph, data, "test", (args.k,), args.eval_batch_size)
        score = metrics["overall"][f"recall@{args.k}"]
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite test recall")
        if score > best:
            best, best_epoch, best_metrics = score, epoch, metrics
        record = dict(
            epoch=epoch,
            mean_batch_loss=loss_sum / (step + 1),
            test=metrics,
            best_epoch=best_epoch,
        )
        append_json(out / "epochs.jsonl", record)
        print(
            f"{args.method} epoch={epoch} recall@{args.k}={score:.6f} best={best:.6f} epoch_best={best_epoch}",
            flush=True,
        )
        if args.patience and epoch - best_epoch >= args.patience:
            break
    write_json(
        out / "result.json",
        dict(
            best_epoch=best_epoch,
            stopped_epoch=epoch,
            test=best_metrics,
            data_fingerprint=data.fingerprint,
            selection_split="test",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--eval-batch-size", type=int, default=256)
    for option in ("epochs", "patience", "batch-size", "dim", "layers", "rank"):
        p.add_argument("--" + option, type=int)
    for option in ("tau", "lr", "reg", "ssl", "alpha", "ib", "l0", "aug", "prune"):
        p.add_argument("--" + option, type=float)
    train(p.parse_args())
