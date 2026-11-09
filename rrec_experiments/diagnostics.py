import math
import torch
import torch.nn.functional as F

from .data import GROUP_NAMES
from .model import contrastive_loss, normalized_graph
from .sampling import generator, negative_items


def sampled_uniformity(embedding, rng, pairs=2048):
    """log E exp(-2 ||normalized(x)-normalized(y)||^2); lower is more uniform."""
    n = embedding.size(0)
    if n < 2:
        return None
    left = torch.randint(n, (pairs,), generator=rng)
    right = (left + torch.randint(1, n, (pairs,), generator=rng)) % n
    x = F.normalize(embedding, dim=1)
    distance = (x[left.to(x.device)] - x[right.to(x.device)]).square().sum(1)
    return float(torch.logsumexp(-2 * distance, dim=0) - math.log(pairs))


def gradient_diagnostics(model, graph, data, args):
    """Same fixed probe for every run/checkpoint, including full graph gradients.

    Measures BPR vs weighted CL (no L2). These are diagnostics, not updates.
    """
    device = model.embedding.weight.device
    rng = generator(args.split_seed, "fixed_gradient_probe")
    ids = torch.randperm(data.num_edges, generator=rng)[:min(args.diagnostic_batch_size, data.num_edges)]
    edges = data.train[:, ids]
    negatives = negative_items(data, edges[0], generator(args.split_seed, "fixed_probe_negatives"))
    g1 = normalized_graph(data, device, args.drop_ratio,
                          generator(args.split_seed, "fixed_probe_view1"), args.self_loops)
    g2 = normalized_graph(data, device, args.drop_ratio,
                          generator(args.split_seed, "fixed_probe_view2"), args.self_loops)
    rec = model.bpr_terms(model(graph), edges.to(device), negatives.to(device)).mean()
    rec_grad, = torch.autograd.grad(rec, model.embedding.weight)
    raw_cl = contrastive_loss(model, g1, g2, edges.to(device), args.temperature,
                              args.anchor_chunk_size, args.ssl_side)
    if args.ssl_weight:
        cl_grad, = torch.autograd.grad(args.ssl_weight * raw_cl, model.embedding.weight)
    else:
        cl_grad = torch.zeros_like(rec_grad)
    with torch.no_grad():
        h = model(graph)
        result = {"bpr_loss": float(rec), "raw_cl_loss": float(raw_cl),
                  "weighted_cl_loss": float(args.ssl_weight * raw_cl),
                  "probe_edges": edges.size(1), "groups": {}}
        normalized = F.normalize(h, dim=1)
        result["positive_alignment"] = float((normalized[edges[0].to(device)] -
                                               normalized[(edges[1] + data.num_users).to(device)]).square().sum(1).mean())
        result["user_uniformity"] = sampled_uniformity(h[:data.num_users], generator(args.split_seed, "user_uniformity"))
        result["item_uniformity"] = sampled_uniformity(h[data.num_users:], generator(args.split_seed, "item_uniformity"))
        for side, offset, size, groups in [
            ("user", 0, data.num_users, data.user_group),
            ("item", data.num_users, data.num_items, data.item_group),
        ]:
            for g, name in enumerate(GROUP_NAMES):
                ids = torch.where(groups == g)[0].to(device) + offset
                if not ids.numel():
                    continue
                r, c = rec_grad[ids], cl_grad[ids]
                rn, cn = r.norm(dim=1), c.norm(dim=1)
                valid = (rn > 1e-12) & (cn > 1e-12)
                cosine = F.cosine_similarity(r[valid], c[valid], dim=1, eps=1e-12) if valid.any() else None
                ratio = cn[valid] / rn[valid] if valid.any() else None
                def quantile(value, q):
                    return float(torch.quantile(value, q)) if value is not None and value.numel() else None
                result["groups"][f"{side}_{name}"] = {
                    "nodes": ids.numel(), "valid_gradient_nodes": int(valid.sum()),
                    "rec_gradient_norm": float(r.norm()), "cl_gradient_norm": float(c.norm()),
                    "norm_ratio": float(c.norm() / r.norm()) if float(r.norm()) > 1e-12 else None,
                    "node_ratio_median": quantile(ratio, .5), "node_ratio_p90": quantile(ratio, .9),
                    "cosine_median": quantile(cosine, .5),
                    "conflict_fraction": float((cosine < 0).float().mean()) if cosine is not None else None,
                    "embedding_norm_mean": float(h[ids].norm(dim=1).mean()),
                    "within_group_uniformity": sampled_uniformity(h[ids], generator(args.split_seed, side, name, "uniformity")),
                }
    return result
