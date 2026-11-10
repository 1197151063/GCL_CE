import torch
import torch.nn.functional as F

from .data import GROUP_NAMES, edge_codes, offsets_for
from .sampling import contains


@torch.no_grad()
def evaluate(model, graph, data, split="test", ks=(20, 50), batch_size=256, score="dot"):
    if split != "test":
        raise ValueError("Only test evaluation is supported")
    if min(ks) <= 0 or max(ks) > data.num_items:
        raise ValueError("Ranking K must be between 1 and number of items")
    model.eval()
    representation = model(graph)
    eu, ei = representation[:data.num_users], representation[data.num_users:]
    if score == "cosine":
        eu, ei = F.normalize(eu, dim=-1), F.normalize(ei, dim=-1)
    targets = data.test
    excluded = data.train
    excluded = excluded[:, torch.argsort(edge_codes(excluded, data.num_items))]
    offsets = offsets_for(excluded, data.num_users)
    target_codes = edge_codes(targets, data.num_items).to(eu.device)
    target_count = torch.bincount(targets[0], minlength=data.num_users)
    group_target_counts = [torch.bincount(targets[0, data.item_group[targets[1]] == g],
                                       minlength=data.num_users) for g in range(4)]
    user_result = {name: {"users": 0, **{f"{m}@{k}": 0. for k in ks for m in ("recall", "ndcg")}}
                   for name in ("overall", *GROUP_NAMES)}
    item_result = {name: {"target_users": int((group_target_counts[g] > 0).sum()),
                          "target_interactions": int(group_target_counts[g].sum()),
                          **{f"{m}@{k}": 0. for k in ks for m in ("recall_contribution", "conditional_recall")}}
                   for g, name in enumerate(GROUP_NAMES)}
    discount = 1 / torch.log2(torch.arange(2, max(ks) + 2, device=eu.device).float())
    item_groups_device = data.item_group.to(eu.device)
    for start in range(0, data.num_users, batch_size):
        end = min(start + batch_size, data.num_users)
        counts = target_count[start:end].to(eu.device)
        eligible = counts > 0
        if not eligible.any():
            continue
        scores = eu[start:end] @ ei.T
        lo, hi = int(offsets[start]), int(offsets[end])
        masked_edges = excluded[:, lo:hi].to(eu.device)
        scores[masked_edges[0] - start, masked_edges[1]] = -torch.inf
        ranked_scores, ranked = scores.topk(max(ks), dim=1)
        user_ids = torch.arange(start, end, device=eu.device).unsqueeze(1)
        # A fully exhausted candidate set must never produce a masked hit.
        hits = contains(target_codes, user_ids * data.num_items + ranked) & torch.isfinite(ranked_scores)
        user_groups = data.user_group[start:end].to(eu.device)
        masks = {"overall": eligible, **{name: eligible & (user_groups == g) for g, name in enumerate(GROUP_NAMES)}}
        for name, mask in masks.items():
            user_result[name]["users"] += int(mask.sum())
        for k in ks:
            recall = hits[:, :k].sum(1) / counts.clamp(min=1)
            dcg = (hits[:, :k] * discount[:k]).sum(1)
            ideal = discount[:k].cumsum(0)[counts.clamp(min=1, max=k) - 1]
            ndcg = dcg / ideal
            for name, mask in masks.items():
                user_result[name][f"recall@{k}"] += float(recall[mask].sum())
                user_result[name][f"ndcg@{k}"] += float(ndcg[mask].sum())
            for g, name in enumerate(GROUP_NAMES):
                g_hits = (hits[:, :k] & (item_groups_device[ranked[:, :k]] == g)).sum(1)
                g_counts = group_target_counts[g][start:end].to(eu.device)
                item_result[name][f"recall_contribution@{k}"] += float((g_hits / counts.clamp(min=1))[eligible].sum())
                mask = g_counts > 0
                item_result[name][f"conditional_recall@{k}"] += float((g_hits / g_counts.clamp(min=1))[mask].sum())
    for result in user_result.values():
        n = result["users"]
        for key in result:
            if key != "users":
                result[key] = result[key] / n if n else None
    eligible_count = user_result["overall"]["users"]
    for result in item_result.values():
        for k in ks:
            result[f"recall_contribution@{k}"] = result[f"recall_contribution@{k}"] / eligible_count if eligible_count else None
            n = result["target_users"]
            result[f"conditional_recall@{k}"] = result[f"conditional_recall@{k}"] / n if n else None
    return {"overall": user_result.pop("overall"), "users": user_result, "items": item_result}
