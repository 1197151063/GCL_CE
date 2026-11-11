"""Independent random streams keep each intervention reproducible."""
import hashlib
import math

import torch


def generator(seed: int, *parts) -> torch.Generator:
    token = repr((seed, *parts)).encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return torch.Generator(device="cpu").manual_seed(value % (2**63 - 1))


def epoch_indices(data, sampler: str, seed: int, epoch: int, stream: str):
    """Fast: permutation; Full: 1/(degree+1); user: official uniform-user law."""
    rng = generator(seed, epoch, stream)
    if sampler == "fast":
        return torch.randperm(data.num_edges, generator=rng)
    if sampler == "full":
        weights = 1.0 / (data.user_degree[data.train[0]].double() + 1.0)
        return torch.multinomial(weights, data.num_edges, replacement=True, generator=rng)
    if sampler == "user":
        # Each user's d edges sum to weight 1, then pick its positive uniformly.
        weights = 1.0 / data.user_degree[data.train[0]].double()
        return torch.multinomial(weights, data.num_edges, replacement=True, generator=rng)
    raise ValueError(f"Unknown sampler: {sampler}")


def negative_items(data, users, rng):
    """Uniform unseen train items; test labels are never consulted."""
    users = users.cpu()
    if (data.user_degree[users] >= data.num_items).any():
        raise ValueError("A user has no available training-negative item.")
    negatives = torch.randint(data.num_items, users.shape, generator=rng)
    pending = torch.arange(users.numel())
    while pending.numel():
        codes = users[pending] * data.num_items + negatives[pending]
        seen = contains(data.train_codes, codes)
        pending = pending[seen]
        if pending.numel():
            negatives[pending] = torch.randint(data.num_items, (pending.numel(),), generator=rng)
    return negatives


def contains(sorted_values, queries):
    if sorted_values.numel() == 0:
        return torch.zeros_like(queries, dtype=torch.bool)
    positions = torch.searchsorted(sorted_values, queries)
    return (positions < sorted_values.numel()) & (
        sorted_values[positions.clamp(max=sorted_values.numel() - 1)] == queries
    )


def batch_occupancy(data, bpr_edges, cl_edges, anchor_mode="repeated"):
    result = {}
    for side, groups in [(0, data.user_group), (1, data.item_group)]:
        name = "user" if side == 0 else "item"
        ids = bpr_edges[side]
        anchors = cl_edges[side] if anchor_mode == "repeated" else torch.unique(cl_edges[side])
        result[f"{name}_unique_bpr"] = float(ids.unique().numel())
        result[f"{name}_unique_cl"] = float(cl_edges[side].unique().numel())
        result[f"{name}_head_bpr_share"] = float((groups[ids] == 0).float().mean())
        result[f"{name}_head_cl_anchor_share"] = float((groups[anchors] == 0).float().mean())
    return result


def sample_fingerprint(indices):
    return hashlib.sha256(indices.contiguous().numpy().tobytes()).hexdigest()


def num_batches(data, batch_size):
    return math.ceil(data.num_edges / batch_size)
