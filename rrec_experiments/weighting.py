"""User-anchor group reweighting, normalized under the actual sampling law."""
import torch


def user_sampling_mass(data, sampler):
    degree = data.user_degree.double()
    if sampler == "fast":
        mass = degree
    elif sampler == "user":
        mass = (degree > 0).double()
    elif sampler == "full":
        mass = degree / (degree + 1)
    else:
        raise ValueError(f"Unknown CL sampler: {sampler}")
    return mass / mass.sum()


def user_anchor_weights(data, sampler, target="none"):
    """Match head mass only; preserve within-group probabilities and E[w]=1.

    This reweights complete user InfoNCE rows, not item rows or candidate keys.
    It does not normalize separately in each minibatch.
    """
    base = user_sampling_mass(data, sampler)
    head = data.user_group == 0
    base_head = float(base[head].sum())
    if target == "none":
        return None, {"target": "none", "base_head_mass": base_head,
                      "target_head_mass": base_head, "expected_weight": 1.}
    desired_head = float(user_sampling_mass(data, target)[head].sum())
    if not 0 < base_head < 1 or not 0 < desired_head < 1:
        raise ValueError("Head CL reweighting requires active head and non-head users")
    weights = torch.full_like(base, (1 - desired_head) / (1 - base_head))
    weights[head] = desired_head / base_head
    assert torch.allclose((base * weights).sum(), torch.tensor(1., dtype=torch.double))
    return weights.float(), {"target": target, "base_head_mass": base_head,
                             "target_head_mass": desired_head, "expected_weight": float((base * weights).sum()),
                             "head_weight": float(weights[head][0]),
                             "nonhead_weight": float(weights[~head][0])}
