import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def normalized_graph(data, device, drop_ratio=0., rng=None, self_loops=False):
    edges = data.train
    if drop_ratio:
        keep = torch.randperm(edges.size(1), generator=rng)[:int(edges.size(1) * (1 - drop_ratio))]
        edges = edges[:, keep]
    users, items = edges[0], edges[1] + data.num_users
    row, col = torch.cat([users, items]), torch.cat([items, users])
    n = data.num_users + data.num_items
    if self_loops:
        nodes = torch.arange(n)
        row, col = torch.cat([row, nodes]), torch.cat([col, nodes])
    degree = torch.bincount(row, minlength=n).float()
    inv = degree.clamp(min=1).rsqrt()
    values = inv[row] * inv[col]
    return torch.sparse_coo_tensor(torch.stack([row, col]), values, (n, n)).coalesce().to(device)


class SGL(nn.Module):
    def __init__(self, num_users, num_items, dim=64, layers=3, init_std=.01, init="xavier_uniform"):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.embedding = nn.Embedding(num_users + num_items, dim)
        if init == "xavier_uniform":
            # Match the shapes of the official separate user/item tables.
            nn.init.xavier_uniform_(self.embedding.weight[:num_users])
            nn.init.xavier_uniform_(self.embedding.weight[num_users:])
        elif init == "normal":
            nn.init.normal_(self.embedding.weight, std=init_std)
        else:
            raise ValueError(f"Unknown initializer: {init}")

    def forward(self, graph):
        x = self.embedding.weight
        total = x
        for _ in range(self.layers):
            x = torch.sparse.mm(graph, x)
            total = total + x
        return total / (self.layers + 1)

    def bpr_terms(self, representation, edges, negatives):
        u, i = edges
        eu = representation[u]
        ei = representation[i + self.num_users]
        en = representation[negatives + self.num_users]
        return F.softplus((eu * en).sum(-1) - (eu * ei).sum(-1))

    def l2(self, edges, negatives):
        u, i = edges
        indices = torch.cat([u, i + self.num_users, negatives + self.num_users])
        return .5 * self.embedding(indices).square().sum()


def all_node_infonce(view1, view2, anchor_ids, temperature=.2, chunk_size=256,
                     anchor_mode="repeated", reduction="sum"):
    """Official SGL: repeated batch anchors versus all same-type keys.

    Checkpointing bounds the saved logits memory without changing the objective.
    """
    if temperature <= 0 or chunk_size <= 0:
        raise ValueError("temperature and chunk_size must be positive")
    if anchor_mode not in ("repeated", "unique") or reduction not in ("sum", "mean"):
        raise ValueError("Invalid anchor_mode or reduction")
    anchors = anchor_ids if anchor_mode == "repeated" else torch.unique(anchor_ids)
    if not anchors.numel():
        raise ValueError("At least one contrastive anchor is required")
    x, y = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    def part(a, b, labels):
        return F.cross_entropy(a @ b.T / temperature, labels, reduction="sum")

    loss = x.new_zeros(())
    for ids in anchors.split(chunk_size):
        if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
            loss = loss + checkpoint(part, x[ids], y, ids, use_reentrant=False,
                                     preserve_rng_state=False)
        else:
            loss = loss + part(x[ids], y, ids)
    # The optional unique-anchor intervention preserves the batch loss scale.
    if reduction == "mean":
        return loss / anchors.numel()
    return loss if anchor_mode == "repeated" else loss * (anchor_ids.numel() / anchors.numel())


def contrastive_loss(model, graph1, graph2, anchor_edges, temperature=.2,
                     chunk_size=256, side="both", anchor_mode="repeated", reduction="sum"):
    view1, view2 = model(graph1), model(graph2)
    n = model.num_users
    loss = view1.new_zeros(())
    if side in ("both", "user"):
        loss = loss + all_node_infonce(view1[:n], view2[:n], anchor_edges[0], temperature, chunk_size, anchor_mode, reduction)
    if side in ("both", "item"):
        loss = loss + all_node_infonce(view1[n:], view2[n:], anchor_edges[1], temperature, chunk_size, anchor_mode, reduction)
    return loss
