"""Collaborative filtering baselines, with loss conventions from the author implementations."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def table(n, dim):
    return nn.Parameter(nn.init.xavier_uniform_(torch.empty(n, dim)))


def propagate(graph, x, depth):
    states = [x]
    for _ in range(depth):
        states.append(torch.sparse.mm(graph, states[-1]))
    return states


def bpr(u, i, j):
    return F.softplus(-(u * (i - j)).sum(1)).mean()


class GraphCF(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self.nu = data.num_users
        self.user = table(data.num_users, cfg.dim)
        self.item = table(data.num_items, cfg.dim)
        self.cfg = cfg

    def states(self, graph):
        return propagate(graph, torch.cat((self.user, self.item)), self.cfg.layers)

    def forward(self, graph):
        return torch.stack(self.states(graph)).mean(0)


class SGCL(GraphCF):
    def objective(self, graph, u, i, j):
        z = self(graph)
        a, b = F.normalize(z[u]), F.normalize(z[self.nu + i])
        log_u = (a @ a.T / self.cfg.tau).logsumexp(1)
        log_i = (b @ b.T / self.cfg.tau).logsumexp(1)
        return (torch.logaddexp(log_u, log_i) - (a * b).sum(1) / self.cfg.tau).sum()


class NLGCL(GraphCF):
    def objective(self, graph, u, i, j):
        states = self.states(graph)
        z = torch.stack(states).mean(0)
        loss = bpr(z[u], z[self.nu + i], z[self.nu + j])
        reg = sum(x.norm() for x in (self.user[u], self.item[i], self.item[j])) / len(u)
        labels = torch.arange(len(u), device=u.device)
        for prev, cur in zip(states, states[1:]):
            for anchors, keys, weight in (
                (cur[self.nu + i], prev[u], self.cfg.alpha),
                (cur[u], prev[self.nu + i], 1 - self.cfg.alpha),
            ):
                logits = F.normalize(anchors) @ F.normalize(keys).T / self.cfg.tau
                loss = loss + self.cfg.ssl * weight * (
                    F.cross_entropy(logits, labels, reduction="sum") + 1e-6
                )
        return loss + self.cfg.reg * reg


class LightCCF(GraphCF):
    def objective(self, graph, u, i, j):
        z = self(graph)
        a, b, c = z[u], z[self.nu + i], z[self.nu + j]
        reg = sum(
            x.square().sum() for x in (self.user[u], self.item[i], self.item[j])
        ) / (2 * len(u))
        ranking = -torch.log(((a * (b - c)).sum(1)).sigmoid() + 1e-7).mean()
        a, b = F.normalize(a), F.normalize(b)
        log_den = ((a @ b.T + a @ a.T) / self.cfg.tau).logsumexp(1)
        log_prob = (a * b).sum(1) / self.cfg.tau - log_den
        na = -torch.logaddexp(log_prob, log_prob.new_tensor(1e-5).log()).mean()
        return ranking + self.cfg.reg * reg + self.cfg.ssl * na


def all_key_partition(anchors, keys, tau):
    # Keep the official all-key denominator at the original batch size on an 8 GiB GPU.
    def block(query, bank):
        return (query @ bank.T / tau).logsumexp(1)

    if len(anchors) <= 256:
        return block(anchors, keys)
    return torch.cat(
        [checkpoint(block, q, keys, use_reentrant=False) for q in anchors.split(256)]
    )


class LightGCL(GraphCF):
    def __init__(self, data, cfg):
        super().__init__(data, cfg)
        u, i = data.train.to(cfg.device)
        degree = data.user_degree[u.cpu()].float() * data.item_degree[i.cpu()].float()
        adj = torch.sparse_coo_tensor(
            torch.stack((u, i)),
            degree.rsqrt().to(cfg.device),
            (data.num_users, data.num_items),
        ).coalesce()
        left, s, right = torch.svd_lowrank(adj, q=min(cfg.rank, *adj.shape))
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.register_buffer("singular", s)

    def forward(self, graph):
        return sum(self.states(graph))

    def objective(self, graph, u, i, j):
        states = self.states(graph)
        eu, ei = sum(states).split((self.nu, len(self.item)))
        gu, gi = self.user, self.item
        for prev in states[:-1]:
            gu = gu + (self.left * self.singular) @ (self.right.T @ prev[self.nu :])
            gi = gi + (self.right * self.singular) @ (self.left.T @ prev[: self.nu])
        loss = bpr(eu[u], ei[i], ei[j])
        for teacher, online, ids in ((gu, eu, u), (gi, ei, torch.cat((i, j)))):
            scores = all_key_partition(teacher[ids], online, self.cfg.tau)
            positive = (teacher[ids] * online[ids]).sum(1) / self.cfg.tau
            loss = (
                loss
                + self.cfg.ssl
                * (
                    scores.logaddexp(scores.new_tensor(1e-8).log())
                    - positive.clamp(-5.0, 5.0)
                ).mean()
            )
        return loss + self.cfg.reg * (
            self.user.square().sum() + self.item.square().sum()
        )


class EdgeMM(torch.autograd.Function):
    """Sparse propagation with trainable weights and an O(E) edge gradient."""

    @staticmethod
    def forward(ctx, nodes, edges, weights):
        size = (len(nodes), len(nodes))
        adj = torch.sparse_coo_tensor(edges, weights, size).coalesce()
        ctx.save_for_backward(nodes, edges, adj)
        return torch.sparse.mm(adj, nodes)

    @staticmethod
    def backward(ctx, grad):
        nodes, edges, adj = ctx.saved_tensors
        dw = (
            torch.cat(
                [
                    (grad[row] * nodes[col]).sum(1)
                    for row, col in edges.split(32768, dim=1)
                ]
            )
            if ctx.needs_input_grad[2]
            else None
        )
        dx = (
            torch.sparse.mm(adj.T, grad.contiguous())
            if ctx.needs_input_grad[0]
            else None
        )
        return dx, None, dw


class RGCF(GraphCF):
    def __init__(self, data, cfg):
        super().__init__(data, cfg)
        self.register_buffer(
            "inter",
            torch.sparse_coo_tensor(
                data.train, torch.ones(data.num_edges), (data.num_users, data.num_items)
            ).coalesce(),
        )

    def encode(self):
        u = torch.sparse.mm(self.inter, self.item)
        i = torch.sparse.mm(self.inter.T, self.user)
        row, col = self.inter.indices()
        a, b = F.normalize(u), F.normalize(i)
        score = torch.cat(
            [(a[r] * b[c]).sum(1) for r, c in self.inter.indices().split(32768, dim=1)]
        )
        score = (score + 1) * 0.5
        score = score.masked_fill(score < self.cfg.prune, 0.0)
        src, dst = torch.cat((row, col + self.nu)), torch.cat((col + self.nu, row))
        edges = torch.stack((src, dst))
        scores = score.repeat(2)
        ego = torch.cat((u, i))
        degree = ego.new_zeros(len(ego)).index_add(0, src, scores)
        weights = scores / (degree[src] + 1e-7)
        return ego, edges, weights

    def aggregate(self, x, edges, weights):
        layers = [x]
        for _ in range(self.cfg.layers):
            layers.append(EdgeMM.apply(layers[-1], edges, weights))
        return torch.stack(layers).mean(0)

    def forward(self, graph):
        return self.aggregate(*self.encode())

    def objective(self, graph, u, i, j):
        ego, edges, weights = self.encode()
        z = self.aggregate(ego, edges, weights)
        loss = bpr(z[u], z[self.nu + i], z[self.nu + j])
        reg = sum(x.norm() for x in (ego[u], ego[self.nu + i], ego[self.nu + j])) / len(
            u
        )
        count = int(edges.size(1) * self.cfg.aug * 0.5)
        with torch.no_grad():
            # Upstream ranks a 10x candidate pool by unnormalized dot product.
            a = torch.randint(self.nu, (10 * count,), device=u.device)
            b = torch.randint(len(self.item), (10 * count,), device=u.device) + self.nu
            if count:
                scores = torch.cat(
                    [
                        (ego[r] * ego[c]).sum(1)
                        for r, c in zip(a.split(32768), b.split(32768))
                    ]
                )
                top = scores.topk(count).indices
                a, b = a[top], b[top]
            extra = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
            aug = torch.sparse_coo_tensor(
                torch.cat((edges, extra), 1),
                torch.cat(
                    (weights.detach(), weights.detach().median().expand(2 * count))
                ),
                (len(ego), len(ego)),
            ).coalesce()
            src = aug.indices()[0]
            degree = ego.new_zeros(len(ego)).index_add(0, src, aug.values())
            # RGCF get_aug_adj uses the row factor twice.
            norm = aug.values() / (degree[src] + 1e-7)
        alt = self.aggregate(ego, aug.indices(), norm)[: self.nu]
        logits = F.normalize(z[u]) @ F.normalize(alt).T / self.cfg.tau
        return (
            loss
            + self.cfg.reg * reg
            + self.cfg.ssl * F.cross_entropy(logits, u, reduction="sum")
        )
