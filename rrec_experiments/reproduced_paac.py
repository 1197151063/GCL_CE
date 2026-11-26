"""PAAC with the author's batch popularity split and separate alignment update."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PAAC(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self.nu = data.num_users
        self.cfg = cfg
        self.user = nn.Parameter(torch.empty(self.nu, cfg.dim).normal_(std=0.1))
        self.item = nn.Parameter(torch.empty(data.num_items, cfg.dim).normal_(std=0.1))
        self.degrees = data.item_degree.numpy()
        # RecData stores canonical edges sorted by user. Sort within each history once.
        histories = np.split(
            data.train[1].numpy(), data.user_degree.numpy().cumsum()[:-1]
        )
        self.histories = [
            h[np.argsort(self.degrees[h])] for h in histories if len(h) >= 2
        ]

    def forward(self, graph, perturbed=False):
        x = torch.cat((self.user, self.item))
        states = []
        for _ in range(self.cfg.layers):
            x = torch.sparse.mm(graph, x)
            if perturbed:
                x = x + self.cfg.eps * x.sign() * F.normalize(torch.rand_like(x))
            states.append(x)
        return torch.stack(states).mean(0)

    def item_contrast(self, a, same, other):
        if not len(a):
            return same.new_zeros(())
        a, same, other = F.normalize(a), F.normalize(same), F.normalize(other)
        pos = (a * same).sum(1) / self.cfg.tau
        den = (a @ same.T / self.cfg.tau).logsumexp(1)
        # The release includes the positive once in the same-group sum and once explicitly.
        den = torch.logaddexp(den, pos)
        if len(other) and self.cfg.cross_weight:
            cross = (a @ other.T / self.cfg.tau).logsumexp(1)
            den = torch.logaddexp(den, cross + math.log(self.cfg.cross_weight))
        return (den - pos).mean()

    def objective(self, graph, u, i, j):
        z = self(graph)
        a, b, c = z[u], z[self.nu + i], z[self.nu + j]
        ranking = -((a * (b - c)).sum(1).sigmoid() + 1e-7).log().mean()
        reg = self.cfg.reg * (a.norm() + b.norm())
        v1, v2 = self(graph, True), self(graph, True)
        users = u.unique()
        logits = F.normalize(v1[users]) @ F.normalize(v2[users]).T / self.cfg.tau
        user_cl = F.cross_entropy(logits, torch.arange(len(users), device=u.device))
        # Sort occurrences before splitting, then deduplicate each half separately.
        ids = i.detach().cpu().numpy()
        ordered = ids[np.argsort(self.degrees[ids])]
        half = len(ids) // 2
        lo, hi = [
            torch.as_tensor(np.unique(part), device=i.device) + self.nu
            for part in (ordered[:half], ordered[half:])
        ]
        item_cl = self.cfg.gamma * self.item_contrast(v1[lo], v2[lo], v2[hi]) + (
            1 - self.cfg.gamma
        ) * self.item_contrast(v1[hi], v2[hi], v2[lo])
        return ranking + reg + self.cfg.ssl * (user_cl + item_cl)

    def alignment(self, graph, epoch):
        if not self.histories:
            return None
        rng = np.random.default_rng([self.cfg.seed, epoch])
        lows, highs = [], []
        for h in self.histories:
            if len(h) % 2:
                h = np.delete(h, rng.integers(len(h)))
            half = len(h) // 2
            lows.append(h[:half])
            highs.append(h[half:])
        items = F.normalize(self(graph)[self.nu :])
        lo = torch.as_tensor(np.concatenate(lows), device=items.device)
        hi = torch.as_tensor(np.concatenate(highs), device=items.device)
        return self.cfg.align * (items[lo] - items[hi]).square().sum(1).mean()
