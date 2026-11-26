"""AdaGCL: VGAE, hard-concrete denoiser and the three author-model optimizer steps."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .reproduced_cf import GraphCF, EdgeMM, bpr


def square_norm(module):
    return sum(w.square().sum() for w in module.parameters())


class VGAE(GraphCF):
    def __init__(self, data, cfg):
        super().__init__(data, cfg)
        d = cfg.dim
        self.mean = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.std = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Softplus()
        )
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )

    def encode(self, graph):
        h = sum(self.states(graph))
        mean, std = self.mean(h), self.std(h)
        return mean + torch.randn_like(mean) * std, mean, std

    @torch.no_grad()
    def generate(self, graph):
        z, _, _ = self.encode(graph)
        keep = torch.cat(
            [
                self.decoder(z[r] * z[c]).flatten() >= 0
                for r, c in graph.indices().split(32768, dim=1)
            ]
        )
        weight = graph.values()[keep] / keep.float().mean().clamp_min(1 / graph._nnz())
        return torch.sparse_coo_tensor(
            graph.indices()[:, keep], weight, graph.shape
        ).coalesce()

    def objective(self, graph, u, i, j):
        z, mu, std = self.encode(graph)
        a, b, c = z[u], z[self.nu + i], z[self.nu + j]
        rec = (
            F.softplus(-self.decoder(a * b)).mean()
            + F.softplus(self.decoder(a * c)).mean()
        )
        kl = (
            0.5
            * (mu.square() + std.square() - 1 - 2 * std.clamp_min(1e-12).log())
            .sum(1)
            .mean()
        )
        reg = sum(
            p.square().sum()
            for name, p in self.named_parameters()
            if not name.startswith("decoder.")
        )
        return (
            rec
            + 0.1 * kl
            + bpr(a, b, c) * len(u) / self.cfg.batch_size
            + self.cfg.reg * reg
        )


class Gate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.neighbor = nn.Linear(d, d)
        self.node = nn.Linear(d, d)
        self.out = nn.Linear(2 * d, 1)

    def forward(self, x, r, c):
        # Factor the final linear scoring layer before gathering per-edge features.
        a, b = self.out.weight.chunk(2, dim=1)
        left = F.linear(self.neighbor(x).relu(), a, self.out.bias).flatten()
        right = F.linear(self.node(x).relu(), b).flatten()
        return left[r] + right[c]


class AdaGCL(GraphCF):
    def __init__(self, data, cfg):
        super().__init__(data, cfg)
        self.vgae = VGAE(data, cfg)
        self.gates = nn.ModuleList(Gate(cfg.dim) for _ in range(cfg.layers))
        # The upstream denoiser trainer freezes its feature snapshot at construction.
        self.register_buffer("features", torch.cat((self.user, self.item)).detach())

    def optimizers(self):
        return (
            torch.optim.Adam([self.user, self.item], lr=self.cfg.lr),
            torch.optim.Adam(self.vgae.parameters(), lr=self.cfg.lr),
            torch.optim.Adam(self.gates.parameters(), lr=self.cfg.lr, eps=0.001),
        )

    def forward(self, graph):
        return sum(self.states(graph))

    def denoise(self, graph, beta, learn_gates):
        x = self.features if learn_gates else torch.cat((self.user, self.item))
        states = [x]
        r, c = graph.indices()
        l0 = x.new_zeros(())
        for gate in self.gates:
            with torch.set_grad_enabled(learn_gates):
                logit = gate(x, r, c)
                if learn_gates:
                    noise = torch.rand_like(logit).clamp(1e-7, 1 - 1e-7)
                    prob = ((torch.logit(noise) + logit) / beta).sigmoid()
                    l0 = l0 + (logit - beta * math.log(0.45 / 1.05)).sigmoid().mean()
                else:
                    prob = logit.sigmoid()
                mask = (prob * 1.5 - 0.45).clamp(0, 1)
                degree = x.new_zeros(len(x)).index_add(0, r, mask)
                inv = (degree + (1e-6 if learn_gates else 0.0)).rsqrt().clamp(max=10.0)
                weight = mask * inv[r] * inv[c]
            x = EdgeMM.apply(x, graph.indices(), weight)
            states.append(x)
        return sum(states), l0

    def contrast(self, x, y, ids):
        a, b = F.normalize(x[ids]), F.normalize(y[ids])
        logits = a @ b.T / self.cfg.tau
        pos = logits.diagonal()
        diagonal = torch.eye(len(ids), device=ids.device, dtype=torch.bool)
        return (logits.masked_fill(diagonal, -torch.inf).logsumexp(1) - pos).mean()

    def update(self, graph, u, i, j, optimizers, epoch):
        opt, vopt, gopt = optimizers
        view = self.vgae.generate(graph)
        ids = torch.cat((u, i + self.nu))
        opt.zero_grad()
        a, b = self(view), self.denoise(graph, 1.0, False)[0]
        mi = self.cfg.ssl * self.contrast(a, b, ids)
        mi.backward()
        opt.step()
        opt.zero_grad()
        ib = self.cfg.ib * (
            self.contrast(self(view), a.detach(), ids)
            + self.contrast(self.denoise(graph, 1.0, False)[0], b.detach(), ids)
        )
        ib.backward()
        opt.step()
        opt.zero_grad()
        vopt.zero_grad()
        gopt.zero_grad()
        z = self(graph)
        scale = len(u) / self.cfg.batch_size
        rec = scale * bpr(z[u], z[self.nu + i], z[self.nu + j]) + self.cfg.reg * (
            self.user.square().sum() + self.item.square().sum()
        )
        beta = max(0.05, 2.0 * 0.98 ** (epoch - 1))
        h, l0 = self.denoise(graph, beta, True)
        gen = self.vgae.objective(graph, u, i, j) + scale * bpr(
            h[u], h[self.nu + i], h[self.nu + j]
        )
        gen = gen + self.cfg.l0 * l0 + self.cfg.reg * square_norm(self.gates)
        (rec + gen).backward()
        for optimizer in optimizers:
            optimizer.step()
        return (mi + ib + rec + gen).detach()
