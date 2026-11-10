from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

import torch

GROUP_NAMES = ("head", "middle", "tail", "cold")


def read_edges(path):
    users, items = [], []
    for lineno, line in enumerate(Path(path).open(), 1):
        values = list(map(int, line.split()))
        if not values:
            continue
        if min(values) < 0:
            raise ValueError(f"Negative ID in {path}:{lineno}")
        users.extend([values[0]] * (len(values) - 1))
        items.extend(values[1:])
    if not users:
        raise ValueError(f"No interactions in {path}")
    return torch.tensor([users, items], dtype=torch.long)


def groups_by_degree(degree, head_fraction=.1, tail_fraction=.5):
    groups = torch.full_like(degree, 3)
    active = torch.where(degree > 0)[0]
    if not active.numel():
        return groups
    # Stable ties use ascending IDs, and groups never overlap.
    order = active[torch.argsort(degree[active], descending=True, stable=True)]
    n = len(order)
    nh = max(1, math.ceil(n * head_fraction))
    nt = min(n - nh, max(1, int(n * tail_fraction)))
    groups[order] = 1
    groups[order[:nh]] = 0
    if nt:
        groups[order[-nt:]] = 2
    return groups


def edge_codes(edges, num_items):
    return edges[0] * num_items + edges[1]


def canonical_edges(edges, num_items):
    codes = edge_codes(edges, num_items)
    if codes.unique().numel() != codes.numel():
        raise ValueError("Duplicate (user, item) interactions found; clean input explicitly.")
    return edges[:, torch.argsort(codes)]


def offsets_for(edges, num_users):
    return torch.cat([torch.tensor([0]), torch.bincount(edges[0], minlength=num_users).cumsum(0)])


@dataclass
class RecData:
    train: torch.Tensor
    test: torch.Tensor
    num_users: int
    num_items: int
    user_degree: torch.Tensor
    item_degree: torch.Tensor
    user_group: torch.Tensor
    item_group: torch.Tensor
    train_codes: torch.Tensor
    fingerprint: str

    @property
    def num_edges(self):
        return self.train.size(1)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        paths = [directory / "train.txt", directory / "test.txt"]
        train, test = [read_edges(p) for p in paths]
        all_edges = torch.cat([train, test], 1)
        nu, ni = int(all_edges[0].max()) + 1, int(all_edges[1].max()) + 1
        train = canonical_edges(train, ni)
        test = canonical_edges(test, ni)
        from .sampling import contains
        if contains(edge_codes(train, ni), edge_codes(test, ni)).any():
            raise ValueError("Overlapping interactions between train and test.")
        ud = torch.bincount(train[0], minlength=nu)
        idg = torch.bincount(train[1], minlength=ni)
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(b"full_train_test_v1")
        return cls(train, test, nu, ni, ud, idg,
                   groups_by_degree(ud), groups_by_degree(idg),
                   edge_codes(train, ni), digest.hexdigest())

    def summary(self):
        return {
            "num_users": self.num_users, "num_items": self.num_items,
            "train_edges": self.num_edges,
            "test_edges": self.test.size(1), "fingerprint": self.fingerprint,
            "user_groups": {n: int((self.user_group == k).sum()) for k, n in enumerate(GROUP_NAMES)},
            "item_groups": {n: int((self.item_group == k).sum()) for k, n in enumerate(GROUP_NAMES)},
        }
