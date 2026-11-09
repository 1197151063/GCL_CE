from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

import torch

from .sampling import generator

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


def holdout(train, num_users, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1.")
    counts = torch.bincount(train[0], minlength=num_users)
    offsets = torch.cat([torch.tensor([0]), counts.cumsum(0)])
    chosen = torch.zeros(train.size(1), dtype=torch.bool)
    rng = generator(seed, "validation_split")
    for u in torch.where(counts >= 2)[0].tolist():
        degree = int(counts[u])
        count = min(degree - 1, max(1, int(degree * ratio)))
        selected = torch.randperm(degree, generator=rng)[:count] + offsets[u]
        chosen[selected] = True
    if not chosen.any():
        raise ValueError("Cannot create validation split: need a user with at least 2 train edges.")
    return train[:, ~chosen], train[:, chosen]


def offsets_for(edges, num_users):
    return torch.cat([torch.tensor([0]), torch.bincount(edges[0], minlength=num_users).cumsum(0)])


@dataclass
class RecData:
    train: torch.Tensor
    validation: torch.Tensor
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
    def load(cls, directory, validation_ratio=.1, split_seed=20261109):
        directory = Path(directory)
        paths = [directory / "train.txt", directory / "test.txt"]
        raw_train, test = [read_edges(p) for p in paths]
        valid_path = directory / "valid.txt"
        validation = read_edges(valid_path) if valid_path.exists() else None
        if validation is not None:
            paths.append(valid_path)
        all_edges = torch.cat([raw_train, test] + ([] if validation is None else [validation]), 1)
        nu, ni = int(all_edges[0].max()) + 1, int(all_edges[1].max()) + 1
        raw_train = canonical_edges(raw_train, ni)
        test = canonical_edges(test, ni)
        if validation is None:
            train, validation = holdout(raw_train, nu, validation_ratio, split_seed)
        else:
            train = raw_train
            validation = canonical_edges(validation, ni)
        for left_name, left, right_name, right in [
            ("train", train, "validation", validation),
            ("train", train, "test", test),
            ("validation", validation, "test", test),
        ]:
            left_codes, right_codes = edge_codes(left, ni), edge_codes(right, ni)
            from .sampling import contains
            if contains(left_codes, right_codes).any():
                raise ValueError(f"Overlapping interactions between {left_name} and {right_name}.")
        ud = torch.bincount(train[0], minlength=nu)
        idg = torch.bincount(train[1], minlength=ni)
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(repr((validation_ratio, split_seed)).encode())
        return cls(train, validation, test, nu, ni, ud, idg,
                   groups_by_degree(ud), groups_by_degree(idg),
                   edge_codes(train, ni), digest.hexdigest())

    def summary(self):
        return {
            "num_users": self.num_users, "num_items": self.num_items,
            "train_edges": self.num_edges, "validation_edges": self.validation.size(1),
            "test_edges": self.test.size(1), "fingerprint": self.fingerprint,
            "user_groups": {n: int((self.user_group == k).sum()) for k, n in enumerate(GROUP_NAMES)},
            "item_groups": {n: int((self.item_group == k).sum()) for k, n in enumerate(GROUP_NAMES)},
        }
