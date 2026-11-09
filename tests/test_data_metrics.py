import math
import pytest
import torch

from rrec_experiments.data import RecData, edge_codes, groups_by_degree
from rrec_experiments.metrics import evaluate
from rrec_experiments.sampling import contains


def test_validation_split_is_disjoint_and_fixed(tiny_data):
    assert not contains(tiny_data.train_codes, edge_codes(tiny_data.validation, tiny_data.num_items)).any()
    assert (tiny_data.user_degree > 0).all()
    assert tiny_data.validation.size(1) == tiny_data.num_users


def test_overlapping_input_is_rejected(tmp_path):
    (tmp_path / "train.txt").write_text("0 0 1\n1 1 2\n")
    (tmp_path / "test.txt").write_text("0 0\n1 0\n")
    with pytest.raises(ValueError, match="Overlapping"):
        RecData.load(tmp_path)


def test_groups_do_not_confuse_cold_with_tail():
    groups = groups_by_degree(torch.tensor([100, 2, 0, 1]))
    assert groups[0] == 0 and groups[2] == 3
    assert groups[3] == 2


def test_ranking_masks_known_edges_and_uses_each_users_own_ndcg_denominator():
    data = RecData(
        train=torch.tensor([[0, 1], [0, 1]]),
        validation=torch.tensor([[0, 1], [1, 0]]),
        test=torch.tensor([[0, 0, 1], [2, 3, 3]]),
        num_users=2, num_items=5,
        user_degree=torch.tensor([1, 1]), item_degree=torch.tensor([1, 1, 0, 0, 0]),
        user_group=torch.tensor([0, 2]), item_group=torch.tensor([0, 0, 1, 2, 2]),
        train_codes=torch.tensor([0, 6]), fingerprint="test",
    )
    class FixedModel(torch.nn.Module):
        def forward(self, graph):
            return torch.tensor([[1., 0.], [0., 1.], [100., 0.], [0., 100.], [4., 2.], [3., 4.], [5., 3.]])
    result = evaluate(FixedModel(), None, data, "test", [1, 2], 1)
    assert result["overall"]["recall@2"] == .75
    discount = 1 / math.log2(3)
    expected_ndcg = (discount / (1 + discount) + 1) / 2
    assert result["overall"]["ndcg@2"] == pytest.approx(expected_ndcg)
    assert result["users"]["head"]["recall@2"] == .5
    assert sum(g["recall_contribution@2"] for g in result["items"].values()) == pytest.approx(.75)
    assert result["items"]["cold"]["conditional_recall@2"] is None
