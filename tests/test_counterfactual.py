import json
from pathlib import Path

import torch

from rrec_experiments.counterfactual import BudgetTransfer, fork_state, main, split_probe
from rrec_experiments.data import edge_codes, read_edges
from rrec_experiments.model import SGL


def test_probe_split_partitions_train_and_preserves_node_coverage(tiny_data):
    path = Path(__file__).parent / "fixtures/tiny/train.txt"
    fit, probe = split_probe(path, .2, 17)
    original = read_edges(path)
    joined = torch.cat((fit.train, probe), dim=1)
    assert torch.equal(edge_codes(original, fit.num_items).sort().values,
                       edge_codes(joined, fit.num_items).sort().values)
    for side in (0, 1):
        assert torch.equal(original[side].unique(), fit.train[side].unique())
    assert not fit.test.numel()
    again, probe_again = split_probe(path, .2, 17)
    assert torch.equal(fit.train, again.train) and torch.equal(probe, probe_again)


def test_budget_transfer_is_symmetric_and_keeps_the_other_side(tiny_data):
    item_ids = [int(torch.where(tiny_data.item_group == g)[0][0]) for g in range(3)]
    edges = torch.tensor([[0, 0, 0, 0, 1, 1], [item_ids[g] for g in (0, 0, 0, 1, 1, 2)]])
    for donor, recipient in [(0, 1), (1, 0)]:
        transfer = BudgetTransfer(tiny_data, "item", donor, recipient, .5)
        user, item = transfer(edges)
        assert user is None and torch.isclose(item.sum(), torch.tensor(6.))
        assert torch.isclose(item[edges[1] == item_ids[donor]].sum(),
                             torch.tensor(float((edges[1] == item_ids[donor]).sum()) - 1))
        assert item[-1] == 1 and transfer.report()["transferred_weight"] == 1
    transfer.reset()
    assert transfer(edges[:, :1]) == (None, None)
    assert transfer.report()["batches_without_both_groups"] == 1


def test_fork_copies_adam_moments_without_aliasing(tiny_data):
    model = SGL(tiny_data.num_users, tiny_data.num_items, 4, 1)
    optimizer = torch.optim.Adam(model.parameters())

    def step(m, opt):
        opt.zero_grad()
        m.embedding.weight.square().sum().backward()
        opt.step()

    step(model, optimizer)
    branch, adam = fork_state(model, optimizer)
    param, copied = model.embedding.weight, branch.embedding.weight
    assert adam.param_groups[0]["params"][0] is copied
    assert torch.equal(optimizer.state[param]["exp_avg"], adam.state[copied]["exp_avg"])
    assert optimizer.state[param]["exp_avg"].data_ptr() != adam.state[copied]["exp_avg"].data_ptr()
    original = param.detach().clone()
    step(branch, adam)
    assert torch.equal(param, original)
    step(model, optimizer)
    assert torch.equal(param, copied)


def test_counterfactual_matrix_runs_without_a_test_file(tmp_path):
    data = tmp_path / "fit_source"
    data.mkdir()
    fixture = Path(__file__).parent / "fixtures/tiny/train.txt"
    (data / "train.txt").write_bytes(fixture.read_bytes())
    output = tmp_path / "result"
    main(["--data-dir", str(data), "--output-dir", str(output), "--fork-epochs", "1", "2",
          "--horizons", "1", "2", "--device", "cpu", "--dim", "4", "--layers", "1",
          "--threads", "1", "--batch-size", "16", "--ks", "2", "3", "--selection-k", "2"])
    result = json.loads((output / "influence.json").read_text())
    assert result["complete"] and set(result["forks"]) == {"1", "2"}
    for fork in result["forks"].values():
        assert set(fork["matrix"]) == {"step_1", "epoch_1", "epoch_2"}
        assert all(len(columns) == 12 for columns in fork["matrix"].values())
        for branch in fork["transfers"].values():
            assert branch["sample_streams"] == fork["baseline"]["sample_streams"]
    rows = [json.loads(line) for line in (output / "branches.jsonl").read_text().splitlines()]
    baseline = {(r["fork_epoch"], r["epoch_offset"]): r["training"]["cl_loss"]
                for r in rows if r["branch"] == "fast"}
    for side in ("user", "item"):
        assert any(r["branch"].startswith(side + ":") and r["budget"]["transferred_weight"] > 0 and
                   abs(r["training"]["cl_loss"] - baseline[(r["fork_epoch"], r["epoch_offset"])]) > 1e-7
                   for r in rows)
    assert not list(output.rglob("*.pt*"))
