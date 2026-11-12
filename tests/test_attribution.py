import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rrec_experiments import attribution
from rrec_experiments.model import SGL, all_node_infonce, contrastive_components, contrastive_loss, normalized_graph
from rrec_experiments.sampling import generator
from rrec_experiments.train import epoch_samples, parser, sampling_policy, validate
from rrec_experiments.weighting import user_anchor_weights, user_sampling_mass


def test_weighted_infonce_matches_dense_loss_and_gradients():
    x = torch.randn(7, 4, generator=generator(45), dtype=torch.double, requires_grad=True)
    y = torch.randn(7, 4, generator=generator(46), dtype=torch.double, requires_grad=True)
    ids = torch.tensor([0, 0, 6, 1, 4])
    weights = torch.tensor([3., 3., .7, .7, .7], dtype=torch.double)
    dense = (F.cross_entropy(F.normalize(x, dim=1)[ids] @ F.normalize(y, dim=1).T / .2,
                             ids, reduction="none") * weights).sum()
    actual = all_node_infonce(x, y, ids, .2, 2, anchor_weights=weights)
    expected_grad = torch.autograd.grad(dense, (x, y))
    actual_grad = torch.autograd.grad(actual, (x, y))
    assert torch.allclose(dense, actual, atol=1e-12)
    for a, b in zip(expected_grad, actual_grad):
        assert torch.allclose(a, b, atol=1e-10)
    with pytest.raises(ValueError, match="repeated"):
        all_node_infonce(x, y, ids, anchor_mode="unique", anchor_weights=weights)


@pytest.mark.parametrize("sampler,target", [("user", "fast"), ("fast", "user"), ("full", "fast")])
def test_head_weighting_preserves_expected_strength_and_within_group_law(tiny_data, sampler, target):
    w, metadata = user_anchor_weights(tiny_data, sampler, target)
    base = user_sampling_mass(tiny_data, sampler)
    desired = user_sampling_mass(tiny_data, target)
    head = tiny_data.user_group == 0
    weighted = base * w.double()
    assert abs(float(weighted.sum()) - 1) < 1e-7
    assert torch.allclose(weighted[head].sum(), desired[head].sum(), atol=1e-7)
    for mask in [head, ~head]:
        assert torch.allclose(weighted[mask]/weighted[mask].sum(), base[mask]/base[mask].sum(), atol=1e-7)
    assert abs(metadata["expected_weight"] - 1) < 1e-12


def test_alignment_partition_exactly_reconstruct_loss_and_gradients(tiny_data):
    model = SGL(tiny_data.num_users, tiny_data.num_items, 4, 1).double()
    views = [normalized_graph(tiny_data, "cpu", .1, generator(j)).double() for j in [7, 8]]
    edges = tiny_data.train[:, torch.tensor([0, 0, 1, 3, 19, 19])]
    alignment, partition = contrastive_components(model, *views, edges, chunk_size=2)
    reconstructed = alignment + partition
    gradient, = torch.autograd.grad(reconstructed, model.embedding.weight)
    loss = contrastive_loss(model, *views, edges, chunk_size=2, reduction="mean")
    expected, = torch.autograd.grad(loss, model.embedding.weight)
    assert torch.allclose(reconstructed, loss, atol=1e-12)
    assert torch.allclose(gradient, expected, atol=1e-10)


def test_study_changes_only_intended_streams(tiny_data, tmp_path):
    args, samples = {}, {}
    common = ["--data-dir", str(tmp_path), "--output-dir", str(tmp_path)]
    for name, flags in attribution.PROFILES.items():
        args[name] = parser().parse_args([*common, *flags])
        validate(args[name])
        samples[name] = epoch_samples(tiny_data, args[name], 7)
    assert all(torch.equal(row[0], samples["baseline"][0]) for row in samples.values())
    for name in ["l2-fast", "head-up", "no-cl"]:
        assert torch.equal(samples[name][1], samples["baseline"][1])
    for name in ["cl-fast", "head-up", "no-cl"]:
        assert torch.equal(samples[name][2], samples["baseline"][2])
    assert torch.equal(samples["cl-fast"][1], samples["cl-l2-fast"][1])
    assert not torch.equal(samples["cl-l2-fast"][1], samples["shared-fast"][1])
    assert torch.equal(samples["shared-fast"][1], samples["head-down"][1])
    assert not sampling_policy(args["cl-l2-fast"])["cl_bpr_shared"]
    assert sampling_policy(args["shared-fast"])["cl_bpr_shared"]


def test_study_rejects_treatment_overrides_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["attribution", "--cl-sampler", "fast"])
    with pytest.raises(SystemExit) as error:
        attribution.main()
    assert error.value.code == 2


def test_study_stops_on_source_change_before_next_run(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures/tiny"
    fingerprints = ["initial"]
    launched = []
    monkeypatch.setattr(attribution, "source_fingerprint", lambda: fingerprints[0])
    monkeypatch.setattr(attribution, "summarize_study", lambda path: None)

    def change_source(command, check):
        launched.append(command)
        fingerprints[0] = "changed"

    monkeypatch.setattr(attribution.subprocess, "run", change_source)
    monkeypatch.setattr("sys.argv", ["attribution", "--data-dir", str(fixture), "--output-dir", str(tmp_path),
                                      "--ks", "2", "--selection-k", "2"])
    with pytest.raises(RuntimeError, match="Source changed"):
        attribution.main()
    assert len(launched) == 1


def test_all_study_runs_report_portably_and_reject_tampered_treatments(tmp_path, monkeypatch):
    from rrec_experiments.train import run

    fixture = Path(__file__).parent / "fixtures/tiny"
    output = tmp_path / "study"

    def run_locally(command, check):
        run(parser().parse_args(command[3:]))

    monkeypatch.setattr(attribution, "subprocess", SimpleNamespace(run=run_locally))
    monkeypatch.setattr("sys.argv", ["attribution", "--dataset", "custom", "--data-dir", str(fixture),
                                    "--output-dir", str(output), "--epochs", "1", "--device", "cpu",
                                    "--threads", "1", "--dim", "4", "--layers", "1", "--batch-size", "16",
                                    "--ks", "2", "3", "--selection-k", "2", "--diagnostic-batch-size", "8"])
    attribution.main()
    downloaded = tmp_path / "downloaded"
    shutil.copytree(output, downloaded)
    report = attribution.summarize_study(downloaded)
    assert report["complete"] and len(report["runs"]) == 8
    assert len(report["contrasts"]) == 8
    assert report["cl_l2_interaction"] is not None
    assert not list(downloaded.rglob("*.pt*"))
    config_path = downloaded / "cl-fast/seed_2021/config.json"
    config = json.loads(config_path.read_text())
    config["cl_sampler"] = "user"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatch"):
        attribution.summarize_study(downloaded)
