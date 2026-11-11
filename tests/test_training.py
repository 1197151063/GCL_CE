import json
from pathlib import Path
import torch
import pytest

from rrec_experiments import train
from rrec_experiments.train import parser, run


def test_cpu_training_is_reproducible_and_saves_diagnostics(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "tiny"
    common = ["--data-dir", str(fixture), "--variant", "fast", "--device", "cpu", "--threads", "1",
              "--epochs", "2", "--batch-size", "16", "--dim", "4", "--layers", "1", "--ks", "2", "3",
              "--selection-k", "2", "--eval-every", "1", "--diagnostics-every", "1",
              "--diagnostic-batch-size", "8", "--anchor-chunk-size", "3"]
    results, models = [], []
    model_type = train.SGL

    def capture_model(*args, **kwargs):
        model = model_type(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(train, "SGL", capture_model)
    for name in ["one", "two"]:
        args = parser().parse_args([*common, "--output-dir", str(tmp_path / name)])
        result = run(args)
        results.append(result)
        assert not list((tmp_path / name).glob("*.pt*"))
    assert results[0]["test"] == results[1]["test"]
    assert "validation" not in results[0]
    config = json.loads((tmp_path / "one" / "config.json").read_text())
    assert config["data"]["train_edges"] == 36
    assert "validation_edges" not in config["data"]
    assert config["data_protocol"] == "full_train_test_v1"
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
    rows = [json.loads(l) for l in (tmp_path / "one" / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[0]["epoch"] == 0
    assert rows[-1]["checkpoint"] == "best"
    assert rows[-1]["groups"]["user_head"]["nodes"] > 0
    assert results[0]["optimizer_steps"] == 6


@pytest.mark.parametrize("patience,improvement_epoch,expected_stop", [
    (10, 1, 11),
    (10, 5, 15),
    (0, 1, 20),
])
def test_early_stopping_counts_epochs_and_restores_best(
        tmp_path, monkeypatch, patience, improvement_epoch, expected_stop):
    fixture = Path(__file__).parent / "fixtures" / "tiny"
    args = parser().parse_args([
        "--data-dir", str(fixture), "--output-dir", str(tmp_path / "run"),
        "--variant", "bpr-fast", "--device", "cpu", "--threads", "1",
        "--epochs", "20", "--dim", "2", "--layers", "0", "--ks", "2",
        "--selection-k", "2", "--diagnostics-every", "0", "--eval-every", "5",
        "--patience", str(patience),
    ])
    calls = []

    def fake_train(model, optimizer, graph, data, args, epoch):
        with torch.no_grad():
            model.embedding.weight.fill_(epoch)
        return {"optimizer_steps": 2, "bpr_loss": 0., "cl_loss": 0.}

    def fake_evaluate(model, graph, data, split, ks, batch_size, score):
        epoch = int(model.embedding.weight[0, 0].item())
        calls.append((split, epoch))
        assert split == "test"
        value = .2 if epoch >= improvement_epoch else .1
        # NDCG keeps improving but official early stopping only monitors Recall.
        return {"overall": {"recall@2": value, "ndcg@2": epoch / 100.}}

    monkeypatch.setattr(train, "train_epoch", fake_train)
    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    result = run(args)
    assert result["stopped_epoch"] == expected_stop
    assert result["best_epoch"] == improvement_epoch
    assert result["best_selection_score"] == .2
    assert result["optimizer_steps"] == 2 * expected_stop
    assert result["stop_reason"] == ("early_stopping" if patience else "max_epochs")
    assert result["selection_split"] == "test"
    assert result["test_used_for_selection"]
    # Final test must evaluate the restored best model, not the last epoch.
    assert calls[-1] == ("test", improvement_epoch)
    if patience:
        assert calls[:-1] == [("test", epoch) for epoch in range(1, expected_stop + 1)]
    else:
        assert calls[:-1] == [("test", epoch) for epoch in [1, 5, 10, 15, 20]]
    rows = [json.loads(line) for line in (args.output_dir / "epochs.jsonl").read_text().splitlines()]
    assert rows[-1]["epochs_without_improvement"] == expected_stop - improvement_epoch
    assert rows[-1]["best_epoch"] == improvement_epoch
    assert not list(args.output_dir.glob("*.pt*"))


def test_default_protocol_is_single_seed_test_early_stopping(tmp_path, monkeypatch):
    from rrec_experiments import suite

    fixture = Path(__file__).parent / "fixtures" / "tiny"
    args = parser().parse_args(["--data-dir", str(fixture), "--output-dir", str(tmp_path)])
    assert (args.patience, args.eval_every) == (10, 1)
    assert (args.init, args.score, args.self_loops) == ("xavier_uniform", "dot", False)
    assert (args.ssl_weight, args.temperature, args.drop_ratio) == (.1, .2, .1)
    assert (args.cl_sampler, args.cl_anchor_mode, args.l2_source, args.selection_objective) == ("user", "repeated", "official", "recall")
    launched = []

    def fake_subprocess(command, check):
        run_args = parser().parse_args(command[3:])
        assert (run_args.seed, run_args.patience) == (2021, 10)
        launched.append(run_args.variant)

    monkeypatch.setattr("sys.argv", ["suite", "--data-dir", str(fixture), "--output-dir", str(tmp_path / "suite")])
    monkeypatch.setattr(suite.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(suite, "summarize", lambda path: None)
    suite.main()
    assert launched == ["official", "fast", "full"]


def test_suite_stops_if_source_changes_between_runs(tmp_path, monkeypatch):
    from rrec_experiments import suite
    fixture = Path(__file__).parent / "fixtures" / "tiny"
    fingerprint = ["initial"]
    launched = []
    monkeypatch.setattr(suite, "source_fingerprint", lambda: fingerprint[0])

    def change_code(command, check):
        launched.append(command[command.index("--variant") + 1])
        fingerprint[0] = "changed"

    monkeypatch.setattr(suite.subprocess, "run", change_code)
    monkeypatch.setattr("sys.argv", ["suite", "--data-dir", str(fixture), "--output-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="Source code changed"):
        suite.main()
    assert launched == ["official"]
