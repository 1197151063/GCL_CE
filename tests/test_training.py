import json
from pathlib import Path
import torch

from rrec_experiments.train import parser, run


def test_cpu_training_is_reproducible_and_saves_diagnostics(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "tiny"
    common = ["--data-dir", str(fixture), "--variant", "C", "--device", "cpu", "--threads", "1",
              "--epochs", "2", "--batch-size", "16", "--dim", "4", "--layers", "1", "--ks", "2", "3",
              "--selection-k", "2", "--eval-every", "1", "--diagnostics-every", "1",
              "--diagnostic-batch-size", "8", "--anchor-chunk-size", "3"]
    results, checkpoints = [], []
    for name in ["one", "two"]:
        args = parser().parse_args([*common, "--output-dir", str(tmp_path / name)])
        result = run(args)
        results.append(result)
        checkpoints.append(torch.load(tmp_path / name / "best.pt", weights_only=True)["model"])
    assert results[0]["test"] == results[1]["test"]
    assert results[0]["validation"] == results[1]["validation"]
    for key in checkpoints[0]:
        assert torch.equal(checkpoints[0][key], checkpoints[1][key])
    rows = [json.loads(l) for l in (tmp_path / "one" / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[0]["epoch"] == 0
    assert rows[-1]["checkpoint"] == "best"
    assert rows[-1]["groups"]["user_head"]["nodes"] > 0
    assert results[0]["optimizer_steps"] == 4
