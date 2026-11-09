from argparse import Namespace

import pytest

from rrec_experiments.datasets import commands
from rrec_experiments.presets import PRESETS, preset_options
from rrec_experiments.train import parser


def test_all_dataset_plans_use_separate_data_results_and_profiles(tmp_path):
    for name in PRESETS:
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        (folder / "train.txt").write_text("0 0 1\n")
        (folder / "test.txt").write_text("0 2\n")
    args = Namespace(datasets=["all"], data_root=tmp_path / "data", output_root=tmp_path / "runs", dry_run=True)
    planned = commands(args, ["--device", "cpu"])
    assert len(planned) == 4
    for name, command in zip(PRESETS, planned):
        assert command[command.index("--dataset") + 1] == name
        assert command[command.index("--data-dir") + 1] == str(tmp_path / "data" / name)
        assert command[command.index("--output-dir") + 1] == str(tmp_path / "runs" / name)
        assert "--dry-run" in command
    args.datasets = ["all", "yelp2018"]
    with pytest.raises(ValueError):
        commands(args, [])


def test_explicit_training_options_override_dataset_defaults(tmp_path):
    for name in PRESETS:
        args = parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "runs"),
                                    "--dataset", name, *preset_options(name), "--temperature", ".7"])
        assert args.temperature == .7
        assert args.ssl_weight == PRESETS[name]["ssl_weight"]
