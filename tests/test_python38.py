import argparse
import ast
from pathlib import Path

import pytest

from rrec_experiments.train import parser


@pytest.mark.parametrize("flags,expected", [([], False), (["--self-loops"], True), (["--no-self-loops"], False)])
def test_self_loop_flags_without_python39_boolean_action(monkeypatch, tmp_path, flags, expected):
    monkeypatch.delattr(argparse, "BooleanOptionalAction", raising=False)
    args = parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "run"), *flags])
    assert args.self_loops is expected


def test_runtime_sources_parse_as_python38():
    root = Path(__file__).resolve().parents[1]
    for path in [root / "run.py", *sorted((root / "rrec_experiments").glob("*.py"))]:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))
