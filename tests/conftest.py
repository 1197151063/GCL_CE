from pathlib import Path
import pytest
import torch

from rrec_experiments.data import RecData


@pytest.fixture
def tiny_data():
    torch.set_num_threads(1)
    return RecData.load(Path(__file__).parent / "fixtures" / "tiny")
