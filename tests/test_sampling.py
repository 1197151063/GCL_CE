import torch

from rrec_experiments.sampling import epoch_indices, generator, negative_items, contains


def test_fast_covers_every_edge_and_streams_are_reproducible(tiny_data):
    a = epoch_indices(tiny_data, "fast", 0, 1, "bpr")
    assert torch.equal(a.sort().values, torch.arange(tiny_data.num_edges))
    torch.rand(1000)
    assert torch.equal(a, epoch_indices(tiny_data, "fast", 0, 1, "bpr"))
    assert not torch.equal(a, epoch_indices(tiny_data, "fast", 0, 1, "cl"))


def test_full_matches_inverse_user_degree_weighted_replacement(tiny_data):
    samples = torch.cat([epoch_indices(tiny_data, "full", 0, e, "bpr") for e in range(500)])
    frequency = torch.bincount(tiny_data.train[0, samples], minlength=tiny_data.num_users).double()
    observed = frequency / samples.numel()
    expected = tiny_data.user_degree.double() / (tiny_data.user_degree + 1)
    expected = expected / expected.sum()
    assert torch.allclose(observed, expected, atol=.015, rtol=0)
    assert epoch_indices(tiny_data, "full", 0, 0, "bpr").unique().numel() < tiny_data.num_edges


def test_negatives_exclude_training_positives_only(tiny_data):
    users = torch.arange(tiny_data.num_users).repeat(100)
    negatives = negative_items(tiny_data, users, generator(0, "neg"))
    codes = users * tiny_data.num_items + negatives
    assert not contains(tiny_data.train_codes, codes).any()
    test_codes = tiny_data.test[0] * tiny_data.num_items + tiny_data.test[1]
    assert contains(test_codes, codes).any(), "Held-out labels must not filter training negatives"


def test_official_sampling_is_uniform_over_active_users(tiny_data):
    samples = torch.cat([epoch_indices(tiny_data, "user", 2021, epoch, "official") for epoch in range(500)])
    frequencies = torch.bincount(tiny_data.train[0, samples], minlength=tiny_data.num_users).double() / samples.numel()
    expected = (tiny_data.user_degree > 0).double()
    expected /= expected.sum()
    assert torch.allclose(frequencies, expected, atol=.015, rtol=0)


def test_fast_full_change_only_bpr_stream_with_default_settings(tiny_data, tmp_path):
    from rrec_experiments.train import parser, epoch_samples
    samples = {}
    for variant in ["official", "fast", "full"]:
        args = parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path), "--variant", variant])
        samples[variant] = epoch_samples(tiny_data, args, 3)
    reference = samples["official"][0]
    for variant, (bpr, cl, l2) in samples.items():
        assert torch.equal(cl, reference)
        assert torch.equal(l2, reference)
        assert torch.equal(bpr, reference) == (variant == "official")
    assert not torch.equal(samples["fast"][0], samples["full"][0])
