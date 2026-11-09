import torch
import torch.nn.functional as F

from rrec_experiments.model import all_node_infonce, normalized_graph, SGL
from rrec_experiments.sampling import generator


def test_chunked_loss_and_gradients_equal_dense_all_node_infonce():
    rng = generator(7)
    x = torch.randn(9, 5, generator=rng, dtype=torch.double, requires_grad=True)
    y = torch.randn(9, 5, generator=rng, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([7, 2, 2, 0, 7])
    labels = torch.unique(ids)
    dense = F.cross_entropy(F.normalize(x, dim=1)[labels] @ F.normalize(y, dim=1).T / .2, labels)
    expected = torch.autograd.grad(dense, (x, y))
    chunked = all_node_infonce(x, y, ids, .2, 2)
    actual = torch.autograd.grad(chunked, (x, y))
    assert torch.allclose(chunked, dense, atol=1e-12)
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, atol=1e-10)
    assert actual[1][5].norm() > 0, "Nodes outside the anchor batch still receive key gradients"
    assert actual[0][5].norm() == 0


def test_graph_is_symmetric_and_dropout_uses_fixed_stream(tiny_data):
    graph = normalized_graph(tiny_data, "cpu", .2, generator(3, "view"))
    repeat = normalized_graph(tiny_data, "cpu", .2, generator(3, "view"))
    assert torch.equal(graph.indices(), repeat.indices())
    assert torch.equal(graph.values(), repeat.values())
    assert torch.allclose(graph.to_dense(), graph.to_dense().T)
    assert (graph.to_dense().diag() > 0).all()
    model = SGL(tiny_data.num_users, tiny_data.num_items, 4, 2)
    representation = model(graph)
    assert torch.isfinite(representation).all()
    representation.square().mean().backward()
    assert model.embedding.weight.grad is not None
