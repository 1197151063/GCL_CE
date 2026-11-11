"""Reference equations from wujcan/SGL-Torch model/general_recommender/SGL.py."""
import torch
import torch.nn.functional as F

from rrec_experiments.model import SGL, contrastive_loss, normalized_graph
from rrec_experiments.sampling import generator


def test_official_objective_gradients_and_adam_update_match(tiny_data):
    model = SGL(tiny_data.num_users, tiny_data.num_items, 4, 3).double()
    reference = SGL(tiny_data.num_users, tiny_data.num_items, 4, 3).double()
    reference.load_state_dict(model.state_dict())
    graph = normalized_graph(tiny_data, "cpu").double()
    g1 = normalized_graph(tiny_data, "cpu", .1, generator(1)).double()
    g2 = normalized_graph(tiny_data, "cpu", .1, generator(2)).double()
    # Repeat anchors deliberately: unique-anchor objectives must fail this check.
    edges = tiny_data.train[:, torch.tensor([0, 0, 1, 5, 5])]
    negative = torch.tensor([19, 18, 17, 16, 15])
    n = tiny_data.num_users
    u, i = edges

    def official_gcn(adjacency):
        x = reference.embedding.weight
        layers = [x]
        for _ in range(reference.layers):
            x = torch.sparse.mm(adjacency, x)
            layers.append(x)
        return torch.stack(layers, dim=1).mean(dim=1)

    h = official_gcn(graph)
    delta = (h[u] * (h[i+n] - h[negative+n])).sum(-1)
    bpr = -F.logsigmoid(delta).sum()
    v1, v2 = F.normalize(official_gcn(g1), dim=1), F.normalize(official_gcn(g2), dim=1)
    cl = h.new_zeros(())
    for offset, size, ids in [(0,n,u), (n,tiny_data.num_items,i)]:
        x, y = v1[offset:offset+size], v2[offset:offset+size]
        positive = (x[ids] * y[ids]).sum(-1)
        cl += torch.logsumexp((x[ids] @ y.T - positive[:,None]) / .2, dim=1).sum()
    initial = reference.embedding.weight
    reg = .5 * sum(initial[ids].square().sum() for ids in [u, i+n, negative+n])
    expected = bpr + .1*cl + .0001*reg
    optimizer_ref = torch.optim.Adam(reference.parameters(), lr=.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    expected.backward()
    optimizer_ref.step()

    rec = model.bpr_terms(model(graph), edges, negative).sum() + .0001*model.l2(edges,negative)
    actual_cl = .1 * contrastive_loss(model,g1,g2,edges,.2,2)
    assert torch.allclose(expected.detach(), (rec+actual_cl).detach(), atol=1e-12)
    rec.backward()
    actual_cl.backward()
    assert torch.allclose(reference.embedding.weight.grad, model.embedding.weight.grad, atol=1e-11, rtol=1e-10)
    optimizer.step()
    assert torch.allclose(reference.embedding.weight, model.embedding.weight, atol=1e-12, rtol=1e-10)


def test_xavier_initializes_user_and_item_tables_separately(monkeypatch):
    initializer = torch.nn.init.xavier_uniform_
    shapes = []

    def capture(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return initializer(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.init, "xavier_uniform_", capture)
    SGL(7, 13, 4)
    assert shapes == [(7, 4), (13, 4)]
