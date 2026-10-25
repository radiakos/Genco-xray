"""Acceptance tests for trace.py (WS-A). Needs torch (CPU is fine), not graphkit or a checkpoint.

A fake model reproduces the call pattern of GNS_heterogeneous: per step, branch_flow_layer and
node_residuals_layer are called, and physics_mlp is called on all but the last step.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gencoxray import trace  # noqa: E402

T, NB = 4, 3


class _Flow(nn.Module):
    def forward(self, bus_data, edge_index, edge_attr):
        return bus_data[edge_index[0], 0] * 0.1, bus_data[edge_index[0], 1] * 0.0 + 0.2


class _Res(nn.Module):
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        return bus_data_pred[:, 2] * 1.0, bus_data_pred[:, 3] * 1.0


class FakeGNS(nn.Module):
    def __init__(self, new_api=True):
        super().__init__()
        self.num_layers = T
        self.branch_flow_layer = _Flow()
        self.node_residuals_layer = _Res()
        self.physics_mlp = nn.Linear(2, 4)
        self.new_api = new_api

    def _run(self, ei):
        for i in range(T):
            vm = torch.ones(NB) + 0.01 * i
            va = torch.full((NB,), math.pi / 180 * i)  # i degrees
            res = torch.full((NB,), 2.0 ** (-i))
            state = torch.stack([vm, va, res, res], 1)
            self.branch_flow_layer(state, ei, None)
            rp, rq = self.node_residuals_layer(None, None, state, None)
            if i < T - 1:
                self.physics_mlp(torch.stack([rp, rq], -1))
        return {"bus": state, "gen": torch.zeros(1, 1)}


class NewAPI(FakeGNS):
    def forward(self, batch):
        return self._run(batch.edge_index_dict[("bus", "connects", "bus")])


class OldAPI(FakeGNS):
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        return self._run(edge_index_dict[("bus", "connects", "bus")])


class FakeBatch:
    def __init__(self):
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.edge_index_dict = {("bus", "connects", "bus"): ei}
        self.edge_attr_dict = {("bus", "connects", "bus"): torch.zeros(4, 10)}
        self.x_dict = {"bus": torch.zeros(NB, 15)}
        self.mask_dict = {"PQ": torch.tensor([True, True, False]), "PV": torch.tensor([False, False, True]),
                          "REF": torch.tensor([False, False, False])}
        self.batch_dict = {"bus": torch.zeros(NB, dtype=torch.long)}
        self._store = {"scenario_id": torch.tensor([42])}

    def __getitem__(self, k):
        return self._store[k]


@pytest.mark.parametrize("cls", [NewAPI, OldAPI])
def test_counts_shapes_units(cls):
    model, batch = cls(), FakeBatch()
    task = type("Task", (), {"model": model})()
    rec = trace.TraceRecorder(model, record_encoder_input=True)
    with torch.inference_mode(), rec:
        trace.forward_genco(task, batch)
    assert rec.n_calls["node_residuals_layer"] == T
    assert rec.n_calls["branch_flow_layer"] == T
    assert rec.n_calls["physics_mlp"] == T - 1
    tb = rec.to_trace_batch(batch, base_mva=100.0, rate_a_mva=np.full(4, 50.0))
    assert tb.res_p.shape == (T, NB) and tb.loading.shape == (T, 4)
    np.testing.assert_allclose(tb.res_p[:, 0], 100.0 * 2.0 ** -np.arange(T), rtol=1e-6)
    np.testing.assert_allclose(tb.va[:, 0], np.arange(T), atol=1e-5)  # degrees
    assert tb.bus_type.tolist() == ["PQ", "PQ", "PV"]
    assert tb.struct_q.tolist() == [False, False, True]
    assert tb.scenario.tolist() == [42]


def test_hooks_removed_on_exit_and_on_error():
    model, batch = NewAPI(), FakeBatch()
    task = type("Task", (), {"model": model})()
    rec = trace.TraceRecorder(model)
    with pytest.raises(RuntimeError):
        with rec:
            raise RuntimeError("boom")
    for m in (model.node_residuals_layer, model.branch_flow_layer, model.physics_mlp):
        assert len(m._forward_hooks) == 0 and len(m._forward_pre_hooks) == 0


def test_reset_clears():
    model, batch = NewAPI(), FakeBatch()
    task = type("Task", (), {"model": model})()
    rec = trace.TraceRecorder(model)
    with torch.inference_mode(), rec:
        trace.forward_genco(task, batch)
        rec.reset()
        trace.forward_genco(task, batch)
    assert rec.n_calls["node_residuals_layer"] == T
