import pytest
import torch

from unilab.training.tensor_diagnostics import NanGuardCfg, TensorNanGuard


@pytest.mark.parametrize("check", ["control", "observations"])
def test_nan_guard_detects_any_nonfinite_element_in_each_row(check):
    values = torch.tensor(
        [[1.0, float("nan")], [float("inf"), 2.0], [float("nan"), float("nan")], [1.0, 2.0]]
    )
    guard = TensorNanGuard(NanGuardCfg(enabled=True), 4, supports_state_playback=False)
    detected = (
        guard.check_ctrl(values)
        if check == "control"
        else guard.check({"policy": values}, torch.zeros(4))
    )
    assert torch.equal(detected, torch.tensor([0, 1, 2]))


def test_nan_guard_returns_none_for_finite_rows():
    guard = TensorNanGuard(NanGuardCfg(enabled=True), 2, supports_state_playback=False)
    assert guard.check_ctrl(torch.ones(2, 6)) is None
    assert guard.check({"policy": torch.ones(2, 35)}, torch.zeros(2)) is None
