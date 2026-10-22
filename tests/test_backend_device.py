import sys
from types import SimpleNamespace

import pytest
import torch

from unilab.base import backend_factory


@pytest.mark.parametrize("requested", ["cuda", "cuda:1"])
def test_mjwarp_binds_warp_to_the_same_index_as_torch(monkeypatch, requested):
    selected = [2]
    calls = []

    def set_torch_device(device):
        assert device.index is not None, "torch.cuda.set_device rejects an unspecified index"
        selected[0] = device.index
        calls.append(("torch", device.index))

    monkeypatch.setattr(torch.cuda, "set_device", set_torch_device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: selected[0])
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(set_device=lambda device: calls.append(("warp", device))),
    )
    monkeypatch.setattr(
        backend_factory.unisim,
        "create_backend",
        lambda *args, **kwargs: calls.append(("backend", None)),
    )

    backend_factory.create_backend("mjwarp", None, 1, 0.005, tensor_device=requested)

    expected = [("torch", 1)] if requested == "cuda:1" else []
    index = 1 if requested == "cuda:1" else 2
    assert calls == [*expected, ("warp", f"cuda:{index}"), ("backend", None)]
