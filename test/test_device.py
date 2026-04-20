import pytest

torch = pytest.importorskip("torch")

from diffpack.device import choose_torch_device, move_to_device


def test_move_to_device_tensor_cpu():
    x = torch.tensor([1, 2, 3])
    y = move_to_device(x, torch.device("cpu"))
    assert y.device.type == "cpu"


def test_move_to_device_nested():
    batch = {"x": torch.tensor([1]), "y": [torch.tensor([2])]}
    moved = move_to_device(batch, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["y"][0].device.type == "cpu"


def test_choose_torch_device_rejects_mps():
    with pytest.raises(RuntimeError, match="MPS is not supported"):
        choose_torch_device("mps")
