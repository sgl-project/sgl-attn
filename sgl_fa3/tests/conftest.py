import pytest
import torch


# Keeps torch defaults from leaking between tests (e.g. when a test fails before
# restoring them), which would make later tests fail. Carried over from
# sglang-kernel's tests/conftest.py together with test_flash_attention.py.
@pytest.fixture(autouse=True)
def reset_torch_defaults():
    orig_default_device = torch.get_default_device()
    orig_default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig_default_dtype)
    torch.set_default_device(orig_default_device)
