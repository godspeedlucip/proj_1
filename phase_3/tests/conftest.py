import pytest
import torch


def pytest_configure(config):
    for marker in ("monai", "prior_artifact", "gpu", "slow"):
        config.addinivalue_line("markers", f"{marker}: explicit FDK acceptance tier")


@pytest.fixture(autouse=True)
def fdk_cpu_threads(request):
    if request.node.path.name.startswith("test_fdk_"):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        yield
        torch.set_num_threads(previous)
    else:
        yield
