import pytest


def pytest_addoption(parser):
    # Options for tests/quafu/algorithms/qnn_test.py
    parser.addoption("--epoch", action="store", default=1, help="The number of epochs")
    parser.addoption("--bsz", action="store", default=1, help="Batch size")


@pytest.fixture
def num_epochs(request):
    return int(request.config.getoption("--epoch"))


@pytest.fixture
def batch_size(request):
    return int(request.config.getoption("--bsz"))
