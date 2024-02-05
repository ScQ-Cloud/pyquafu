import pytest


def pytest_addoption(parser):
    parser.addoption("--epoch", action="store", default=1, help="The number of epochs")


@pytest.fixture
def num_epochs(request):
    return int(request.config.getoption("--epoch"))
