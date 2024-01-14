"""Implementation for Selenium-based unit testing."""
import importlib.util
import shutil

import pytest

from live_server import LiveServer
from tests.test_utils import TestConfig


@pytest.fixture
def live_server(request, config: None | TestConfig = None):
    """Fixture for spawning a live server.
    """
    if hasattr(request, "param"):
        config = request.param
    with LiveServer(config=config) as live_server:
        yield live_server


def test_live_server(live_server):
    """Test that spawning a live server works.
    """
    assert live_server.can_ping_server()
    assert live_server.db_uri.value != ""


# ==========================================================================================
# Selenium tests
# ==========================================================================================
SKIP_SELENIUM_TESTS = True


if importlib.util.find_spec("selenium") is not None \
        and shutil.which("chromedriver") is not None:
    from chrome_client import ChromeClient

    SKIP_SELENIUM_TESTS = False


@pytest.fixture
def driver(request):
    """Fixture for spawning a chrome driver.
    """
    kwargs = {}
    if hasattr(request, "param"):
        kwargs.update(request.param)
    driver = ChromeClient(**kwargs)
    yield driver
    driver.quit()


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
def test_selenium(live_server, driver):
    """Test that the driver can access the server.
    """
    url = live_server.server_url()
    driver.get(f"{url}/")

    # check the consent page got loaded
    assert driver.is_displayed("introduction_text")


@pytest.fixture
def initalized_server_and_driver(request):
    """Fixture for spawning a chrome driver which has already accepted the conditions.
    """
    config = TestConfig()
    # initialize a config for the server
    driver_args = {}

    if hasattr(request, "param"):
        params = request.param
        if isinstance(params, tuple):
            params, driver_args = params

        for key, value in params.items():
            setattr(config, key, value)

    driver = ChromeClient(**driver_args)

    with LiveServer(config=config) as live_server:
        # create user and move on to first step
        url = live_server.server_url()
        driver.get(f"{url}/consent")

        driver.accept_conditions()

        yield (live_server, driver)

    driver.quit()


@pytest.mark.parametrize(
    "initalized_server_and_driver",
    [
        {
            "SURVEY_TYPE": "comparison",
            "SURVEY_STEPS": ["experiments.image_comparison"],
        },
    ],
    indirect=True,
)
def test_initalized_server_and_driver(initalized_server_and_driver):
    """Test that accepting the conditions works for chrome driver-
    """
    live_server, driver = initalized_server_and_driver

    assert live_server.app().survey_steps == ["experiments.image_comparison"]
    assert "image_comparison" in driver.current_url
