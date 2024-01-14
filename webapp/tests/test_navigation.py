import pytest
from dfsurvey.views import SESS_ACC_COND, SESS_STEP, next_step

from test_selenium import SKIP_SELENIUM_TESTS
from test_utils import client


def test_redirect(client):
    """Check that the user gets redirected when they have not
    accepted the conditions."""

    res = client.get("/questionnaire_demographics")
    assert res.status_code == 302
    assert res.location == "http://localhost/consent"

    with client.session_transaction() as sess:
        # "accept" conditions
        sess[SESS_ACC_COND] = True

        # manually set correct step since testing
        url = next_step()
        sess[SESS_STEP] = 0

    res = client.get(url)
    assert res.status_code == 200
    assert res.location is None


def test_can_only_post_to_current_step(client):
    """Cannot POST to other step.
    """
    res = client.post("/questionnaire_demographics")

    assert res.status_code == 403


if not SKIP_SELENIUM_TESTS:
    from test_selenium import initalized_server_and_driver


@pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@pytest.mark.parametrize(
    "initalized_server_and_driver",
    [
        {
            "SURVEY_STEPS": ["experiments.image_comparison", "index.final"],
        },
    ],
    indirect=True,
)
def test_selenium_user_cannot_go_back(initalized_server_and_driver):
    """Test that users cannot go back.
    """
    _live_server, driver = initalized_server_and_driver

    url = driver.current_url
    driver._driver.back()
    assert url == driver.current_url
