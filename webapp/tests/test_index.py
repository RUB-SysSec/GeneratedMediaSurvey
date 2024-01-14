import json
import uuid

import pytest
from dfsurvey.app.factory import create_app
from dfsurvey.models import db
from dfsurvey.models.factory import init_db
from dfsurvey.models.user import User
from dfsurvey.views import SESS_ACC_COND, SESS_STEP, SESS_USER_ID
from flask import current_app, session

from test_selenium import SKIP_SELENIUM_TESTS, driver, live_server
from test_utils import TestConfig, client, initialized_client, set_session


def consent_precondition(user_id: int = 1):
    """Verify session and user creation after the first call to consent.
    """
    # verify session
    assert SESS_USER_ID in session
    assert not session[SESS_ACC_COND]
    assert session[SESS_STEP] == -1

    # verify created user
    user_id = session[SESS_USER_ID]
    user = User.query.get(user_id)
    assert user.id == user_id
    assert not user.accepted_conditions


def consent_postcondition():
    """Verify that the user has been accepted correctly.
    """
    user = User.query.get(session[SESS_USER_ID])
    assert user.accepted_conditions
    assert session[SESS_ACC_COND]
    assert session[SESS_STEP] == 0


def final_postcondition(user_id: int):
    """Check that the uuid is displayed on the page.
    """
    user = User.query.get(user_id)
    assert user.finished


def test_user_dynamically_created(client):
    """Check that the user gets dynamically created and the session
    gets correctly initialized.
    """
    _ = client.get("/consent")

    consent_precondition(1)
    assert len(User.query.all()) == 1


def test_user_dynamically_created_with_uuid_as_param(client):
    """Check that the user gets dynamically created and the session
    gets correctly initialized when the uuid is supplied as a parameter.
    """
    uuid_ref = uuid.uuid4()
    _ = client.get(f"/consent?id={uuid_ref}")

    consent_precondition(1)
    assert len(User.query.all()) == 1

    user = User.query.get(1)
    assert user.uuid == str(uuid_ref)


def test_user_already_exists(client):
    """Test that we cannot start a new session with the same id.
    """
    uuid_ref = uuid.uuid4()
    _ = client.get(f"/consent?id={uuid_ref}")

    consent_precondition(1)
    with client.session_transaction() as sess:
        sess.clear()

    res = client.get(f"/consent?id={uuid_ref}")
    assert current_app.config["EARLY_SCREEN_OUT_URL"] in res.location


def test_consent_accept(client):
    """Test that accepting consent works.
    """
    res = client.get("/consent")
    assert res.status_code == 200
    consent_precondition(1)

    res = client.put("/consent", content_type="application/json",
                     data=json.dumps({"decision": "accept"}))
    assert res.status_code == 200
    consent_postcondition()


def test_final_uuid(initialized_client):
    """Test that uuid is correctly displayed.
    """
    set_session(initialized_client, "index.final", res_code=302)
    user_id = session[SESS_USER_ID]
    res = initialized_client.get("/final")
    assert res.status_code == 302

    final_postcondition(user_id)


def test_final_finish_url(initialized_client):
    """Test that the finish url is correctly used.
    """
    config = TestConfig()
    config.FINISHED_URL = "test"
    app = create_app(config)

    with app.test_client() as client:
        with app.app_context():
            init_db()

            _ = client.get("/consent")
            _ = client.put("/consent", content_type="application/json",
                           data=json.dumps({"decision": "accept"}))

            with client.session_transaction() as sess:
                assert "index.final" in current_app.survey_steps
                sess[SESS_STEP] = current_app.survey_steps.index("index.final")

            res = client.get("/final")
            assert res.status_code == 302
            assert "test" in res.location


def test_final_finish_url_with_other_parameter():
    """Test that parameters get correctly recognized.
    """
    config = TestConfig()
    config.FINISHED_URL = f"{config.FINISHED_URL}?test=321"
    app = create_app(config)

    with app.test_client() as client:
        with app.app_context():
            init_db()

            _ = client.get("/consent")
            _ = client.put("/consent", content_type="application/json",
                           data=json.dumps({"decision": "accept"}))

            with client.session_transaction() as sess:
                assert "index.final" in current_app.survey_steps
                sess[SESS_STEP] = current_app.survey_steps.index("index.final")

            res = client.get("/final")
            assert res.status_code == 302
            assert f"{current_app.config['FINISHED_URL']}&id={session[SESS_USER_ID]}" in res.location


if not SKIP_SELENIUM_TESTS:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait


@pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
def test_selenium_accept_conditions(live_server, driver: webdriver.Chrome):
    """Test that users can accept conditions.
    """
    url = live_server.server_url()
    driver.get(f"{url}/consent")

    # check that first part is visible and second is not
    assert driver.is_displayed("introduction_text")
    assert not driver.is_displayed("disclaimer_container")

    driver.click("continue")

    assert not driver.is_displayed("introduction_text")
    assert driver.is_displayed("disclaimer_container")

    # accept conditions, check that we get forwarded correctly
    driver.click("consentCheckboxAccept")
    driver.click("accept_conditions")
    driver.wait_for_url_change()

    assert driver.current_url == f"{url}/questionnaire_demographics"

    # check that we wrote everything to db
    with live_server.app().app_context():
        users = User.query.all()
        assert len(users) == 1

        user = users[0]
        assert user.accepted_conditions


@pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
def test_selenium_reject_conditons(live_server, driver: webdriver.Chrome):
    """Test that users can accept conditions.
    """
    url = live_server.server_url()
    driver.get(f"{url}/consent")

    # check that first part is visible and second is not
    assert driver.is_displayed("introduction_text")
    assert not driver.is_displayed("disclaimer_container")

    driver.click("continue")

    assert not driver.is_displayed("introduction_text")
    assert driver.is_displayed("disclaimer_container")

    # accept conditions, check that we get forwarded correctly
    driver.click("consentCheckboxReject")
    driver.click("accept_conditions")
    driver.wait_for_url_change()

    assert "early_screen_out" in driver.current_url
