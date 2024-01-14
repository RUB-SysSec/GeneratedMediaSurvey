import json
import random
from collections import Counter, deque
from datetime import timedelta
from pathlib import Path

import pytest
from dfsurvey.app.factory import create_app
from dfsurvey.app.questions import build_questions_and_options_mapping
from dfsurvey.app.quotas import (map_age_to_option_id,
                                 option_to_query_and_amount,
                                 quota_to_amount_mapping, quota_to_idx_mapping,
                                 remap_mapping)
from dfsurvey.models import db
from dfsurvey.models.factory import init_db
from dfsurvey.models.question import OptionAnswer
from dfsurvey.models.user import User
from flask import current_app

from test_integration import simulate_clients
from test_selenium import SKIP_SELENIUM_TESTS
from test_utils import IntegrationConfig, TestConfig, request_context


def test_quota_to_id_mapping(request_context):
    """Test if the function calculates the correct keys.
    """
    mapping = quota_to_idx_mapping()
    questions = current_app.questions

    for key, ids in mapping.items():
        if key in ["low", "medium", "high"]:
            assert len(questions["demographics"]["demographics"]
                       ["questions"]["education"][key]) == len(ids)
        else:
            assert len(ids) == 1


def test_option_combination_to_query_and_amount(request_context):
    """Test if the key to amount mapping matches
    """
    mapping = option_to_query_and_amount()
    quota_to_amount = quota_to_amount_mapping()
    quota_to_idx = quota_to_idx_mapping()

    # check if ids + amount match
    for quota, amount in quota_to_amount.items():
        quota_idx = quota_to_idx[quota]

        for idx in quota_idx:
            assert mapping[idx][1] == amount


def test_query_simple_case():
    """Test single query works.
    """
    config = IntegrationConfig()
    config.TIMEOUT = 20  # reduce timeout to 5 seconds
    app = create_app(config)
    app.quotas_raw["general"]["n"] = 1

    with app.test_request_context():
        with app.app_context():
            init_db()

            for _ in range(50):
                # create a new user
                user = User.create_new_user()
                user.finish()

                question_to_id, option_to_id, _ = build_questions_and_options_mapping()
                remapping = remap_mapping()

                picked = []
                answers = []
                # pick random answers
                for data in current_app.questions["demographics"]["demographics"]["questions"].values():
                    question_id = question_to_id[data["question"]]
                    option = random.choice(
                        data.get("options") or data["low"] + data["medium"] + data["medium"])
                    option_id = option_to_id[option]
                    option_id = remapping.get(option_id) or option_id

                    answer = OptionAnswer(
                        user_id=user.id,
                        question_id=question_id,
                        option_id=option_id,
                    )
                    db.session.add(
                        answer
                    )
                    answers.append(answer)

                    picked.append(option_id)

                db.session.commit()
                idx_to_query_and_amount = option_to_query_and_amount()
                for idx in picked:
                    query, amount = idx_to_query_and_amount[idx]
                    count = query.count()
                    assert count == 1
                    assert count == amount

                for answer in answers:
                    db.session.delete(answer)
                db.session.delete(user)
                db.session.commit()


def test_query_remapping(request_context):
    """Test that remapping the queries work.
    """
    user = User.create_new_user()
    user.finish()

    question_to_id, option_to_id, _ = build_questions_and_options_mapping()

    regular_ids = [
        option_to_id["Male"],
        option_to_id["40-49"],
        option_to_id["Bachelor's degree"],
        option_to_id["White"],
    ]

    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your gender?"],
            option_id=regular_ids[0],
        )
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["How old are you?"],
            option_id=regular_ids[1],
        )
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your highest level of education obtained?"],
            option_id=regular_ids[2],
        )
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your ethnicity?"],
            option_id=regular_ids[3],
        )
    )

    remapped_ids = [
        option_to_id["Non-binary"],
        option_to_id["40-49"],
        option_to_id["Bachelor's degree"],
        option_to_id["White"],
    ]

    # create different user
    user = User.create_new_user()
    user.finish()

    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your gender?"],
            option_id=remapped_ids[0],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["How old are you?"],
            option_id=remapped_ids[1],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your highest level of education obtained?"],
            option_id=remapped_ids[2],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your ethnicity?"],
            option_id=regular_ids[3],
        )
    )
    db.session.commit()

    idx_to_query_and_amount = option_to_query_and_amount()
    for idx in regular_ids:
        query, _ = idx_to_query_and_amount[idx]
        assert query.count() == 2

    for idx in remapped_ids:
        query, _ = idx_to_query_and_amount[idx]
        assert query.count() == 2


def test_query_timeout(request_context):
    """Test that users who timeout are not considered anymore.
    """
    # create different user
    user = User.create_new_user()
    question_to_id, option_to_id, _ = build_questions_and_options_mapping()

    ids = [
        option_to_id["Male"],
        option_to_id["40-49"],
        option_to_id["Bachelor's degree"],
        option_to_id["White"],
    ]

    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your gender?"],
            option_id=ids[0],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["How old are you?"],
            option_id=ids[1],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your highest level of education obtained?"],
            option_id=ids[2],
        ),
    )
    db.session.add(
        OptionAnswer(
            user_id=user.id,
            question_id=question_to_id["What is your ethnicity?"],
            option_id=ids[3],
        )
    )
    db.session.commit()

    # User still in timeout limit
    idx_to_qa = option_to_query_and_amount()
    for idx in ids:
        query, _ = idx_to_qa[idx]
        assert query.count() == 1

    # User is very late
    user.last_seen -= timedelta(days=1)
    db.session.commit()
    assert query.count() == 0


def test_quota_correctly_calculated():
    """Run in integration test and verify that the quotas are correctly applied.
    """
    config = IntegrationConfig()
    config.TIMEOUT = 20  # reduce timeout to 5 seconds
    app = create_app(config)
    app.quotas_raw["general"]["n"] = 100

    clients = simulate_clients(app, 200)

    with app.app_context():
        # all user either complete the first step or give up in the beginning
        for user in User.query.all():
            assert user.accepted_conditions

        # collect all picked combinations
        combination_ctx = Counter()
        for client in clients:
            if client.finished:
                remapping = remap_mapping()
                comb = []
                comb = [remapping.get(
                    val) or val for val in client.questionnaire_picks["demographics"].values()]
                combination_ctx[tuple(sorted(comb))] += 1

        assert sum(combination_ctx.values()) == User.query.filter(
            User.finished).count()

        idx_to_query_and_amount = option_to_query_and_amount()
        for combination, amount in combination_ctx.items():
            for idx in combination:
                query, _ = idx_to_query_and_amount[idx]

                assert query.count() >= amount  # did not count to few


# n = 1
# categories default to 1 participant
QUOTA_PATH = Path(__file__).parent.parent.joinpath(Path("quotas/test"))


def test_user_gets_redirected():
    """Test that user gets redirect when quota is full.
    """
    config = IntegrationConfig()
    config.QUOTA_PATH = QUOTA_PATH
    app = create_app(config)

    with app.app_context():
        init_db()

        # prepare data
        question_to_id, option_to_id, _ = build_questions_and_options_mapping()

        question_ids = (
            question_to_id["What is your gender?"],
            question_to_id["How old are you?"],
            question_to_id["What is your highest level of education obtained?"],
            question_to_id["What is your ethnicity?"],
        )
        option_ids = (
            option_to_id["Male"],
            option_to_id["40-49"],
            option_to_id["Bachelor's degree"],
            option_to_id["White"],
        )

        answers = []
        for question_id, option_id in zip(question_ids, option_ids):
            answers.append((question_id, {
                "option_id": option_id,
                "question_type": "options",
            }))
        data = {
            "category": "demographics",
            "answers": answers,
        }

        # check at least 1
        for idx in option_ids:
            assert option_to_query_and_amount()[
                idx][1] == 1

        with app.test_client() as client:
            # accept conditions
            res = client.get("/consent")
            assert res.status_code == 200

            res = client.put("/consent", content_type="application/json",
                             data=json.dumps({"decision": "accept"}))
            assert res.status_code == 200

            # set session variables
            res = client.get("questionnaire_demographics")
            assert res.status_code == 200

            # submit result
            res = client.post("questionnaire_demographics",
                              data=json.dumps(data),
                              content_type="application/json",
                              )
            assert res.status_code == 200

            for idx in option_ids:
                assert option_to_query_and_amount()[
                    idx][0].count() == 1

            ret_data = json.loads(res.data.decode("utf-8"))
            assert "presurvey" in ret_data["url"]

            # user passes
            res = client.get("questionnaire_presurvey")
            assert res.status_code == 200

            User.query.get(1).finish()  # user passes survey

            # create new user
            with client.session_transaction() as sess:
                sess.clear()

            # accept conditions
            res = client.get("/consent")
            assert res.status_code == 200
            res = client.put("/consent", content_type="application/json",
                             data=json.dumps({"decision": "accept"}))
            assert res.status_code == 200

            # user gets redirected to reset page
            ret_data = json.loads(res.data.decode("utf-8"))
            assert "reset" in ret_data["url"]

            # user invalid
            res = client.get("questionnaire_presurvey")
            assert res.status_code == 302
            assert "consent" in res.location


def test_age_to_option_id():
    """Test age to option_id mapping.
    """
    app = create_app(TestConfig())
    with app.app_context():
        init_db()

        _,  option_to_id, _ = build_questions_and_options_mapping()
        assert map_age_to_option_id(18) == option_to_id["18-29"]
        assert map_age_to_option_id(23) == option_to_id["18-29"]
        assert map_age_to_option_id(29) == option_to_id["18-29"]
        assert map_age_to_option_id(30) == option_to_id["18-29"] + 1


if not SKIP_SELENIUM_TESTS:
    import time

    from selenium.webdriver.common.by import By

    from test_selenium import initalized_server_and_driver


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver",
    [
        (
            {
                "QUOTA_PATH": QUOTA_PATH,
                "SURVEY_STEPS": ["questionnaire.demographics", "index.final"],
            },
            {
                "mobile": False,
            }
        ),
        (
            {
                "QUOTA_PATH": QUOTA_PATH,
                "SURVEY_STEPS": ["questionnaire.demographics", "index.final"],
            },
            {
                "mobile": True,
            }
        ),
    ],
    indirect=True,
)
def test_selenium_questionnaire_submit(initalized_server_and_driver):
    """Test that submitting the questionnaire works.
    """
    live_server, driver = initalized_server_and_driver
    assert "questionnaire" in driver.current_url

    # perform questionnaire
    cat = driver._driver.find_elements(By.CLASS_NAME, "category")[0]
    for question in cat.find_elements(By.CLASS_NAME, "question"):
        question_type = question.get_attribute("data-type")
        if question_type == "age":
            scale = question.find_element(By.TAG_NAME, "input")
            driver._driver.execute_script(
                "arguments[0].value = arguments[1]", scale, 20)
        else:
            # always pick first option
            driver.click_element(
                question.find_elements(By.TAG_NAME, "label")[0])

    # move forwards
    driver.click("continue")
    time.sleep(driver.wait_amount)

    # works
    assert "reset" in driver.current_url

    # new run
    driver._driver.delete_all_cookies()

    url = live_server.server_url()
    driver.get(f"{url}/consent")

    driver.accept_conditions()

    # perform questionnaire
    cat = driver._driver.find_elements(By.CLASS_NAME, "category")[0]
    for question in cat.find_elements(By.CLASS_NAME, "question"):
        question_type = question.get_attribute("data-type")
        if question_type == "age":
            scale = question.find_element(By.TAG_NAME, "input")
            driver._driver.execute_script(
                "arguments[0].value = arguments[1]", scale, 20)
        else:
            # always pick first option
            driver.click_element(
                question.find_elements(By.TAG_NAME, "label")[0])

    # move forwards
    driver.click("continue")
    time.sleep(driver.wait_amount)

    # another user already doing the survey
    assert "reset?reason=quota_full" in driver.current_url


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver",
    [
        (
            {
                "SURVEY_STEPS": ["questionnaire.demographics", "index.final"],
            },
            {
                "mobile": False,
            }
        ),
        (
            {
                "SURVEY_STEPS": ["questionnaire.demographics", "index.final"],
            },
            {
                "mobile": True,
            }
        ),
    ],
    indirect=True,
)
def test_selenium_under_18_rejected(initalized_server_and_driver):
    """Test that under 18 year olds get rejected
    """
    live_server, driver = initalized_server_and_driver
    assert "questionnaire" in driver.current_url

    # perform questionnaire
    cat = driver._driver.find_elements(By.CLASS_NAME, "category")[0]
    for question in cat.find_elements(By.CLASS_NAME, "question"):
        question_type = question.get_attribute("data-type")
        if question_type == "age":
            scale = question.find_element(By.TAG_NAME, "input")
            driver._driver.execute_script(
                "arguments[0].value = arguments[1]", scale, 17)
        else:
            # always pick first option
            driver.click_element(
                question.find_elements(By.TAG_NAME, "label")[0])

    # move forwards
    driver.click("continue")
    time.sleep(driver.wait_amount)

    # works
    assert "reset?reason=early_screen_out" in driver.current_url
