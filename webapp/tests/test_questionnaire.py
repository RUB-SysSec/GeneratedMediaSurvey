"""Test the automatic questionnaire generation."""
import json
import random
import re
from typing import Dict, List, Optional, Set, Tuple

import pytest
from bs4 import BeautifulSoup
from dfsurvey.app.factory import create_app
from dfsurvey.app.questions import build_questions_and_options_mapping
from dfsurvey.app.quotas import map_age_to_option_id
from dfsurvey.models.factory import init_db
from dfsurvey.models.question import (FloatAnswer, IntegerAnswer, Option,
                                      OptionAnswer, Question, TextAnswer,
                                      Timer, all_answers)
from dfsurvey.models.user import User
from dfsurvey.views import (SESS_ACC_COND, SESS_MEDIA_CHOOSEN, SESS_STEP,
                            SESS_USER_ID)
from dfsurvey.views.questionnaire import (SESS_ATTENTION_CORRECT_KEY, SESS_CTX,
                                          SESS_ORDER, SESS_TIMER)
from flask import current_app, session
from flask.testing import FlaskClient

from test_selenium import SKIP_SELENIUM_TESTS
from test_utils import (TEST_RAND_CHOICES, TestConfig, client,
                        initialized_client, set_session)
from tests.test_selenium import live_server


def _collect_items_of_key(data: Dict, key: str, list_only: bool = False) -> Set:
    if key in data:
        key_data = data[key]
        if isinstance(key_data, List):
            return set(key_data)
        elif list_only:
            return set()
        elif isinstance(key_data, str):
            return set([key_data])
        else:
            raise ValueError(f"Unsupported type for collecting: {key_data}")

    res: Set[str] = set()
    for possible_dict in data.values():
        if isinstance(possible_dict, Dict):
            res = res.union((_collect_items_of_key(
                possible_dict, key, list_only=list_only)))

    return res


def questionnaire_preconditions():
    """Preconditions for the questionnaire
    """
    assert SESS_TIMER in session
    assert SESS_CTX in session


def questionnaire_postconditions(ref: Dict, check_only_user: bool = False, check_equal_amount: bool = True):
    """Assert quetionnaire postconditions.
    """
    # check if correctly stored; delete all found ones
    user_id = session[SESS_USER_ID]

    saved_questions = {}
    for query in [OptionAnswer.query, IntegerAnswer.query, TextAnswer.query, FloatAnswer.query]:
        if check_only_user:
            query = query.filter_by(user_id=user_id)

        saved_questions.update({
            x.question_id: x for x in query.all()
        })
    ids = list(ref.keys())

    if check_equal_amount:
        assert len(saved_questions) == len(ref)

    for question_id, value in ref.items():
        saved = saved_questions[question_id]
        if isinstance(saved, OptionAnswer):
            assert value == saved.option_id
        else:
            assert value == saved.answer

        ids.remove(question_id)

    # all found
    assert len(ids) == 0

    # session cleanup
    assert SESS_TIMER not in session
    assert SESS_CTX not in session

    timer = Timer.query.filter_by(user_id=user_id).first()
    assert timer.finished


def _search_attention_template(data: Dict) -> Optional[str]:
    if "question" in data:
        if "attention" in data["question"]:
            return data["question"]
        return None

    for val in data.values():
        if not isinstance(val, Dict):
            continue

        possible = _search_attention_template(val)
        if possible:
            return possible
    return None


def solve_attention_check(data: str) -> Optional[int]:
    """Solve the attention check.
    """
    soup = BeautifulSoup(data, "html.parser")
    _, options_to_id, _ = build_questions_and_options_mapping()

    for question in soup.find_all(class_="question"):
        if "attention" in question.text:
            question_text = question.find("p").text
            attention_template = _search_attention_template(
                current_app.questions)
            assert attention_template is not None

            attention_re = attention_template.replace(
                "{}", "(.*)").replace("?", "\?")

            answer = re.search(attention_re, question_text)[1]
            assert answer is not None

            return options_to_id[answer]

    return None


def complete_questionnaire(test_client: FlaskClient, category_str: str, key: str, attention_correct: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Complete the questionnaire for some client.
    """
    ref = {}
    questions_to_id, options_to_id, _ = build_questions_and_options_mapping()

    questions = test_client.application.questions[category_str][key]["questions"]

    answers = []
    for question in questions.values():
        name = question["question"]

        # collect possible answers
        if question["question_type"] == "options":
            options = list(map(options_to_id.get, question["options"]))
            key = "option_id"
        elif question["question_type"] == "education":
            options = list(
                map(options_to_id.get, question["low"] + question["medium"] + question["high"]))
            key = "option_id"
        elif question["question_type"] == "attention":
            key = "option_id"
            options = [attention_correct]
        elif question["question_type"] == "likert":
            options = list(map(
                options_to_id.get, test_client.application.questions["likert_scales"][question["options"]]["options"]))
            key = "option_id"
        elif question["question_type"] == "scale":
            options = list(range(0, 101))
            key = "value"
        elif question["question_type"] == "textfield":
            options = TEST_RAND_CHOICES
            key = "value"
        elif question["question_type"] == "number":
            options = list(range(5))
            key = "value"
        elif question["question_type"] == "age":
            options = list(range(18, 99))
            key = "value"

        option_id = random.choice(options)
        question_id = questions_to_id[name]

        answers.append((question_id, {
            key: option_id,
            "question_type": question["question_type"],
        }))

        if question["question_type"] == "age":
            option_id = map_age_to_option_id(option_id)
            ref[question_id] = option_id
        elif question["question_type"] != "attention":
            ref[question_id] = option_id

    data = {
        "category": category_str,
        "answers": answers,
    }

    return data, ref


STEPS = ["demographics", "presurvey", "postsurvey"]


@pytest.mark.parametrize("step", STEPS)
def test_invalid_post(initialized_client, step):
    """Test if we can post invalid data.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()

    res = initialized_client.post(f"questionnaire_{step}",
                                  data={},
                                  )

    assert res.status_code == 400
    assert b"No data provided!" in res.data


@pytest.mark.parametrize("step", STEPS)
def test_malformed(initialized_client, step):
    """Test if we can post malformed data.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()

    res = initialized_client.post(f"questionnaire_{step}",
                                  data=json.dumps(
                                      [() for _ in range(sum(map(len, initialized_client.application.questions.values())))]),
                                  content_type="application/json",
                                  )

    assert res.status_code == 400
    assert b"Data malformed!" in res.data


@pytest.mark.parametrize("step", STEPS)
def test_wrong_ids(initialized_client, step):
    """Test if we can post correctly formatted data, but with ids that do not exist.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()

    res = initialized_client.post(f"questionnaire_{step}",
                                  data=json.dumps(
                                      [[(random.randint(1, 100), random.randint(1, 100))] for _ in range(sum(map(len, initialized_client.application.questions.values())))]),
                                  content_type="application/json",
                                  )

    assert res.status_code == 400
    assert b"Data malformed!" in res.data


@pytest.mark.parametrize("step", STEPS)
def test_questions_timer_user_id_regression(initialized_client, step):
    """Regression where an incorrect timer was selected when two users
    fill out the questionnaire simultaneously.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()
    url = f"questionnaire_{step}"
    _ = initialized_client.get(url)
    timer_id = session[SESS_TIMER]

    # initialize new session
    with initialized_client.session_transaction() as sess:
        sess.clear()

    _ = initialized_client.get(url)

    # "accept" conditions
    with initialized_client.session_transaction() as sess:
        sess[SESS_ACC_COND] = True
        sess[SESS_STEP] = current_app.survey_steps.index(
            f"questionnaire.{step}")

    res = initialized_client.get(url)
    assert session[SESS_USER_ID] == 2
    assert session[SESS_TIMER] != timer_id

    timer_id = session[SESS_TIMER]

    attention_correct = None
    res_data = res.data.decode(encoding=res.charset)
    if "attention check" in res_data:
        attention_correct = solve_attention_check(
            res_data)

    for key in session[SESS_ORDER]:
        data, _ = complete_questionnaire(
            initialized_client, step, key, attention_correct=attention_correct)

        res = initialized_client.post(url,
                                      data=json.dumps(data),
                                      content_type="application/json",
                                      )
        assert res.status_code == 200, res.data

    timers = Timer.query.all()

    assert timers[0].user_id == 1
    assert not timers[0].finished

    assert timers[1].user_id == 2
    assert timers[1].finished

    # session cleanup
    assert SESS_TIMER not in session


@pytest.mark.parametrize("step", STEPS)
def test_questions_stored_correctly(initialized_client, step):
    """Test that the questions get storred correctly.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()
    url = f"questionnaire_{step}"
    res = initialized_client.get(url)

    assert res.status_code == 200
    questionnaire_preconditions()

    attention_correct = None
    res_data = res.data.decode(encoding=res.charset)
    if "attention check" in res_data:
        attention_correct = solve_attention_check(
            res_data)

    ref = {}
    for key in session[SESS_ORDER]:
        data, key_ref = complete_questionnaire(
            initialized_client, step, key, attention_correct=attention_correct)

        res = initialized_client.post(url,
                                      data=json.dumps(data),
                                      content_type="application/json",
                                      )
        assert res.status_code == 200

        ref.update(key_ref)

    questionnaire_postconditions(ref)


def test_questions_random_order(client):
    """Test thaat we properly randomize the order of questions.
    """
    step = "postsurvey"

    def authenticate_and_extract_order():
        # authenitcate client
        _ = client.get("/consent")
        _ = client.put("/consent", content_type="application/json",
                       data=json.dumps({"decision": "accept"}))

        # warp to postsurvey
        set_session(client, f"questionnaire.{step}")
        url = f"questionnaire_{step}"
        res = client.get(url)

        assert res.status_code == 200

        return session[SESS_ORDER]

    keep_order = current_app.questions_as_one[step]

    seen = set()
    for _ in range(10):
        cur = tuple(authenticate_and_extract_order())

        # assert not seen before
        assert cur not in seen
        seen.add(cur)

        # check that same stay together and in order
        for i, token in enumerate(cur):
            for same in keep_order:
                # found first token for same
                if token == same[0]:
                    assert list(cur[i:i+len(same)]) == same
                    break

        with client.session_transaction() as sess:
            sess.clear()


def test_questions_attention_failed(initialized_client):
    """Test user gets redirected when he failed the attention check.
    """
    step = "presurvey"
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()
    url = f"questionnaire_{step}"
    res = initialized_client.get(url)

    assert res.status_code == 200
    questionnaire_preconditions()

    ctx = 1
    supplied_link = False
    while res.status_code == 200:
        attention_correct = session.get(SESS_ATTENTION_CORRECT_KEY)
        if attention_correct:
            attention_correct += 1
        data, _ = complete_questionnaire(
            initialized_client, step, f"deepfakes_{ctx}", attention_correct=attention_correct)

        res = initialized_client.post(url,
                                      data=json.dumps(data),
                                      content_type="application/json",
                                      )
        assert res.status_code == 200
        if "url" in res.json:
            assert res.json[
                "url"] == f"{current_app.config['DATA_QUALITY_URL']}?id={User.query.get(1).uuid}"
            supplied_link = True

        res = initialized_client.get(url)
        ctx += 1

    assert "consent" in res.location

    user = User.query.get(1)
    assert not user.passed_attention
    assert supplied_link


@ pytest.mark.parametrize("step", STEPS)
def test_questions_correctly_displayed(initialized_client, step):
    """Test html layout of the questionnaire.
    """
    set_session(initialized_client, f"questionnaire.{step}")
    questionnaire_preconditions()
    url = f"questionnaire_{step}"
    res = initialized_client.get(url)

    # parse html tree + setup
    questions_to_id, options_to_id, _ = build_questions_and_options_mapping()

    for key in session[SESS_ORDER]:
        res = initialized_client.get(url)
        res_data = res.data.decode(encoding=res.charset)

        attention_correct = None
        if "attention check" in res_data:
            attention_correct = solve_attention_check(
                res_data)

        data, _ = complete_questionnaire(
            initialized_client, step, key, attention_correct=attention_correct)

        soup = BeautifulSoup(res_data, "html.parser")

        questionnaire = soup.find(id="questionnaire")
        current_category = questionnaire.find_next("div")

        # check category
        category_data = initialized_client.application.questions[step][key]
        if "description" in category_data:
            assert current_category.find_next(
                class_="category-description").text == category_data["description"]

        # iterate over questions in category
        current_question = current_category.find_next("div")

        for question in category_data["questions"].values():
            question_type = question["question_type"]
            if question_type == "attention":
                continue

            name = question["question"]
            question_id = questions_to_id[name]

            # check that question in displaed
            headline = current_question.find_next("p")
            assert headline.contents[0] == name

            question_type = question["question_type"]
            if question_type == "textfield":
                textarea = current_question.find_next(
                    "textarea")
                assert textarea is not None
                assert textarea.get(
                    "data-question-id") == str(questions_to_id[name])
                assert textarea.get(
                    "data-question-type") == question_type
                assert textarea.get(
                    "id") == f"question_{question_id}_textarea"
            elif question_type == "number" or question_type == "age":
                scale = current_question.find_next(
                    "input")
                assert scale is not None
                assert scale.get(
                    "data-question-id") == str(questions_to_id[name])
                assert scale.get(
                    "data-question-type") == question_type
                assert scale.get(
                    "id") == f"question_{question_id}_number"
            elif question_type == "scale":
                scale = current_question.find_next(
                    "input")
                assert scale is not None
                assert scale.get(
                    "data-question-id") == str(questions_to_id[name])
                assert scale.get(
                    "data-question-type") == question_type
                assert scale.get(
                    "id") == f"question_{question_id}_scale"
            else:
                # collect possible answers
                if question_type == "options":
                    options = question["options"]
                elif question_type == "education":
                    options = question["low"] + \
                        question["medium"] + question["high"]
                elif question_type == "likert":
                    options = initialized_client.application.questions[
                        "likert_scales"][question["options"]]["options"]

                # iterate over options
                current_option = current_question.find_next(
                    "div", class_="option")

                for option in options:
                    option_id = options_to_id[option]
                    idx = f"question_{question_id}_{option_id}"

                    assert idx == current_option.find_next("input").get("id")

                    label = current_option.find_next("label")
                    assert idx == label.get("for")
                    assert str(option) in label.text

                    current_option = current_option.find_next_sibling("div")

            current_question = current_question.find_next_sibling("div")

        # post results and complete current stage
        res = initialized_client.post(url,
                                      data=json.dumps(data),
                                      content_type="application/json",
                                      )
        assert res.status_code == 200


def test_db_questions_init():
    """Test questions correctly inserted into the db.
    """
    app = create_app(TestConfig())

    with app.app_context():
        init_db()

        questions = _collect_items_of_key(app.questions, "question")
        for question in Question.query.all():
            questions.remove(question.text)

        # all questions fond
        assert len(questions) == 0

        options = _collect_items_of_key(
            app.questions, "options", list_only=True)
        options = options.union(_collect_items_of_key(app.questions, "low"))
        options = options.union(_collect_items_of_key(app.questions, "medium"))
        options = options.union(_collect_items_of_key(app.questions, "high"))

        for option in Option.query.all():
            options.remove(option.text)

        assert len(options) == 0, options


def test_audio_conditional_displayed(initialized_client):
    """Test that audio question is conditionally displayed.
    """
    set_session(initialized_client, "questionnaire.postsurvey",
                key_value_pairs={SESS_MEDIA_CHOOSEN: "audio"})

    res = initialized_client.get("questionnaire_postsurvey")
    assert res.status_code == 200

    assert "Audio" in session[SESS_ORDER]


def test_audio_conditional_not_displayed(initialized_client):
    """Test that audio question is conditionally displayed.
    """
    set_session(initialized_client, "questionnaire.postsurvey",
                key_value_pairs={SESS_MEDIA_CHOOSEN: "text"})

    res = initialized_client.get("questionnaire_postsurvey")
    assert res.status_code == 200

    assert "Audio" not in session[SESS_ORDER]


if not SKIP_SELENIUM_TESTS:
    import random

    from test_selenium import initalized_server_and_driver


def _build_parameter():
    res = []

    for steps in [["questionnaire.demographics", "index.final"], ["questionnaire.presurvey", "index.final"], ["questionnaire.postsurvey", "index.final"]]:
        for mobile in [True, False]:
            res.append((
                {
                    "SURVEY_STEPS": steps,
                },
                {
                    "mobile": mobile,
                }
            ))

    return res


SELENIUM_PARAMETER = _build_parameter()


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver",
    SELENIUM_PARAMETER,
    indirect=True,


)
def test_selenium_questionnaire_missing_answers(initalized_server_and_driver):
    """Test not answering displays the error message.
    """
    live_server, driver = initalized_server_and_driver
    assert "questionnaire" in driver.current_url

    with live_server.app().app_context():
        assert len(OptionAnswer.query.all()) == 0
        assert len(IntegerAnswer.query.all()) == 0
        assert len(TextAnswer.query.all()) == 0

    driver.click("continue")

    with live_server.app().app_context():
        assert len(OptionAnswer.query.all()) == 0
        assert len(IntegerAnswer.query.all()) == 0
        assert len(TextAnswer.query.all()) == 0

    assert driver.is_displayed("error")


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver",
    SELENIUM_PARAMETER,
    indirect=True,
)
def test_selenium_questionnaire_submit(initalized_server_and_driver):
    """Test that submitting the questionnaire works.
    """
    live_server, driver = initalized_server_and_driver
    assert "questionnaire" in driver.current_url

    # precondition
    with live_server.app().app_context():
        assert len(OptionAnswer.query.all()) == 0
        assert len(IntegerAnswer.query.all()) == 0
        assert len(TextAnswer.query.all()) == 0

    # perform questionnaire
    driver.questionnaire()

    # check correctly stored
    with live_server.app().app_context():
        # remove age variable
        age = User.query.get(1).age
        if age:
            driver.elements_choosen.remove(str(age))

        # collect possible options
        possible_age_options = set(
            current_app.questions["demographics"]["demographics"]["questions"]["age"]["options"])

        for option in all_answers():
            assert option.user_id == 1

            if isinstance(option, OptionAnswer):
                if option.option.text in possible_age_options:
                    continue
                answer = str(option.option_id)
            elif isinstance(option, FloatAnswer):
                answer = str(int(option.answer))
            else:
                answer = str(option.answer)
            assert answer in driver.elements_choosen
            driver.elements_choosen.remove(answer)

        timers = Timer.query.all()
        assert len(timers) == 1

        timer = timers[0]
        assert timer.start_time < timer.end_time

    # all stored
    assert len(driver.elements_choosen) == 0


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver",
    [

        {
            "SURVEY_STEPS": ["questionnaire.presurvey", "index.final"],
        }
    ],
    indirect=True,
)
def test_selenium_attention_failed(initalized_server_and_driver):
    """Test that submitting the questionnaire works.
    """
    _live_server, driver = initalized_server_and_driver
    assert "questionnaire" in driver.current_url

    # perform questionnaire
    driver.pass_attention = False
    driver.questionnaire()

    assert "failed_attention" in driver.current_url
