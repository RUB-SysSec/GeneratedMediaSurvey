import json
import random
import time
from typing import List, Optional

import dfsurvey
import pytest
from bs4 import BeautifulSoup
from dfsurvey.app.factory import create_app
from dfsurvey.models.experiment import ExperimentCounter, Guess, Rating
from dfsurvey.models.factory import init_db
from dfsurvey.views.experiments import (SESS_CORRECT, SESS_EXP_ID, SESS_FILE,
                                        SESS_FILE_FAKE, SESS_FILE_REAL,
                                        SESS_MEDIA, SESS_NUM_EXP,
                                        SESS_SCALE_ORDER, __load_file)
from flask import current_app, session
from werkzeug.test import TestResponse

from test_selenium import SKIP_SELENIUM_TESTS
from test_utils import (MEDIA_AND_TEST_TYPES, MEDIA_TO_TEST, TestConfig,
                        client, initialized_client, set_session)


def precondition_all():
    """Preconditions used in all experiments.
    """
    assert session[SESS_NUM_EXP] == 1


def likert_precondition(media: str):
    """Preconditions for likert based tests.
    """
    assert media in session[SESS_FILE]

    precondition_all()


def comparison_precondition(media: str):
    """Precondition for comparison based tests.
    """
    assert media in session[SESS_FILE_REAL]
    assert media in session[SESS_FILE_FAKE]
    assert SESS_CORRECT in session

    precondition_all()


def postcondition_all():
    """Postconditions for everything.
    """
    assert SESS_NUM_EXP not in session
    assert SESS_MEDIA not in session  # technically only for random experiments
    assert SESS_EXP_ID not in session


def likert_postcondition():
    """Postconditions for likert experiments.
    """
    assert SESS_FILE_REAL not in session
    assert SESS_FILE_FAKE not in session
    assert SESS_CORRECT not in session

    postcondition_all()


def comparison_postcondition():
    """Postconditions for comparison experiments.
    """
    assert SESS_FILE not in session

    postcondition_all()


def pick_random_option(res: TestResponse):
    """Pick a random availble option.
    """
    if current_app.config["SURVEY_TYPE"] == "likert":
        data = res.data.decode(encoding=res.charset)
        soup = BeautifulSoup(data, "html.parser")
        choice = random.choice(list(soup.find_all("input")))
        choice = choice.get("data-choice")
    else:
        choice = random.randint(0, 100)

    return {"choice": choice}


def check_guess(
    choice: str,
    correct: str,
    file_path_fake: str,
    file_path_real: str,
    index: Optional[int] = None,
):
    """Check if a guess is correct.
    """
    guesses = Guess.query.all()
    guess = guesses[-1]
    assert guess.correct == (choice == correct)
    assert guess.file_path_fake == file_path_fake
    assert guess.file_path_real == file_path_real

    if index:
        assert guess.index_of_experiment == index


def check_rating(
    choice: int,
    file_path: str,
    index: Optional[int] = None,
):
    """Check that ratings are correct.
    """
    ratings = Rating.query.all()
    rating = ratings[-1]
    assert rating.rating == choice
    assert rating.file_path == file_path

    if index:
        assert rating.index_of_experiment == index


@ pytest.mark.parametrize(
    "exp_type",
    ["comparison", "scale"],
)
def test_random_experiment(initialized_client, exp_type):
    """Test that we get all types of media when requesting random experiments.
    """
    set_session(initialized_client, f"experiments.random_{exp_type}")

    res = initialized_client.get(f"/random_{exp_type}")
    assert res.status_code == 200
    assert SESS_MEDIA in session

    seen = set()
    for _ in range(15):
        seen.add(session[SESS_MEDIA])

        if b"img" in res.data:
            for _ in range(10):
                res = initialized_client.get(f"/random_{exp_type}")
                assert b"img" in res.data
        elif b"audio" in res.data:
            for _ in range(10):
                res = initialized_client.get(f"/random_{exp_type}")
                assert b"audio" in res.data

        with initialized_client.session_transaction() as sess:
            del sess[SESS_MEDIA]
            if SESS_SCALE_ORDER in sess:
                del sess[SESS_SCALE_ORDER]

        res = initialized_client.get(f"/random_{exp_type}")

    assert len(seen) == 3
    assert "text" in seen
    assert "image" in seen
    assert "audio" in seen


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_get(initialized_client, media):
    """Test if endpoint is available and session is set.
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    res = initialized_client.get(f"/{media}_scale")
    assert res.status_code == 200
    likert_precondition(media)


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_get_same(initialized_client, media):
    """Test tat endpoint always returns same html.
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    res = initialized_client.get(f"/{media}_scale")
    assert res.status_code == 200

    data = res.data.decode(encoding=res.charset)

    # reload
    res = initialized_client.get(f"/{media}_scale")
    assert res.status_code == 200
    assert data == res.data.decode(encoding=res.charset)
    likert_precondition(media)


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_counter(initialized_client, media):
    """Test that the result gets stored correctly
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    url = f"/{media}_scale"
    res = initialized_client.get(url)
    choice = pick_random_option(res)

    with initialized_client.session_transaction() as sess:
        test_file_path = f"/data/{media}/fake/something.exe"
        sess[SESS_FILE] = test_file_path

    data = choice
    ctx_ref = random.randint(1, 5)
    data["count"] = ctx_ref

    res = initialized_client.post(url,
                                  data=json.dumps(data),
                                  content_type="application/json",
                                  )

    assert res.status_code == 200

    ratings = Rating.query.all()
    assert len(ratings) == 1
    check_rating(
        choice=int(choice["choice"]),
        file_path=test_file_path,
        index=1,
    )

    counter = ExperimentCounter.query.all()
    assert len(counter) == 1
    assert counter[0].count == ctx_ref

    assert session[SESS_NUM_EXP] == 1
    assert SESS_FILE not in session


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_post(initialized_client, media):
    """Test that the result gets stored correctly
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    url = f"/{media}_scale"
    res = initialized_client.get(url)
    choice = pick_random_option(res)

    with initialized_client.session_transaction() as sess:
        test_file_path = f"/data/{media}/fake/something.exe"
        sess[SESS_FILE] = test_file_path

    data = choice
    res = initialized_client.post(url,
                                  data=json.dumps(data),
                                  content_type="application/json",
                                  )

    assert res.status_code == 200

    ratings = Rating.query.all()
    assert len(ratings) == 1
    check_rating(
        choice=int(choice["choice"]),
        file_path=test_file_path,
        index=1,
    )

    assert session[SESS_NUM_EXP] == 1
    assert SESS_FILE not in session


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_order_randomized(initialized_client, media):
    """Assert that media order is randomized.
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    url = f"/{media}_scale"

    orders = set()

    for _ in range(10):
        res = initialized_client.get(url)
        assert res.status_code == 200

        assert session[SESS_SCALE_ORDER] not in orders
        orders.add(session[SESS_SCALE_ORDER])

        with initialized_client.session_transaction() as sess:
            del sess[SESS_SCALE_ORDER]


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_order_unique(media, monkeypatch):
    """Assert that the order of media shown is unique
    """
    config = TestConfig()
    config.EXPERIMENTS_TO_PERFORM = 5
    app = create_app(config)

    with app.test_client() as test_client:
        with app.app_context():
            init_db()

            _ = test_client.get("/consent")
            _ = test_client.put("/consent", content_type="application/json",
                                data=json.dumps({"decision": "accept"}))

            url = f"/{media}_scale"
            set_session(test_client, f"experiments.{media}_scale")

            # Monkeypatch load_file to trace filepaths which get accessed
            seen = set()

            def trace_load_file(media_tyoe, path):
                assert path not in seen
                seen.add(path)
                return __load_file(media_tyoe, path)

            monkeypatch.setattr(dfsurvey.views.experiments,
                                "__load_file", trace_load_file)

            for _ in range(current_app.experiments_to_perform - 1):
                res = test_client.get(url)
                assert res.status_code == 200

                choice = pick_random_option(res)
                res = test_client.post(url,
                                       data=json.dumps(choice),
                                       content_type="application/json",
                                       )

                assert res.status_code == 200


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_likert_redirect(initialized_client, media):
    """Check that we get redirected when we complete the exerpiment
    """
    set_session(initialized_client, f"experiments.{media}_scale")
    url = f"/{media}_scale"
    res = initialized_client.get(url)
    assert res.status_code == 200

    with initialized_client.session_transaction() as sess:

        test_file_path = f"/data/{media}/fake/something.exe"

        sess[SESS_FILE] = test_file_path
        sess[SESS_NUM_EXP] = 9

    data = pick_random_option(res)

    res = initialized_client.post(url,
                                  data=json.dumps(data),
                                  content_type="application/json",
                                  )

    assert res.status_code == 200
    assert b"move_on" in res.data
    likert_postcondition()


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_comparison_get(initialized_client, media):
    """Test that we can fetch a comparison and the session is correctly initialized.
    """
    set_session(initialized_client, f"experiments.{media}_comparison")
    res = initialized_client.get(f"/{media}_comparison")
    assert res.status_code == 200
    comparison_precondition(media)


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_comparison_post(initialized_client, media):
    """Check that the correct results get saved.
    """
    set_session(initialized_client, f"experiments.{media}_comparison")

    for i, choice in enumerate(["left", "right"], 1):
        url = f"/{media}_comparison"
        _ = initialized_client.get(url)

        with initialized_client.session_transaction() as sess:
            test_file_path_fake = f"/data/{media}/fake/something.exe"
            test_file_path_real = f"/data/{media}/real/something.exe"

            sess[SESS_FILE_FAKE] = test_file_path_fake
            sess[SESS_FILE_REAL] = test_file_path_real

            sess[SESS_CORRECT] = "left"

        # user picks correctly
        data = {"choice": choice}

        res = initialized_client.post(url,
                                      data=json.dumps(data),
                                      content_type="application/json",
                                      )

        assert res.status_code == 200

        # check Guess correctly created
        guesses = Guess.query.all()
        assert len(guesses) == i

        check_guess(
            choice=choice,
            correct="left",
            file_path_fake=test_file_path_fake,
            file_path_real=test_file_path_real,
            index=i,
        )

        # check session clean up
        assert SESS_FILE_FAKE not in session
        assert SESS_FILE_REAL not in session
        assert SESS_CORRECT not in session


@ pytest.mark.parametrize(
    "media",
    MEDIA_TO_TEST,
)
def test_comparison_redirect(initialized_client, media):
    """Check that we get redirected after comparison experiment.
    """
    set_session(initialized_client, f"experiments.{media}_comparison")
    url = f"/{media}_comparison"
    _ = initialized_client.get(url)

    with initialized_client.session_transaction() as sess:
        test_file_path_fake = f"/data/{media}/fake/something.exe"
        test_file_path_real = f"/data/{media}/real/something.exe"

        sess[SESS_FILE_FAKE] = test_file_path_fake
        sess[SESS_FILE_REAL] = test_file_path_real
        sess[SESS_NUM_EXP] = current_app.experiments_to_perform

        sess[SESS_CORRECT] = "left"

    data = {"choice": "left"}
    res = initialized_client.post(url,
                                  data=json.dumps(data),
                                  content_type="application/json",
                                  )

    assert res.status_code == 200
    assert b"move_on" in res.data
    comparison_postcondition()


@ pytest.mark.parametrize(
    "media,test_type",
    filter(lambda x: x[1] != "scale", MEDIA_AND_TEST_TYPES)
)
def test_timestamp(initialized_client, media, test_type):
    """Test that the timestamp work correctly.
    """
    endpoint = f"experiments.{media}_{test_type}" if test_type == "comparison" else f"experiments.{media}_scale"
    set_session(initialized_client, endpoint)

    if test_type == "likert":
        cls = Rating
    elif test_type == "comparison":
        cls = Guess
    else:
        raise NotImplementedError(
            f"Testing for invalid test type: {test_type}")

    # start run
    url = f"/{media}_{test_type}" if test_type == "comparison" else f"/{media}_scale"
    _ = initialized_client.get(url)
    with initialized_client.session_transaction():
        timer = cls.query.get(1)
        assert len(cls.query.all()) == 1

    # complete step
    if test_type == "comparison":
        choice = random.choice(["left", "right"])
    elif test_type == "likert":
        choice = random.choice(
            list(range(1, len(current_app.likert))))
    else:
        raise NotImplementedError(
            "Comparison type not implemented yet!")

    data = {"choice": choice}
    time.sleep(1)
    res = initialized_client.post(url,
                                  data=json.dumps(data),
                                  content_type="application/json",
                                  )
    assert res.status_code == 200
    assert b"move_on" in res.data

    timer = cls.query.get(1)
    assert timer.end_time is not None
    assert timer.start_time < timer.end_time


if not SKIP_SELENIUM_TESTS:
    from test_selenium import initalized_server_and_driver, live_server


def _build_comparison_parameters():
    parameters = []
    for media in MEDIA_TO_TEST:
        for mobile in [True, False]:
            parameters.append(
                (
                    (
                        {
                            "SURVEY_STEPS": [f"experiments.{media}_comparison", "index.final"],
                        },
                        {
                            "mobile": mobile,
                        }
                    ),
                    media,
                )
            )
    return parameters


COMPARISON_PARAMETERS = _build_comparison_parameters()


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver,media_type",
    COMPARISON_PARAMETERS,
    indirect=["initalized_server_and_driver"],
)
def test_selenium_comparison_submit(initalized_server_and_driver, media_type):
    """Selenium based comparison test.
    """
    live_server, driver = initalized_server_and_driver
    assert f"{media_type}_comparison" in driver.current_url

    with live_server.app().app_context():
        assert len(Guess.query.all()) == 1
        steps = current_app.experiments_to_perform

    for i in range(1, steps + 1):
        driver.perform_experiment_step()

        with live_server.app().app_context():
            guesses = Guess.query.all()

            # new one already inserted
            # except last round since we get redirected to final screen
            if i == steps:
                assert len(guesses) == i
                guess = guesses[-1]
            else:
                assert len(guesses) == i+1
                guess = guesses[-2]

            assert guess.id == i
            assert guess.index_of_experiment == i
            assert guess.end_time is not None
            assert guess.end_time > guess.start_time
            assert media_type in guess.file_path_real
            assert media_type in guess.file_path_fake

    assert "finished_survey" in driver.current_url
    with live_server.app().app_context():
        assert len(Guess.query.all()) == steps


def _build_scale_parameters() -> List:
    parameters = []
    for media in MEDIA_TO_TEST:
        for mobile in [True, False]:
            parameters.append(
                (
                    (
                        {
                            "SURVEY_STEPS": [f"experiments.{media}_scale", "index.final"],
                            "SURVEY_TYPE": "scale",
                        },
                        {
                            "mobile": mobile,
                        }
                    ),
                    media,

                )
            )
            parameters.append(
                (
                    (
                        {
                            "SURVEY_STEPS": [f"experiments.{media}_scale", "index.final"],
                            "SURVEY_TYPE": "likert",
                        },
                        {
                            "mobile": mobile,
                        }
                    ),
                    media,

                )
            )

    return parameters


SCALE_PARAMETERS = _build_scale_parameters()


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver,media_type",
    SCALE_PARAMETERS,
    indirect=["initalized_server_and_driver"],
)
def test_selenium_scale_submit(initalized_server_and_driver, media_type):
    """Test that submitting a scale (likert) test works.
    """
    live_server, driver = initalized_server_and_driver
    assert f"{media_type}_scale" in driver.current_url

    with live_server.app().app_context():
        assert len(Rating.query.all()) == 1
        steps = current_app.experiments_to_perform

    for i in range(1, steps + 1):
        driver.perform_experiment_step()

        with live_server.app().app_context():
            ratings = Rating.query.all()

            if i == steps:
                assert len(ratings) == i
                rating = ratings[-1]
            else:
                assert len(ratings) == i+1
                rating = ratings[-2]

            assert rating.id == i
            assert rating.index_of_experiment == i
            assert rating.rating == driver.elements_choosen[i-1]
            assert rating.end_time is not None
            assert rating.end_time > rating.start_time

    assert "finished_survey" in driver.current_url
    with live_server.app().app_context():
        assert len(Rating.query.all()) == steps


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver,media_type,test_type",
    [
        # no point in testing scale, they have default values
        *[(*x, "likert")
          for x in filter(lambda x:  x[0][0]["SURVEY_TYPE"] == "likert", SCALE_PARAMETERS)],
        *[(*x, "comparison") for x in COMPARISON_PARAMETERS],
    ],
    indirect=["initalized_server_and_driver"],
)
def test_selenium_experiment_fail(initalized_server_and_driver, media_type, test_type):
    """Test that experiments fail correctly.
    """
    live_server, driver = initalized_server_and_driver

    def check_db():
        if test_type == "likert":
            ratings = Rating.query.all()
            assert len(ratings) == 1
            assert not ratings[-1].finished
            assert ratings[-1].end_time is None

        elif test_type == "comparison":
            guesses = Guess.query.all()
            assert len(guesses) == 1
            assert not guesses[-1].finished
            assert guesses[-1].end_time is None

        else:
            raise NotImplementedError(f"Unsupported test type: {test_type}")

    with live_server.app().app_context():
        check_db()

    driver.submit(check_error=False)

    with live_server.app().app_context():
        check_db()

    assert driver.is_displayed("error")


@ pytest.mark.skipif(SKIP_SELENIUM_TESTS, reason="Selenium not installed or available!")
@ pytest.mark.parametrize(
    "initalized_server_and_driver,exp_type",
    [
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_scale", "index.final"],
                    "SURVEY_TYPE": "scale",
                },
                {
                    "mobile": False,
                }
            ),
            "scale",
        ),
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_scale", "index.final"],
                    "SURVEY_TYPE": "likert",
                },
                {
                    "mobile": False,
                }
            ),
            "scale",
        ),
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_comparison", "index.final"],
                },
                {
                    "mobile": False,
                }
            ),
            "comparison",
        ),
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_scale", "index.final"],
                    "SURVEY_TYPE": "likert",
                },
                {
                    "mobile": True,
                }
            ),
            "scale",
        ),
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_scale", "index.final"],
                    "SURVEY_TYPE": "likert",
                },
                {
                    "mobile": True,
                }
            ),
            "scale",
        ),
        (
            (
                {
                    "SURVEY_STEPS": ["experiments.random_comparison", "index.final"],
                    "SURVEY_TYPE": "likert",
                },
                {
                    "mobile": True,
                }
            ),
            "comparison",
        ),
    ],
    indirect=["initalized_server_and_driver"],
)
def test_selenium_random_submit(initalized_server_and_driver, exp_type):
    """Test that submitting random experiments work.
    """
    live_server, driver = initalized_server_and_driver
    assert f"random_{exp_type}" in driver.current_url

    with live_server.app().app_context():
        if exp_type == "scale":
            assert len(Rating.query.all()) == 1
        else:
            assert len(Guess.query.all()) == 1

        steps = current_app.experiments_to_perform

    for i in range(1, steps + 1):
        driver.perform_experiment_step()

        with live_server.app().app_context():
            if exp_type == "scale":
                ratings = Rating.query.all()

                if i == steps:
                    assert len(ratings) == i
                    rating = ratings[-1]
                else:
                    assert len(ratings) == i+1
                    rating = ratings[-2]

                assert rating.id == i
                assert rating.index_of_experiment == i
                assert rating.rating == driver.elements_choosen[i-1]
                assert rating.end_time is not None
                assert rating.end_time > rating.start_time
            else:
                guesses = Guess.query.all()

                # new one already inserted
                # except last round since we get redirected to final screen
                if i == steps:
                    assert len(guesses) == i
                    guess = guesses[-1]
                else:
                    assert len(guesses) == i+1
                    guess = guesses[-2]

                assert guess.id == i
                assert guess.index_of_experiment == i
                assert guess.end_time is not None
                assert guess.end_time > guess.start_time

    assert "finished_survey" in driver.current_url
    with live_server.app().app_context():
        if exp_type == "scale":
            assert len(Rating.query.all()) == steps
        else:
            assert len(Guess.query.all()) == steps
