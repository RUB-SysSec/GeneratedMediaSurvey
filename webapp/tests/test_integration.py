import json
import random
import re
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from dfsurvey.app.cli import export
from dfsurvey.app.factory import create_app
from dfsurvey.models.factory import init_db
from dfsurvey.models.user import User
from dfsurvey.views import SESS_USER_ID
from dfsurvey.views.experiments import (SESS_CORRECT, SESS_FILE,
                                        SESS_FILE_FAKE, SESS_FILE_REAL,
                                        SESS_MEDIA, _amount_of_experiments)
from dfsurvey.views.questionnaire import SESS_ORDER
from flask import Flask, current_app, session
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from test_experiments import (check_guess, check_rating,
                              comparison_postcondition,
                              comparison_precondition, likert_postcondition,
                              likert_precondition, pick_random_option)
from test_index import (consent_postcondition, consent_precondition,
                        final_postcondition)
from test_questionnaire import (complete_questionnaire,
                                questionnaire_postconditions,
                                questionnaire_preconditions,
                                solve_attention_check)
from test_utils import IntegrationConfig


class TestClient:
    """A wrapper class for flask's TestClient.
    The class is mainly used to store some information about the client
    and the current state in the survey.
    """

    def __init__(self, client: FlaskClient) -> None:
        self.client = client

        # mostly used internally
        self.current_step: Optional[str] = None
        self.current_response: Optional[TestResponse] = None
        self.user_id: Optional[int] = None
        self.performed_experiments = 0

        # track when client is done
        self.stopped = False
        self.finished = False
        self.gave_up = False

        # Picks in the different questionnaires
        # questionnaire_category -> Dict[question_id, option_id]
        self.questionnaire_picks: Dict[str, Dict[int, int]] = {}
        self.ratings_given: Dict[int, int] = {}

        # Reason if client hits reset page
        self.reset: Optional[str] = None

    def step(self):
        """Take a random step in the survey.
        """

        if not self.stopped:
            random.choices(
                population=[
                    self.correct_step,
                    self.give_up
                ],
                weights=(.9, .1),
            )[0]()

    def correct_step(self):
        """Perform the correct step by first getting rerouted to the correct one.
        """
        self.get("/")

        if "consent" in self.current_step:
            self.accept_conditions()

        elif "questionnaire" in self.current_step:
            self.questionnaire()

        elif "comparison" in self.current_step:
            comparison_precondition(session[SESS_MEDIA])
            self.performed_experiments = 0
            self.experiment(current_app.config["SURVEY_TYPE"])
            comparison_postcondition()

        elif "scale" in self.current_step:
            likert_precondition(session[SESS_MEDIA])
            self.performed_experiments = 0
            self.experiment(current_app.config["SURVEY_TYPE"])
            likert_postcondition()

        elif "reset" in self.current_step:
            if "finished_survey" in self.current_step:
                final_postcondition(self.user_id)
                self.finished = True
                self.stopped = True
            else:
                self.reset_page()
                self.stopped = True

        else:
            raise NotImplementedError(
                f"No action implemented for {self.current_step}")

    def experiment(self, exp_type: str):
        """Perform a comparison between multiple objects.
        """
        for _ in range(_amount_of_experiments()):
            res = self.get(self.current_step)
            choice = pick_random_option(res)

            # collect information
            file_path_fake = session.get(SESS_FILE_FAKE)
            file_path_real = session.get(SESS_FILE_REAL)
            correct = session.get(SESS_CORRECT)
            file_path = session.get(SESS_FILE)

            res = self.client.post(self.current_step,
                                   data=json.dumps(choice),
                                   content_type="application/json")
            assert res.status_code == 200

            if choice != 0:
                self.performed_experiments += 1
                if exp_type == "likert" or exp_type == "scale":
                    choice = int(choice["choice"])
                    self.ratings_given[self.performed_experiments] = choice

                    check_rating(
                        choice=choice,
                        file_path=file_path,
                        index=self.performed_experiments,
                    )
                elif exp_type == "comparison":
                    check_guess(
                        choice=choice,
                        correct=correct,
                        file_path_fake=file_path_fake,
                        file_path_real=file_path_real,
                        index=self.performed_experiments,
                    )

    def questionnaire(self):
        """Perform the questionnaire by picking random answers.
        """
        res = self.get(self.current_step)
        res_data = res.data.decode(encoding=res.charset)

        attention_correct = None
        if "attention check" in res_data:
            attention_correct = solve_attention_check(
                res_data)

        questionnaire_preconditions()

        category = re.search("questionnaire_([a-zA-z]+)", self.current_step)[1]

        ref = {}
        for key in session[SESS_ORDER]:
            data, key_ref = complete_questionnaire(
                self.client, category, key, attention_correct=attention_correct)

            res = self.client.post(self.current_step,

                                   data=json.dumps(data),
                                   content_type="application/json",
                                   )
            assert res.status_code == 200
            ref.update(key_ref)

        self.questionnaire_picks[category] = ref

        ret_data = json.loads(res.data.decode("utf-8"))
        if "reset" not in ret_data["url"]:
            questionnaire_postconditions(
                ref, check_only_user=True, check_equal_amount=False)

    def accept_conditions(self):
        """Accept the conditions of the survey.
        """
        _ = self.get("/consent")
        self.user_id = int(session[SESS_USER_ID])

        # session object points to last active session
        consent_precondition(user_id=self.user_id)

        _ = self.client.put("/consent", content_type="application/json",
                            data=json.dumps({"decision": "accept"}))
        consent_postcondition()

    def give_up(self):
        """Client decides to give up.
        """
        self.stopped = True
        self.gave_up = True

    def reset_page(self):
        """Client hit the reset page.
        """
        data = self.current_response.data.decode(
            encoding=self.current_response.charset)
        soup = BeautifulSoup(data, "html.parser")
        self.reset = soup.find(id="reason").text

    def get(self, url: str) -> TestResponse:
        """Get request to the specified url.
        If we get redirected, fetch new content and set
        current_step accordingly.
        """
        res = self.client.get(url)

        counter = 0
        while res.location and counter < 5:
            url = res.location
            res = self.client.get(url)

            counter += 1

        assert res.status_code == 200, f"Url: {url}; Location: {res.location}"
        self.current_step = url
        self.current_response = res

        return res

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.client.__exit__(exc_type, exc_value, traceback)


def simulate_clients(app: Flask, amount_of_clients: int = 100) -> List[TestClient]:
    """Simulate a full run through of several clients.
    """
    with app.app_context():
        init_db()

        with ExitStack() as stack:
            clients = [stack.enter_context(TestClient(app.test_client()))
                       for _ in range(amount_of_clients)]

            while True:
                order = list(range(len(clients)))
                random.shuffle(order)
                for i in order:
                    clients[i].step()

                # check if all clients finished/gave up
                stop = True
                for client in clients:
                    stop &= client.stopped

                if stop:
                    break

    return clients


def test_multiple_clients(amount_of_clients: int = 100):
    """Test multiple clients to check for race conditions.
    """
    app = create_app(IntegrationConfig())
    _ = simulate_clients(app, amount_of_clients=amount_of_clients)

    with app.app_context():
        # all user either complete the first step or give up in the beginning
        for user in User.query.all():
            assert user.accepted_conditions

            if user.finished:
                if user.media_assigned == "image":
                    assert len(
                        user.ratings) == IntegrationConfig.IMAGE_EXPERIMENTS
                else:
                    assert len(
                        user.ratings) == IntegrationConfig.EXPERIMENTS_TO_PERFORM

        assert len(User.query.all()) != amount_of_clients
        assert len(User.query.all()) > 50

        runner = app.test_cli_runner()
        with NamedTemporaryFile() as tmp_file:
            runner.invoke(export, ["--output", tmp_file.name])

            tmp_file.seek(0)
            exported = json.load(tmp_file)

        completed = set(
            map(lambda u: u.id, User.query.filter_by(finished=True).all()))
        for exp in exported:
            idx = exp["id"]
            assert idx in completed
            completed.remove(idx)

        assert len(completed) == 0
