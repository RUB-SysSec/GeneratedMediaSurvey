import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pytest
from dfsurvey.app.factory import create_app
from dfsurvey.config import DevConfig
from dfsurvey.models.factory import init_db
from dfsurvey.views import SESS_STEP
from flask import current_app, url_for
from flask.sessions import SecureCookieSessionInterface
from flask.testing import FlaskClient
from itsdangerous import URLSafeTimedSerializer

MEDIA_TO_TEST = ["text", "audio", "image"]
TEST_TYPES_TO_TEST = ["comparison", "likert", "scale"]
LANGUAGES_TO_TEST = ["german", "chinese", "english"]

MEDIA_AND_TEST_TYPES = [(media, test)
                        for media in MEDIA_TO_TEST for test in TEST_TYPES_TO_TEST]

TEST_RAND_CHOICES = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut interdum eros id gravida euismod. Sed pretium est dui, non malesuada neque ultrices at. Praesent tincidunt libero massa, ut efficitur libero consequat eu."]


class FlaskUnsigner(SecureCookieSessionInterface):
    """A class for deserializing the cookie value.
    """

    def get_signing_serializer(self, secret_key: str) -> Optional[URLSafeTimedSerializer]:
        """Overwrite method to take the secret key instead of an actual app.
        """
        if not secret_key:
            return None

        signer_kwargs = dict(
            key_derivation=self.key_derivation,
            digest_method=self.digest_method
        )

        return URLSafeTimedSerializer(
            secret_key,
            salt=self.salt,
            serializer=self.serializer,
            signer_kwargs=signer_kwargs
        )


class TestConfig(DevConfig):
    """Testing config."""
    TESTING = True

    DATA_DIR = Path(__file__).parent.parent.joinpath(Path("data"))
    QUESTION_PATH = Path(__file__).parent.parent.joinpath(Path("questions"))
    QUOTA_PATH = Path(__file__).parent.parent.joinpath(Path("quotas"))

    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

    # Turn on to inspect queries to db
    # SQLALCHEMY_ECHO = True
    SURVEY_TYPE = "scale"

    LIMIT_DATA = 5

    TIMEOUT = 300

    steps = ["questionnaire.demographics", "questionnaire.presurvey"]
    for media in MEDIA_TO_TEST:
        steps.append(f"experiments.{media}_scale")
        steps.append(f"experiments.{media}_comparison")

    steps.append("experiments.random_scale")
    steps.append("experiments.random_comparison")
    steps.append("questionnaire.postsurvey")
    steps.append("index.final")

    SURVEY_STEPS = steps
    LOG_LEVEL = logging.WARN
    BUILD_CLI = False


class IntegrationConfig(TestConfig):
    """The common class for integration tests.
    """
    SURVEY_STEPS = [
        "questionnaire.demographics",
        "questionnaire.presurvey",
        "experiments.random_scale",
        "questionnaire.postsurvey",
        "index.final"
    ]

    SURVEY_TYPE = "likert"
    LIMIT_DATA = None
    EXPERIMENTS_TO_PERFORM = 7

    IMAGE_EXPERIMENTS = 10

    QUOTA_FULL = "reset?reason=quota_full"
    EARLY_SCREEN_OUT_URL = "reset?reason=early_screen_out"

    DATA_QUALITY_URL = "reset?reason=failed_attention"
    FINISHED_URL = "reset?reason=finished_survey"


@pytest.fixture
def request_context():
    """Yields a request context.
    """
    app = create_app(TestConfig())

    with app.test_request_context():
        with app.app_context():
            init_db()

            yield app


@pytest.fixture
def client():
    """Yields a new test client.
    """
    app = create_app(TestConfig())

    with app.test_client() as test_client:
        with app.app_context():
            init_db()

            yield test_client


@pytest.fixture
def initialized_client(client):
    """Yields an already initalized client.
    Note that we do not correctly perform the step,
    we just set the corresponding session items.
    """
    _ = client.get("/consent")
    _ = client.put("/consent", content_type="application/json",
                   data=json.dumps({"decision": "accept"}))

    yield client


def set_session(client: FlaskClient, endpoint: str, res_code: int = 200, key_value_pairs: Optional[Dict] = None):
    """Point the session to the provided endpoint.
    """
    with client.session_transaction() as sess:
        assert endpoint in current_app.survey_steps
        sess[SESS_STEP] = current_app.survey_steps.index(endpoint)
        if key_value_pairs:
            for key, val in key_value_pairs.items():
                sess[key] = val

        url = url_for(endpoint)
    res = client.get(url)
    assert res.status_code == res_code
