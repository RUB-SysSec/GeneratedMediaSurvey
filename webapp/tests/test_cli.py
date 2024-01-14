import os
import tempfile

from dfsurvey.app.factory import create_app

from test_utils import TestConfig


def test_db_creation():
    with tempfile.NamedTemporaryFile() as tmp:
        config = TestConfig()
        config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp.name}"

        app = create_app(config)
        runner = app.test_cli_runner()

        res = runner.invoke(args=["deploy", "build"])

        assert res.exit_code == 0
        assert os.path.getsize(tmp.name) > 0


def test_db_deletion():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Fake DB
        config = TestConfig()
        config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp.name}"

        app = create_app(config)
        runner = app.test_cli_runner()

        res = runner.invoke(args=["deploy", "clean"])

        assert res.exit_code == 0
        assert not os.path.exists(tmp.name)
