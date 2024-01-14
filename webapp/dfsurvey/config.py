"""App configs."""
import logging
import os
from pathlib import Path
from typing import List, Optional

import toml

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def float_or_default(env_var: str, default: Optional[float]) -> Optional[float]:
    """If the variable is specified in the environment, return it as an float value.
    Otherwise the default value is used.
    """
    var = os.environ.get(env_var)
    if var:
        return float(var)
    return default


def int_or_default(env_var: str, default: Optional[int]) -> Optional[int]:
    """If the variable is specified in the environment, return it as an int value.
    Otherwise the default value is used.
    """
    var = os.environ.get(env_var)
    if var:
        return int(var)
    return default


def _log_level() -> str:
    log_level_str = os.environ.get("LOG_LEVEL")
    if log_level_str:
        try:
            log_level = getattr(logging, log_level_str)
        except AttributeError:
            raise ValueError(
                "Logging level not supported: %s", log_level_str)

    else:
        log_level = logging.INFO

    return log_level


def _data_dir() -> Path:
    data_dir = os.environ.get('DATA_DIR')
    return Path(data_dir) if data_dir else Path.cwd().joinpath("data")


def _survey_steps(survey_type: str) -> List[str]:
    steps = ["questionnaire.demographics",
             "questionnaire.presurvey", "index.instruction"]
    if survey_type == "likert" or survey_type == "scale":
        steps.append("experiments.random_scale")
    elif survey_type == "comparison":
        steps.append("experiments.random_comparison")
    else:
        raise NotImplementedError("Survey type not supported!")

    steps += ["questionnaire.postsurvey", "index.final"]

    return steps


class Config:
    """Basic config for app."""
    # general setup
    SECRET_KEY = os.environ.get('SECRET_KEY') or b'42'
    DISABLE_REDIRECT = os.environ.get("DISABLE_REDIRECT") is not None or False

    # Links for redirecting user
    RESET_PAGE = True
    TIMEOUT: int = int_or_default("TIMEOUT", 300)  # 5 minutes

    QUOTA_FULL: Optional[str] = os.environ.get(
        "QUOTA_FULL") or "reset?reason=quota_full"
    EARLY_SCREEN_OUT_URL: Optional[str] = os.environ.get(
        "EARLY_SCREEN_OUT_URL") or "reset?reason=early_screen_out"
    DATA_QUALITY_URL: Optional[str] = os.environ.get(
        "DATA_QUALITY_URL") or "reset?reason=failed_attention"
    FINISHED_URL: Optional[str] = os.environ.get(
        "FINISHED_URL") or "reset?reason=finished_survey"

    if TIMEOUT >= 60 and TIMEOUT % 60 != 0:
        raise ValueError(
            "Timeout must either be below 60s or be evenly divisible by 60!")

    # DB
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # helps with db disconnects
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # setup logging
    LOG_LEVEL = _log_level()

    # mail config
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_SENDER = os.environ.get('MAIL_SENDER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    ADMINS = [mail for mail in (os.environ.get("ADMINS") or "").split(",")]

    # Experiment steps
    DATA_DIR = _data_dir()
    SURVEY_TYPE = os.environ.get("SURVEY_TYPE") or "likert"
    SURVEY_STEPS = _survey_steps(SURVEY_TYPE)
    LANGUAGE = os.environ.get("SURVEY_LANG") or "english"
    MEDIA_OVERWRITE = os.environ.get("MEDIA_OVERWRITE")

    EXPERIMENTS_TO_PERFORM = int_or_default("EXPERIMENTS_TO_PERFORM", 2)
    AUDIO_EXPERIMENTS = int_or_default("AUDIO_EXPERIMENTS", None)
    IMAGE_EXPERIMENTS = int_or_default("IMAGE_EXPERIMENTS", None)
    TEXT_EXPERIMENTS = int_or_default("TEXT_EXPERIMENTS", None)

    QUESTION_PATH = os.environ.get("QUESTION_PATH") or "questions"
    QUOTA_PATH = os.environ.get("QUOTA_PATH") or "quotas"
    QUOTA_SKIP = os.environ.get("QUOTA_SKIP") is not None  # if set -> no check
    QUOTA_MARGIN = float_or_default("QUOTA_MARGIN", 0.03)

    # Expose unsafe api sites (stats/download results)
    # should only be run on locally accessable services
    EXPOSE_UNSAFE = os.environ.get("EXPOSE_UNSAFE") is not None

    @ staticmethod
    def init_app(app):
        """Easier access for frequently used configurations.
        """
        def set_attr(name, attr_str):
            attr = app.config[attr_str]
            assert not hasattr(app, name)
            setattr(app, name, attr)
            del app.config[attr_str]

        set_attr("survey_steps", "SURVEY_STEPS")
        set_attr("experiments_to_perform",
                 "EXPERIMENTS_TO_PERFORM")

        set_attr("audio_experiments",
                 "AUDIO_EXPERIMENTS")
        set_attr("image_experiments",
                 "IMAGE_EXPERIMENTS")
        set_attr("text_experiments",
                 "TEXT_EXPERIMENTS")

        # questions
        questions_path = f"{app.config['QUESTION_PATH']}/{app.config['LANGUAGE']}.toml"
        app.questions = toml.load(questions_path)

        app.likert = app.questions["experiment_scales"]["options"]

        # parse questions as one mapping for keeping some question bateries together when we randomize
        app.questions_as_one = dict(app.questions["config"]["consider_as_one"])

        del app.questions["experiment_scales"]

        # quoataas
        quota_path = f"{app.config['QUOTA_PATH']}/{app.config['LANGUAGE']}.toml"
        app.quotas_raw = toml.load(quota_path)

        # set log level
        app.logger.setLevel(app.config["LOG_LEVEL"])


class DevConfig(Config):
    """Development config."""
    DEBUG = True
    DEVELOPMENT_MODE = True
    TEMPLATES_AUTO_RELOAD = True

    SQLALCHEMY_DATABASE_URI = "sqlite:////tmp/test.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = True


class StageConfig(Config):
    """Configuration for staging test.
    """
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")

    SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db/{POSTGRES_DB}"


class ExportConfig(StageConfig):
    """Config for exporting results.
    """
    SQLALCHEMY_DATABASE_URI = f"postgresql://{StageConfig.POSTGRES_USER}:{StageConfig.POSTGRES_PASSWORD}@localhost/{StageConfig.POSTGRES_DB}"


class ProdConfig(StageConfig):
    """Production config.
    """
    SESSION_COOKIE_SECURE = True
    DISABLE_REDIRECT = False

    # redirect
    RESET_PAGE = os.environ.get("RESET_PAGE") is not None or False

    # Force setting by user in production
    QUOTA_FULL = os.environ.get("QUOTA_FULL")
    EARLY_SCREEN_OUT_URL = os.environ.get("EARLY_SCREEN_OUT_URL")
    DATA_QUALITY_URL = os.environ.get("DATA_QUALITY_URL")
    FINISHED_URL = os.environ.get("FINISHED_URL")


CONFIGS = {
    'development': DevConfig,
    'stage': StageConfig,
    'production': ProdConfig,
    'export': ExportConfig,

    'default': DevConfig


}
