"""App factory for initializing an app context."""
import logging
import os
from logging.handlers import RotatingFileHandler, SMTPHandler
from pathlib import Path
from typing import Dict, Optional, Union

from dfsurvey.api_1_0.api import api_bp
from dfsurvey.app.cli import deploy_cli, export_cli
from dfsurvey.config import CONFIGS, Config
from dfsurvey.models import db
from dfsurvey.utils import strip_including
from dfsurvey.views.experiments import experiments_bp
from dfsurvey.views.index import index_bp, reset_bp
from dfsurvey.views.questionnaire import questionnaire_bp
from flask import Flask
from flask.logging import default_handler


def _parse_file_path(file_path: Path) -> str:
    name = strip_including(str(file_path), "/")

    gen_start = name.find("_gen")
    if gen_start > 0:
        name = name[:gen_start] + name[gen_start + 4:]

    return name


def _load_dir(path: Path, limit: Optional[int] = None) -> Dict[str, Path]:
    res = {}
    for file_path in path.glob("*"):
        name = _parse_file_path(file_path)
        res[name] = file_path

        if limit and len(res) > limit:
            break

    return res


def _load_data(root: Path = Path("data"), limit: Optional[int] = None) -> Dict:
    """Recursively load data directories.
    """
    # read in directories
    directories = []
    for path in root.iterdir():
        if path.is_dir() and "." not in str(path):
            name = strip_including(str(path), "/")
            directories.append((name, path))

    if len(directories) == 0:
        # hit rock bottom
        return _load_dir(root, limit=limit)

    res = {}
    for name, path in directories:
        res[name] = _load_data(path)

    return res


def __setup_logging(app: Flask):
    default_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s]:  %(message)s")

    # setup wsgi handler
    wsgi_handler = logging.StreamHandler()
    wsgi_handler.setFormatter(default_formatter)

    # Overwrite root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(app.config["LOG_LEVEL"])
    root_logger.addHandler(wsgi_handler)

    # We have own logging in place, remove default
    app.logger.removeHandler(default_handler)

    # setup email logging
    if app.config["MAIL_SERVER"]:
        auth = None
        if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
            auth = (app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        secure = None
        if app.config['MAIL_USE_TLS']:
            secure = ()
        mail_handler = SMTPHandler(
            mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
            fromaddr=app.config["MAIL_SENDER"],
            toaddrs=app.config['ADMINS'], subject='[DFSURVEY FAILURE]',
            credentials=auth, secure=secure)
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)
        app.logger.info("Setup mail logging!")

    # setup file logging
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/dfsurvey.log', maxBytes=10240,
                                       backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info("Setup logging complete!")


def create_app(
    config_or_name: Optional[Union[str, Config]] = None,
) -> Flask:
    """Initialize app context.
    """
    if config_or_name:
        config = CONFIGS[config_or_name]() if isinstance(
            config_or_name, str) else config_or_name
    elif os.environ.get("FLASK_CONFIG"):
        config = CONFIGS[os.environ.get("FLASK_CONFIG")]()
    else:
        raise AttributeError("Could not initialize config!")

    app = Flask(__name__, template_folder="../templates",
                static_folder="../static")

    # inital config
    app.config.from_object(config)
    config.init_app(app)

    __setup_logging(app)

    app.cli.add_command(deploy_cli)
    app.cli.add_command(export_cli)

    # Database
    db.init_app(app)

    # experiment data
    app.experiment_data = _load_data(
        app.config["DATA_DIR"], limit=app.config.get("LIMIT_DATA"))

    # blueprints
    if app.config["EXPOSE_UNSAFE"]:
        app.register_blueprint(api_bp)
    else:
        app.register_blueprint(index_bp)
        app.register_blueprint(questionnaire_bp)
        app.register_blueprint(experiments_bp)

    if app.config["RESET_PAGE"]:
        app.register_blueprint(reset_bp)

    if app.config["EARLY_SCREEN_OUT_URL"] is None \
            or app.config["FINISHED_URL"] is None \
            or app.config["QUOTA_FULL"] is None:
        raise RuntimeError("Redirect urls not set!")

    return app
