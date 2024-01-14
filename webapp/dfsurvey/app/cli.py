"""Cli commands for the survey application."""
import json
import os

import click
from dfsurvey.models.factory import init_db
from dfsurvey.models.user import User
from flask import current_app
from flask.cli import AppGroup

deploy_cli = AppGroup("deploy")


class DeployException(Exception):
    """Exception during deployment.
    """


def __build():
    init_db()
    current_app.logger.info("Successfully build the database!")


@deploy_cli.command("build")
def build():
    """Initalize db.
    """
    __build()


def __clean():
    if "DEVELOPMENT_MODE" not in current_app.config:
        raise DeployException(
            "Clean command is only supported in development mode!")

    db_path = current_app.config["SQLALCHEMY_DATABASE_URI"].removeprefix(
        "sqlite:///")

    if os.path.exists(db_path):
        os.remove(db_path)
        current_app.logger.info("Successfully removed %s!", db_path)


@deploy_cli.command("clean")
def clean():
    """Delete database.
    """
    __clean()


@deploy_cli.command("rebuild")
def rebuild():
    """Rebuild database.
    """
    __clean()
    __build()


export_cli = AppGroup("export")


@export_cli.command("json")
@click.option("--output", default="results.json")
def export(output):
    """Export the databse to the provided file (Default: results.json).
    """
    current_app.logger.info("Exporting to %s", output)
    with open(output, "w+", encoding="utf-8") as out_f:
        res = []
        for user in User.query.filter_by(finished=True).all():
            try:
                res.append(user.export())
            except:
                pass

        json.dump(res, out_f, indent=2)

    current_app.logger.info("Export finished.")
