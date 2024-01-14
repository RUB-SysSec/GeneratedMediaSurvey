"""The differnent experiments to run."""
import json
import random
from functools import partial, wraps
from typing import Callable, Tuple

from dfsurvey.models import db
from dfsurvey.models.experiment import (ExperimentBase, ExperimentCounter,
                                        Guess, Rating)
from dfsurvey.models.user import User
from dfsurvey.views import (SESS_MEDIA_CHOOSEN, SESS_USER_ID, next_step,
                            render_template)
from dfsurvey.views.decorators import (conditions_required,
                                       redirect_to_correct_step, user_required)
from flask import Blueprint, Response, abort, current_app, jsonify
from flask import render_template as flask_render_template
from flask import request, session
from sqlalchemy.exc import SQLAlchemyError

experiments_bp = Blueprint(
    "experiments", __name__, template_folder="templates/experiments")

SESS_EXP_ID = "experiment_id"
SESS_NUM_EXP = "experiments_performed"
SESS_MEDIA = "media"

SESS_FILE_REAL = "current_file_real"
SESS_FILE_FAKE = "current_file_fake"
SESS_CORRECT = "correct"

SESS_FILE = "current_file"
SESS_SCALE_ORDER = "scale_order"


def _amount_of_experiments() -> int:
    amount = current_app.experiments_to_perform
    if SESS_MEDIA in session:
        media = session[SESS_MEDIA]
        if media == "image":
            return current_app.image_experiments or amount
        elif media == "audio":
            return current_app.audio_experiments or amount
        elif media == "text":
            return current_app.text_experiments or amount

    return amount


def __start_experiment(f: Callable, cls: ExperimentBase) -> Callable:
    @wraps(f)
    def wrap(*args, **kwargs):
        # initalize experiments
        if SESS_NUM_EXP not in session:
            session[SESS_NUM_EXP] = 0

        # if no experiment running -> start one
        if SESS_EXP_ID not in session:
            user_id = session[SESS_USER_ID]

            session[SESS_NUM_EXP] += 1
            index_of_experiments = session[SESS_NUM_EXP]

            experiment = cls.create(user_id=user_id,
                                    index_of_experiment=index_of_experiments)
            session[SESS_EXP_ID] = experiment.id

        return f(*args, **kwargs)

    return wrap


# experiment specific decorator
start_scale = partial(__start_experiment, cls=Rating)
start_comparison = partial(__start_experiment, cls=Guess)


def __load_file(media_type: str, path: str) -> Tuple[str, str]:
    shortend_path = path[path.find("/data"):]
    current_app.logger.debug("Loading: %s", path)

    if media_type == "text":
        # load content of the file to pass to template
        with open(path, encoding="utf-8") as fake_f:
            content = json.load(fake_f)
    else:
        # otherwise we pass the path to be put into src attr
        content = shortend_path

    return content, shortend_path


def __random_file(media_type: str, language: str, real_or_fake: str) -> Tuple[str, str]:
    data = current_app.experiment_data
    possible_samples = list(data[media_type][language][real_or_fake].keys())
    sample = random.choice(possible_samples)

    path = str(data[media_type][language][real_or_fake][sample])

    return __load_file(media_type, path)


def _render_progress_bar(template: str, **kwargs) -> str:
    # current experiment not completed yet
    progress = session[SESS_NUM_EXP] - 1

    progress_max = _amount_of_experiments()
    progress_percentage = int((progress / progress_max) * 100)

    return render_template(
        template,
        progress=progress_percentage,
        **kwargs
    )


@start_scale
def __render_random_scale(media_type: str) -> str:
    language = current_app.config["LANGUAGE"]
    data = current_app.experiment_data
    amount = _amount_of_experiments()

    fake_data = list(data[media_type][language]["fake"].values())
    real_data = list(data[media_type][language]["real"].values())

    combined = fake_data + real_data

    if SESS_SCALE_ORDER not in session:
        if len(combined) < amount:
            current_app.logger.error(
                "Too few data points to show to the user, will repeat! Datapoints: %s; To show: %s", len(combined), amount)
        elif len(combined) > amount:
            current_app.logger.error(
                "Too many data points, will not show everything! Datapoints: %s; To show: %s", len(combined), amount)

        order = list(range(len(combined)))
        random.shuffle(order)
        current_app.logger.info(
            "Set order for client %s", session[SESS_USER_ID])

        session[SESS_SCALE_ORDER] = tuple(order)

    idx = session[SESS_SCALE_ORDER][(
        session[SESS_NUM_EXP] - 1) % len(combined)]
    sample = combined[idx]

    media, shortend_file_path = __load_file(media_type, str(sample))

    # store actual file path for later
    session[SESS_FILE] = shortend_file_path

    return _render_progress_bar(
        template=f"{media_type}_scale.html",
        media=media,
        options=current_app.likert,
        scale=current_app.config["SURVEY_TYPE"],
    )


@start_comparison
def __render_random_comparison(media_type: str) -> str:
    language = current_app.config["LANGUAGE"]
    fake, fake_shortend_path = __random_file(media_type, language, "fake")
    real, real_shortend_path = __random_file(media_type, language, "real")

    session[SESS_FILE_FAKE] = fake_shortend_path
    session[SESS_FILE_REAL] = real_shortend_path

    if random.random() > .5:
        left, right = real, fake
        session[SESS_CORRECT] = "left"
    else:
        right, left = real, fake
        session[SESS_CORRECT] = "right"

    return _render_progress_bar(
        template=f"{media_type}_comparison.html",
        left=left,
        right=right,
    )


def __handle_post(result_fn: Callable) -> Tuple[Response, int]:
    """This function handles posts to experiment endpoints.

    The actual data gets passed to the result_fn, typically to write the result to the db.
    The result_fn function must also cleanup the session data!
    """
    data = request.get_json()
    if data is None:
        abort(400, "No data provided!")

    result = data["choice"]

    try:
        result_fn(result)

        if "count" in data:
            ctx = ExperimentCounter(
                index_of_experiment=session[SESS_NUM_EXP],
                count=int(data["count"]),
            )
            db.session.add(ctx)
            db.session.commit()

    except KeyError:
        db.session.rollback()
        abort(400, "Data not present!")
    except SQLAlchemyError as err:
        current_app.logger.error(
            "Error creating new rating: %s", err, exc_info=True)
        db.session.rollback()
        abort(400)

    data = {
        "move_on": False,
        "msg": "Saved rating!",
    }

    if session[SESS_NUM_EXP] >= _amount_of_experiments():
        # sufficient experiments performed
        del session[SESS_NUM_EXP]

        data["move_on"] = True
        data["url"] = next_step()

        if SESS_MEDIA in session:
            del session[SESS_MEDIA]
        if SESS_SCALE_ORDER in session:
            del session[SESS_SCALE_ORDER]

    return jsonify(data), 200


def __handle_scale_post() -> Tuple[Response, int]:
    def handle_result(data):
        option = int(data)

        boundry = len(current_app.likert[0]) // 2
        if (current_app.config["SURVEY_TYPE"] == "scale" and (option < 0 or option > 100)) or \
                (current_app.config["SURVEY_TYPE"] == "likert" and (option < -boundry and option > boundry)):
            abort(400, "Data malformed.")

        experiment_id = session[SESS_EXP_ID]
        file_path = session[SESS_FILE]

        rating = Rating.query.get(experiment_id)
        rating.finish_experiment(
            file_path=file_path,
            rating=option,
        )

        del session[SESS_EXP_ID]
        del session[SESS_FILE]

    return __handle_post(handle_result)


def __handle_comparison_post() -> Tuple[Response, int]:
    def handle_result(data):
        option = data

        if option != "left" and option != "right":
            abort(400, "Data malformed.")

        experiment_id = session[SESS_EXP_ID]
        file_path_real = session[SESS_FILE_REAL]
        file_path_fake = session[SESS_FILE_FAKE]

        guess = Guess.query.get(experiment_id)
        guess.finish_experiment(
            correct=session[SESS_CORRECT] == option,
            file_path_real=file_path_real,
            file_path_fake=file_path_fake,
        )

        del session[SESS_EXP_ID]
        del session[SESS_FILE_REAL]
        del session[SESS_FILE_FAKE]
        del session[SESS_CORRECT]

    return __handle_post(handle_result)


def __handle_scale(media_type: str):
    """Handle scale rating
    """
    if request.method == "GET":
        return __render_random_scale(media_type)

    return __handle_scale_post()


def __handle_comparison(media_type: str):
    """Audio comparison.
    """
    if request.method == "GET":
        return __render_random_comparison(media_type)

    return __handle_comparison_post()


@ experiments_bp.route("/random_scale", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def random_scale():
    """Rate some text based on a scale scale.
    """
    if SESS_MEDIA not in session:
        media = current_app.config["MEDIA_OVERWRITE"] or random.choice(
            ["text", "audio", "image"])

        current_app.logger.info("Choose %s for user %s",
                                media, session[SESS_USER_ID])
        session[SESS_MEDIA] = media
        session[SESS_MEDIA_CHOOSEN] = media

        user = User.query.get(session[SESS_USER_ID])
        user.media_assigned = media
        db.session.commit()

    return __handle_scale(session[SESS_MEDIA])


@ experiments_bp.route("/text_scale", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def text_scale():
    """scale based text experiment.
    """
    return __handle_scale("text")


@ experiments_bp.route("/audio_scale", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def audio_scale():
    """Rate an audio based on a scale scale.
    """
    return __handle_scale("audio")


@ experiments_bp.route("/image_scale", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def image_scale():
    """Rate an image based on a scale scale.
    """
    return __handle_scale("image")


@ experiments_bp.route("/random_comparison", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def random_comparison():
    """Rate some text based on a comparison scale.
    """
    if SESS_MEDIA not in session:
        media = random.choice(["text", "audio", "image"])
        session[SESS_MEDIA] = media

    return __handle_comparison(session[SESS_MEDIA])


@ experiments_bp.route("/text_comparison", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def text_comparison():
    """Text comparison.
    """
    return __handle_comparison("text")


@ experiments_bp.route("/audio_comparison", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def audio_comparison():
    """Audio comparison.
    """
    return __handle_comparison("audio")


@ experiments_bp.route("/image_comparison", methods=["GET", "POST"])
@ user_required
@ redirect_to_correct_step
@ conditions_required
def image_comparison():
    """Image comparison.
    """
    return __handle_comparison("image")


@experiments_bp.route("/76c76c5d4de06b96a214d86fe8d06f1dad91e371")
def debrief():
    """Debriefing page.
    """
    language = current_app.config["LANGUAGE"]

    data = current_app.experiment_data
    samples = {}
    for media_type in ["audio", "image", "text"]:
        possible_samples = {}
        possible_samples["real"] = list(
            map(lambda x: __load_file(media_type, str(x))[0], data[media_type][language]["real"].values()))
        possible_samples["fake"] = list(
            map(lambda x: __load_file(media_type, str(x))[0], data[media_type][language]["fake"].values()))
        samples[media_type] = possible_samples

    return flask_render_template(
        "debrief.html",
        samples=samples
    )
