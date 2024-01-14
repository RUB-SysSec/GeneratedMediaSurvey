"""Initial views."""
from dfsurvey.app.quotas import valid_user_ids
from dfsurvey.models.user import User
from dfsurvey.views import (SESS_ACC_COND, SESS_USER_ID, next_step,
                            render_template)
from dfsurvey.views.decorators import (conditions_required,
                                       redirect_to_correct_step, user_required)
from flask import Blueprint, abort, current_app, jsonify, redirect
from flask import render_template as flask_render_template
from flask import request, session, url_for

index_bp = Blueprint("index", __name__, template_folder="templates/index")
reset_bp = Blueprint("reset", __name__, template_folder="templates/index")


@index_bp.route("/")
@index_bp.route("/index")
def index():
    """Generic handler.
    """
    return redirect(url_for('index.consent'))


@index_bp.route("/consent", methods=["GET", "PUT"])
@user_required
@redirect_to_correct_step
def consent():
    """The consent page for accepting conditions.
    If the user accepts the conditions (PUT) modify db + session.
    """
    if request.method == "GET":
        return render_template("consent.html", timeout=current_app.config["TIMEOUT"])

    n_users_finished_and_in_survey = valid_user_ids().count()

    n_users_finished = User.query.with_entities(User.id).filter((User.language == current_app.config["LANGUAGE"]) & (
        (User.finished == True))).count()
    n_users_allowed = int(current_app.quotas_raw["general"]["n"])
    n_users_allowed_with_margin = n_users_allowed + \
        int(n_users_allowed * current_app.config["QUOTA_MARGIN"])
    user = User.query.get(session[SESS_USER_ID])

    if request.json["decision"] != "accept":
        current_app.logger.info(
            "User %s did not accept the conditions!", user.id)
        data = {
            "msg": "User rejected conditions",
            "url": f"{current_app.config['EARLY_SCREEN_OUT_URL']}?id={session[SESS_USER_ID]}"
        }

        session.clear()

    # allow for a safety margin here to account for people in the survey who abandon it
    elif (n_users_finished >= n_users_allowed) or (n_users_finished_and_in_survey >= n_users_allowed_with_margin):
        current_app.logger.info(
            "Rejecting user %s! User finished: %s; Users in survey: %s; Users allowed: %s", user.id, n_users_finished, n_users_finished_and_in_survey, n_users_allowed)
        data = {
            "msg": "Collected enough users.",
            "url": f"{current_app.config['QUOTA_FULL']}?id={session[SESS_USER_ID]}",
        }
        session.clear()

    else:
        user.accept_conditions()
        session[SESS_ACC_COND] = True

        data = {
            "msg": f"Accepted conditions for user: {user.id}",
            "url": next_step(),
        }

    return jsonify(data), 200


@index_bp.route("/instruction", methods=["GET", "PUT"])
@user_required
@redirect_to_correct_step
@conditions_required
def instruction():
    """The instruction page of the survey.
    """
    if request.method == "GET":
        return render_template(
            "instruction.html",
            scale=current_app.config["SURVEY_TYPE"],
            options=current_app.likert,
        )

    data = {
        "msg": f"Accepted conditions for user: {session[SESS_USER_ID]}",
        "url": next_step(),
    }

    return jsonify(data), 200


@index_bp.route("/final")
@user_required
@redirect_to_correct_step
@conditions_required
def final():
    """The final page of the survey.
    """
    user = User.query.get(session[SESS_USER_ID])
    user.finish()
    finish_url = f"{current_app.config['FINISHED_URL']}{'&' if '?' in current_app.config['FINISHED_URL'] else '?'}id={user.uuid}"
    return redirect(finish_url, code=302)


@reset_bp.route("/reset")
def reset():
    """A testpage for resetting users.
    """
    session.clear()
    if "reason" not in request.args:
        abort(400)

    return flask_render_template("reset.html", reason=request.args["reason"])
