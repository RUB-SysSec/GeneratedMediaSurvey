from functools import wraps
from typing import Optional

from dfsurvey.exceptions import UserAlreadyExists
from dfsurvey.models.user import User
from dfsurvey.views import (SESS_ACC_COND, SESS_STEP, SESS_USER_ID,
                            current_endpoint)
from flask import abort, current_app, redirect, request, session, url_for


def __create_new_user(uuid: Optional[str] = None):
    if uuid:
        if User.query.filter(User.uuid == uuid).count() > 0:
            raise UserAlreadyExists()

    user = User.create_new_user(user_uuid=uuid)
    session[SESS_USER_ID] = user.id
    session[SESS_ACC_COND] = user.accepted_conditions
    current_app.logger.info("Created User %s", user)

    session[SESS_STEP] = -1


def user_required(f):
    """If no User exists, create one, otherwise update the last seen field.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if SESS_USER_ID not in session:
            uuid = request.args.get("id")
            try:
                __create_new_user(uuid=uuid)
            except UserAlreadyExists:
                current_app.logger.info("User %s already exists!", uuid)
                return redirect(current_app.config["EARLY_SCREEN_OUT_URL"])

        elif session[SESS_ACC_COND]:
            # Only after the user has offically started, we start checking
            user = User.query.get(session[SESS_USER_ID])
            user.seen_again()

        return f(*args, **kwargs)
    return decorated_function


def conditions_required(f):
    """The user has not accepted the conditions yet.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session[SESS_ACC_COND]:
            current_app.logger.debug(
                "Redirected User %s", session[SESS_USER_ID])
            return redirect(url_for('index.consent'))
        return f(*args, **kwargs)
    return decorated_function


def redirect_to_correct_step(f):
    """Redirect the user to the correct endpoint.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_app.config["DISABLE_REDIRECT"]:
            endpoint = request.endpoint
            cur_endpoint = current_endpoint()

            if endpoint is not None and endpoint != cur_endpoint:
                method = request.method

                if method == "GET":
                    current_app.logger.debug(
                        "Redirected User %s to endpoint %s", session[SESS_USER_ID], cur_endpoint)
                    return redirect(url_for(cur_endpoint)[1:])

                return abort(403)

        return f(*args, **kwargs)
    return decorated_function
