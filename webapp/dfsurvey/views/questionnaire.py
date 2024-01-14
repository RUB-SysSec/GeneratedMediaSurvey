import importlib
import random
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Tuple

from dfsurvey.app.questions import build_questions_and_options_mapping
from dfsurvey.app.quotas import (map_age_to_option_id,
                                 option_to_query_and_amount)
from dfsurvey.models import db
from dfsurvey.models.question import (FloatAnswer, IntegerAnswer, OptionAnswer,
                                      TextAnswer, Timer)
from dfsurvey.models.user import User
from dfsurvey.views import SESS_USER_ID, next_step, render_template
from dfsurvey.views.decorators import (conditions_required,
                                       redirect_to_correct_step, user_required)
from flask import Blueprint, abort, current_app, jsonify, request, session
from sqlalchemy.exc import SQLAlchemyError

SESS_TIMER = "questionnaire_timer"
SESS_CTX = "questionnaire_ctx"
SESS_ORDER = "questionnaire_order"
SESS_ATTENTION_CORRECT_KEY = "correct_attention"

questionnaire_bp = Blueprint(
    "questionnaire", __name__, template_folder="templates/questionnaire")


@lru_cache
def _questionnaire(category: str, key: str) -> Dict:
    questions_to_id, _, _ = build_questions_and_options_mapping()
    questionnaire = current_app.questions[category][key]
    questions = questionnaire["questions"]

    for key in questions:
        data = questions[key]
        data["question_id"] = questions_to_id[data["question"]]

    return questionnaire


def __handle_get(category: str) -> str:
    _, options_to_id, _ = build_questions_and_options_mapping()
    if SESS_TIMER not in session:
        timer = Timer.create(session[SESS_USER_ID])
        session[SESS_TIMER] = timer.id

    if SESS_CTX not in session:
        session[SESS_CTX] = 0

        # retrieve all possible keys
        all_keys = set(current_app.questions[category].keys())

        # filter out conditions
        keys = set()
        for key in all_keys:
            questionnaire = current_app.questions[category][key]

            if "conditional" in questionnaire:
                conditional = questionnaire["conditional"]

                # load correct key for condition
                module = importlib.import_module("dfsurvey.views")
                sess_key = getattr(module, conditional["conditional_on"])

                # skip if condition is not fullfilled
                if sess_key not in session or session[sess_key] != conditional["condition"]:
                    continue

            keys.add(key)

        # remap keys which should stay together
        remapping = {}
        if category in current_app.questions_as_one:
            # consider categories which should be treated as one
            for i, tokens in enumerate(current_app.questions_as_one[category]):
                for token in tokens:
                    if token in keys:
                        keys.remove(token)

                dummy_token = f"remap_{i}"
                remapping[dummy_token] = tokens
                keys.add(dummy_token)

        # randomize order
        order = list(keys)
        random.shuffle(order)
        final_order = []

        # build final order by replacing temp keys
        for token in order:
            if token in remapping:
                final_order.extend(remapping[token])
            else:
                final_order.append(token)

        session[SESS_ORDER] = final_order

    # we assumes python > 3.7 anyway
    keys: Dict[str] = session[SESS_ORDER]
    ctx = session[SESS_CTX]
    key = keys[ctx]
    questionnaire = _questionnaire(category, key)

    # check for attention question
    questions = questionnaire["questions"]
    attention_key = list(questions.keys())[-1]
    likert_scales = current_app.questions["likert_scales"]
    possible_attention_questions = questions[attention_key]

    if possible_attention_questions["question_type"] == "attention":
        # pick a random option
        options = likert_scales[possible_attention_questions["options"]]["options"]
        choice = random.choice(options)

        # store choice for later use
        session[SESS_ATTENTION_CORRECT_KEY] = options_to_id[choice]

        # display option in question
        new_question = possible_attention_questions["question"].replace(
            "{}", choice)

        # copy questionnaire and store
        # this is very slow for big dicts, but we use like 4 questions so w/e
        questionnaire = deepcopy(questionnaire)
        questionnaire["questions"][attention_key]["question"] = new_question

    kwargs = {
        "category": questionnaire,
        "options_to_id": options_to_id,
        "category_name": key,
        "likert_scales": likert_scales,
        "progressbar": False,
    }

    if len(keys) > 1:
        kwargs.update({
            "progressbar": True,
            "progress": int((ctx) / len(keys) * 100),
        })

    return render_template(
        "questionnaire.html",
        **kwargs
    )


def __handle_post(category: str) -> Tuple[Any, int]:
    _, _, allowed_option_ids = build_questions_and_options_mapping()

    # store result
    data = request.get_json()
    if data is None:
        abort(400, "No data provided!")

    user_id = session[SESS_USER_ID]

    try:
        current_app.logger.debug("Saving answers to questionnaire: %s", data)
        for (question_id, res) in data["answers"]:
            question_id = int(question_id)
            question_type = res["question_type"]

            if question_type == "textfield":
                # save text answer
                value = res["value"]
                answer = TextAnswer(
                    user_id=user_id,
                    question_id=question_id,
                    text=value,
                )
            elif question_type == "age":
                try:
                    # save raw value
                    value = int(res["value"])
                    user = User.query.get(user_id)
                    user.age = value
                    db.session.add(user)

                    # save corresponding bin
                    option_id = map_age_to_option_id(value)

                    answer = OptionAnswer(
                        user_id=user_id,
                        question_id=question_id,
                        option_id=option_id
                    )
                except ValueError:
                    current_app.logger.warn(
                        "User %s input non convertable input for a numeric question! Storing as text instead.", user_id)
                    value = res["value"]

                    answer = TextAnswer(
                        user_id=user_id,
                        question_id=question_id,
                        text=value,
                    )
            elif question_type == "number":
                try:
                    value = float(res["value"])
                    answer = FloatAnswer(
                        user_id=user_id,
                        question_id=question_id,
                        scale=value,
                    )
                except ValueError:
                    current_app.logger.warn(
                        "User %s input non convertable input for a numeric question! Storing as text instead.", user_id)
                    value = res["value"]

                    answer = TextAnswer(
                        user_id=user_id,
                        question_id=question_id,
                        text=value,
                    )

            elif question_type == "scale":
                # save scale input
                value = int(res["value"])
                answer = IntegerAnswer(
                    user_id=user_id,
                    question_id=question_id,
                    scale=value,
                )
            elif question_type == "likert" or question_type == "options" or question_type == "education":
                # save choice or likert answer
                option_id = int(res["option_id"])
                if question_id in allowed_option_ids \
                        and option_id not in allowed_option_ids[question_id]:
                    abort(400, "Data malformed!")

                answer = OptionAnswer(
                    user_id=user_id,
                    question_id=question_id,
                    option_id=option_id
                )
            elif question_type == "attention":
                # attention handling is done at the end of the function
                pass
            else:
                current_app.logger.error(
                    "Missing implementation for question_type: %s - %s", question_type, res)

            db.session.add(answer)

        db.session.commit()

    except (ValueError, TypeError, KeyError) as err:
        current_app.logger.error(
            "Error storing questionnaire result: %s", err, exc_info=True)
        db.session.rollback()
        abort(400, "Data malformed!")
    except SQLAlchemyError as err:
        current_app.logger.error(
            "Error storing questionnaire result: %s", err, exc_info=True)
        db.session.rollback()
        abort(400)

    ret_data = {
        "msg": "Saved response!",
    }

    session[SESS_CTX] += 1
    if session[SESS_CTX] == len(session[SESS_ORDER]):
        ret_data["redirect"] = True

        ret_data["url"] = next_step()

        # special quota handling
        if category == "demographics" and not current_app.config["QUOTA_SKIP"]:
            query_and_amount = option_to_query_and_amount()

            # retrieve the correct query and the amount needed for that combination
            ids = []

            for _, val in data["answers"]:
                if "option_id" in val:
                    ids.append(int(val["option_id"]))
                    continue

                # handling for age
                age = int(val["value"])
                if age < 18:
                    ret_data["url"] = f"{current_app.config['EARLY_SCREEN_OUT_URL']}?id={session[SESS_USER_ID]}"
                    session.clear()  # clear user session
                    return jsonify(ret_data), 200

                ids.append(map_age_to_option_id(age))

            # check each id
            for idx in ids:
                query, amount = query_and_amount[idx]
                count = query.count()

                # if we already have enough participants -> redirect
                if count > amount:
                    current_app.logger.info(
                        "Screening out user %s, because id %s is full (%s - %s)", user_id, idx, count, amount)
                    ret_data["url"] = f"{current_app.config['QUOTA_FULL']}?id={session[SESS_USER_ID]}"
                    session.clear()  # clear user session
                    return jsonify(ret_data), 200

        Timer.query.get(session[SESS_TIMER]).finish()

        del session[SESS_TIMER]
        del session[SESS_CTX]

    # attention check present
    if SESS_ATTENTION_CORRECT_KEY in session:
        correct_key = session[SESS_ATTENTION_CORRECT_KEY]
        user_answer = int(data["answers"][-1][1]["option_id"])

        # redirect user to failed attention
        if correct_key != user_answer:
            ret_data["redirect"] = True
            ret_data[
                "url"] = f"{current_app.config['DATA_QUALITY_URL']}?id={User.query.get(user_id).uuid}"
            session.clear()

            return jsonify(ret_data), 200

        # mark as passed + clean up session
        user = User.query.get(user_id)
        user.passed_attention = True
        db.session.commit()

        del session[SESS_ATTENTION_CORRECT_KEY]

    return jsonify(ret_data), 200


@questionnaire_bp.route("/questionnaire_demographics", methods=["GET", "POST"])
@user_required
@redirect_to_correct_step
@conditions_required
def demographics():
    """Build the demographics page with quota handling.
    """
    if request.method == "GET":
        return __handle_get("demographics")

    return __handle_post("demographics")


@questionnaire_bp.route("/questionnaire_presurvey", methods=["GET", "POST"])
@user_required
@redirect_to_correct_step
@conditions_required
def presurvey():
    """Dynamically build the presurvey page.
    """
    if request.method == "GET":
        return __handle_get("presurvey")

    return __handle_post("presurvey")


@questionnaire_bp.route("/questionnaire_postsurvey", methods=["GET", "POST"])
@user_required
@redirect_to_correct_step
@conditions_required
def postsurvey():
    """Dynamically build the postsurvey page.
    """
    if request.method == "GET":
        return __handle_get("postsurvey")

    return __handle_post("postsurvey")
