from typing import List, Tuple

from flask import current_app
from flask import render_template as render_template_flask
from flask import session, url_for

SESS_USER_ID = "user_id"
SESS_ACC_COND = "accepted_conditions"
SESS_STEP = "step"
SESS_MEDIA_CHOOSEN = "global_media"


def current_endpoint() -> str:
    """Get the current endpoint.
    """
    step = session[SESS_STEP]
    if step == -1:
        return "index.consent"

    return current_app.survey_steps[step]


def current_step() -> str:
    """Get the current url.
    """
    return url_for(current_endpoint())[1:]


def next_step() -> str:
    """Get the next url for this session.
    """
    session[SESS_STEP] += 1

    return current_step()


STEPS = {
    "english": {
        "intro": "Intro",
        "demographics": "Demographics",
        "presurvey": "Presurvey",
        "instruction": "Instruction",
        "experiment": "Experiment",
        "postsurvey": "Postsurvey",
        "end": "End",
    },
    "german": {
        "intro": "Einführung",
        "demographics": "Demografische Daten",
        "presurvey": "Vorabbefragung",
        "instruction": "Instruktion",
        "experiment": "Experiment",
        "postsurvey": "Nachbefragung",
        "end": "Ende",
    },
    "chinese": {
        "intro": "简介",
        "demographics": "人口统计",
        "presurvey": "初步调查",
        "instruction": "指令",
        "experiment": "实验",
        "postsurvey": "后续调查",
        "end": "末尾",
    },
}


def __build_step_mapping() -> List[Tuple[int, str]]:
    """Builds a mapping for the steps and corresponding navbar items.
    """
    steps = STEPS[current_app.config["LANGUAGE"]]
    res = [(-1, steps["intro"])]
    for i, step in enumerate(current_app.survey_steps):
        category, endpoint = step.split(".")
        if category == "questionnaire":
            res.append((i, steps[endpoint.removeprefix("questionnaire_")]))
        elif category == "experiments":
            res.append((i, steps["experiment"]))
        elif endpoint == "instruction":
            res.append((i, steps["instruction"]))
        elif endpoint == "final":
            res.append((i, steps["end"]))
        else:
            raise NotImplementedError(
                f"Mapping for endpoints not implemented yet: {step}!")

    return res


def render_template(*args, **kwargs) -> str:
    """Wrapper around render_template to pass application-wide kwargs.
    """
    return render_template_flask(
        *args,
        survey_steps=__build_step_mapping(),
        survey_type=current_app.config["SURVEY_TYPE"],
        current_step=session[SESS_STEP],
        language=current_app.config["LANGUAGE"],
        **kwargs,
    )
