"""Utitiliy functions for question handling"""
from functools import lru_cache
from typing import Dict, List, Tuple

from dfsurvey.models.question import Option, Question
from flask import current_app


def _get_options_for_question(question: str, root: Dict) -> List[str]:
    if "question" in root:
        if root["question"] == question:
            if root["question_type"] == "likert":
                return current_app.questions["likert_scales"][root["options"]]["options"]

            if root["question_type"] == "options":
                return root["options"]

            if root["question_type"] == "education":
                return root["low"] + root["medium"] + root["high"]

        return []

    res = []
    for node in root.values():
        if isinstance(node, Dict):
            res += _get_options_for_question(question, root=node)

    return res


@lru_cache()
def build_questions_and_options_mapping() -> Tuple[Dict, Dict, Dict]:
    """Build a mapping from the current questions and options to their ids.
    """
    questions_to_id = {}
    options_to_id = {}
    allowed_option_ids = {}

    seen_options = set()
    # collect all options first, so we can map them
    for option in Option.query.all():
        idx = option.id
        if idx in seen_options:
            current_app.logger.warn(
                "Possible duplicate db entry for option: %s", option.text)
        seen_options.add(idx)

        options_to_id[option.text] = idx

    seen_questions = set()
    for question in Question.query.all():
        idx = question.id
        if idx in seen_questions:
            current_app.logger.warn(
                "Possible duplicate db entry for question: %s", question.text)
        seen_questions.add(idx)

        questions_to_id[question.text] = idx
        options = _get_options_for_question(
            question.text, current_app.questions)

        if len(options) > 0:
            allowed_option_ids[question.id] = list(
                map(lambda x: options_to_id[x], options))

    return questions_to_id, options_to_id, allowed_option_ids
