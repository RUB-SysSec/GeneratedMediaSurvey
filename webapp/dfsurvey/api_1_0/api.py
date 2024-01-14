"""Statistics views.
"""
from typing import Dict, List

import toml
from dfsurvey.app.quotas import quota_to_amount_mapping
from dfsurvey.models.question import Category, Option, OptionAnswer, Question
from dfsurvey.models.user import User
from flask import Blueprint, current_app, jsonify, render_template
from sqlalchemy import distinct, func

api_bp = Blueprint('api', __name__, template_folder="templates/api")


@api_bp.route("/results")
def results():
    """Export the results of the survey as json.
    """
    users = []
    for user in User.query.filter_by(finished=True).all():
        try:
            users.append(user.export())
        except TypeError:
            pass

    return jsonify(users)


def _count_different_quotas(demo_id: int, country: str) -> Dict:
    """Count how many participants have completed per category.

    Also accounts for demographics and remappings.
    """
    # counts per category
    counts = dict(User.query.filter(
        (User.language == country) & (User.finished)
    ).join(
        User.option_answers
    ).join(
        OptionAnswer.question
    ).join(
        OptionAnswer.option
    ).join(
        Question.category
    ).filter(
        Category.id == demo_id
    ).with_entities(
        Option.text, func.count(distinct(User.id))
    ).group_by(
        Option.text
    ).all())

    # get target counts
    quota_path = f"{current_app.config['QUOTA_PATH']}/{country}.toml"
    quota_raw = toml.load(quota_path)

    quota_and_amount = quota_to_amount_mapping(quota_raw)

    # create a mapping to account for remapped answers
    remapped: Dict[str, List[str]] = {}
    for data in quota_raw.values():
        if "remap" in data:
            for src, dst in data["remap"]:
                entry = remapped.get(dst, [])
                entry.append(src)
                remapped[dst] = entry

    # creating a mapping from demographics to questions
    questions_path = f"{current_app.config['QUESTION_PATH']}/{country}.toml"
    questions_raw = toml.load(questions_path)
    questions = questions_raw["demographics"]["demographics"]["questions"]["education"]
    category_to_questions = {}
    for key in ["low", "medium", "high"]:
        category_to_questions[key] = questions[key]

    # add remapped results back in by
    amount_per_category = {}
    for quota, needed in quota_and_amount.items():
        # check for demographics
        keys = category_to_questions.get(quota) or [quota]

        # extend with remapping
        keys.extend(remapped.get(quota, []))

        # count
        total = 0

        for key in keys:
            total += counts.get(key, 0)

        amount_per_category[quota] = (total, needed)

    return amount_per_category


@api_bp.route("/stats")
def stats():
    """Display the statistics page.
    """
    countries = User.query.with_entities(User.language).distinct().all()
    country_summary = {}

    demographics = Category.query.filter(
        Category.text == "demographics").all()
    assert len(demographics) == 1

    demo_id = demographics[0].id

    for row in countries:
        country = row["language"]
        title = country.title()

        # count users
        total = User.query.filter(User.language == country).count()
        finished = User.query.filter(
            (User.language == country)
            & (User.finished)).count()

        country_summary[title] = {
            "summary": {
                "total": total,
                "finished": finished,
            }
        }

        country_summary[title]["categories"] = _count_different_quotas(
            demo_id, country)

    return render_template(
        "statistics.html",
        country_summary=country_summary,
    )


@api_bp.route("/stats_external")
def stats_external():
    """Display the statistics page.
    """
    countries = User.query.with_entities(User.language).distinct().all()
    country_summary = {}

    demographics = Category.query.filter(
        Category.text == "demographics").all()
    assert len(demographics) == 1

    demo_id = demographics[0].id

    for row in countries:
        country = row["language"]
        title = country.title()

        # count users
        total = User.query.filter(User.language == country).count()
        finished = User.query.filter(
            (User.language == country)
            & (User.finished)).count()

        country_summary[title] = {
            "summary": {
                "total": total,
                "finished": finished,
            }
        }

        country_summary[title]["categories"] = _count_different_quotas(
            demo_id, country)

    return render_template(
        "statistics_external.html",
        country_summary=country_summary,
    )
