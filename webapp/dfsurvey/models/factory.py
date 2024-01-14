from typing import Dict, Set

from dfsurvey.models import db
from dfsurvey.models.question import Category, Option, Question
from flask import current_app


def _insert_options(data: Dict, cat: str, seen_options: Set):
    for option in data[cat]:
        if option not in seen_options:
            idx = Option.query.filter(Option.text == option).first()
            if not idx:
                db.session.add(Option(text=option))
            seen_options.add(option)


def __add_questions(name: str,  seen_questions: Set, seen_options: Set):
    for category, data in current_app.questions[name].items():
        description = data.get("description")

        cat = Category.query.filter(Category.text == category).first()
        if not cat:
            cat = Category(
                text=category,
                description=description,
            )
            db.session.add(cat)
            db.session.commit()

        for key in data["questions"]:
            question_data = data["questions"][key]
            if question_data == description:
                continue

            question = question_data["question"]

            if question not in seen_questions:
                quest = Question.query.filter(
                    Question.text == question).first()
                if not quest:
                    db.session.add(Question(
                        text=question,
                        category_id=cat.id,
                    ))

                seen_questions.add(question)

            qtype = question_data["question_type"]
            if qtype == "options" or qtype == "age":
                _insert_options(question_data, "options", seen_options)

            elif qtype == "education":
                for sub_category in ["low", "medium", "high"]:
                    _insert_options(question_data, sub_category, seen_options)


def init_db():
    """Initalize the database by storing the questions.
    """
    db.create_all()

    seen_questions = set()
    seen_options = set()

    # add likert scales
    for data in current_app.questions["likert_scales"].values():
        _insert_options(data, "options", seen_options)

    # create questions and options for questions
    __add_questions("demographics", seen_questions, seen_options)
    __add_questions("presurvey", seen_questions, seen_options)
    __add_questions("postsurvey", seen_questions, seen_options)

    db.session.commit()
