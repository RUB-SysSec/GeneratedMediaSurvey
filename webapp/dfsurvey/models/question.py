"""Models for questionaire."""
from typing import Any, Dict, Optional

from dfsurvey.models import db
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql import func


class Category(db.Model):
    """The category of a question.
    """
    __tablename__ = "categories"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<Question {self.id} {self.text}>"


class Question(db.Model):
    """The table for storing different questions asked in the questionaire.
    """
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    category_id = db.Column(db.ForeignKey("categories.id"), nullable=False)
    text = db.Column(db.Text, nullable=False)

    category = db.relationship(
        "Category",
        lazy="select",
        uselist=False,
    )

    def __repr__(self):
        return f"<Question {self.id} {self.text}>"


class Option(db.Model):
    """An option for an question.
    """
    __tablename__ = "options"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Option for question: {self.question_id} {self.text}>"


class AnswerBase:
    """Base class for answers.
    """
    id = db.Column(db.Integer, primary_key=True, nullable=False)

    @declared_attr
    def user_id(cls):
        return db.Column(db.Integer, db.ForeignKey("users.id"), index=True, nullable=False)

    @declared_attr
    def question_id(cls):
        return db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)

    @declared_attr
    def question(cls):
        return db.relationship(
            "Question",
            lazy="select",
            uselist=False,
        )

    @property
    def answer(self) -> Any:
        """The answer to the question.
        """
        raise NotImplementedError("Must be implemented by mixin.")

    def __repr__(self):
        return f"<User {self.user_id} picked '{self.answer}' for question '{self.question.text}'>"

    def export(self) -> Dict:
        """Export the entry as json.
        """
        return {
            "user_id": self.user_id,
            "question_id": self.question.id,
            "question": self.question.text,
            "category": self.question.category.text,
            "answer": self.answer,
        }


class OptionAnswer(db.Model, AnswerBase):
    """Option based answer.
    """
    __tablename__ = "question_answer_options"
    option_id = db.Column(db.ForeignKey("options.id"),
                          index=True, nullable=False)

    option = db.relationship(
        "Option",
        lazy="select",
        uselist=False,
    )

    @property
    def answer(self) -> Any:
        """The answer to the question.
        """
        return self.option.text

    def export(self) -> Dict:
        res = super().export()
        res.update({
            "option_id": self.option_id,
        })

        return res


class IntegerAnswer(db.Model, AnswerBase):
    """Rating based answer.
    """
    __tablename__ = "question_answers_integer"
    scale = db.Column(db.Integer, nullable=False)

    @property
    def answer(self) -> Any:
        """The answer to the question.
        """
        return self.scale


class FloatAnswer(db.Model, AnswerBase):
    """Rating based answer.
    """
    __tablename__ = "question_answers_float"
    scale = db.Column(db.Float, nullable=False)

    @property
    def answer(self) -> Any:
        """The answer to the question.
        """
        return self.scale


class TextAnswer(db.Model, AnswerBase):
    """Text based answer.
    """
    __tablename__ = "question_answers_text"
    text = db.Column(db.Text, nullable=False)

    @property
    def answer(self) -> Any:
        """The answer to the question.
        """
        return self.text


class Timer(db.Model):
    """A timer to track time spent on the questionnaire.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    finished = db.Column(db.Boolean, default=False)

    start_time = db.Column(db.DateTime, server_default=func.now())
    end_time = db.Column(db.DateTime, onupdate=func.now())

    @ staticmethod
    def create(user_id: int) -> 'Timer':
        """Create a timer for the given user or return the one thats already in the session.
        """
        timer = Timer(user_id=user_id)
        db.session.add(timer)
        db.session.commit()
        return timer

    def finish(self):
        """Mark the timer as finished.
        """
        self.finished = True
        db.session.commit()

    def export(self) -> Dict:
        """Export this entry to a dict.
        """
        return {
            "user_id": self.user_id,
            "finished": self.finished,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "total_time": str(self.end_time - self.start_time) if self.start_time and self.end_time else None,
        }


def all_answers(user_id: Optional[int] = None) -> AnswerBase:
    """Returns all answers in the databse.
    """
    res = []
    for query in [OptionAnswer.query, IntegerAnswer.query, TextAnswer.query, FloatAnswer.query]:
        if user_id:
            query = query.filter_by(user_id=user_id)

        res.extend(query.all())

    return res
