"""User model."""
import uuid
from typing import Dict, Optional

from dfsurvey.models import db
from dfsurvey.models.question import OptionAnswer
from flask import current_app
from sqlalchemy.sql import func


class User(db.Model):
    """User model.

    Attr:
        id (int) - Running id.
        uuid (str) - UUID of the user stored as string.
        accepted_conditions (bool) - Has the user already accepted the terms and conditions?
    """
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False)
    accepted_conditions = db.Column(db.Boolean, nullable=False, default=False)
    language = db.Column(db.String(16), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    media_assigned = db.Column(db.String(16), nullable=True)

    finished = db.Column(db.Boolean, index=True, default=False)
    passed_attention = db.Column(db.Boolean, default=False)

    start_time = db.Column(db.DateTime, server_default=func.now())
    last_seen = db.Column(
        db.DateTime, server_default=func.now(), onupdate=func.now(), index=True)
    end_time = db.Column(db.DateTime, nullable=True)

    # questionnaire
    option_answers = db.relationship(
        "OptionAnswer",
        lazy="select",
    )

    integer_answers = db.relationship(
        "IntegerAnswer",
        lazy="select",
    )

    float_answers = db.relationship(
        "FloatAnswer",
        lazy="select",
    )

    text_answers = db.relationship(
        "TextAnswer",
        lazy="select",
    )

    questionnaire_timer = db.relationship(
        "Timer",
        lazy="select",
    )

    # experiments
    ratings = db.relationship(
        "Rating",
        lazy="select",
        back_populates="user",
    )
    guesses = db.relationship(
        "Guess",
        lazy="select",
        back_populates="user",
    )

    @staticmethod
    def create_new_user(user_uuid: Optional[str] = None) -> 'User':
        """Create a new user.
        """
        if not user_uuid:
            user_uuid = str(uuid.uuid4())

        language = current_app.config["LANGUAGE"]
        user = User(uuid=str(user_uuid), language=language)
        db.session.add(user)
        db.session.commit()

        return user

    def seen_again(self):
        """Update seen value.
        """
        self.last_seen = func.now()
        db.session.commit()

    def accept_conditions(self):
        """The user accepts the conditions.
        """
        self.accepted_conditions = True
        db.session.commit()

    def finish(self) -> str:
        """User finished the survey.
        """
        if not self.finished:
            self.finished = True
            self.end_time = func.now()
            db.session.commit()

        return self.uuid

    def add_option_answer(self, question_id: int, option_id: int):
        """Add an answer to questions.
        """
        answer = OptionAnswer(
            user_id=self.id,
            question_id=question_id,
            option_id=option_id,
        )

        self.option_answers.append(answer)

    def export(self) -> Dict:
        """Export this specific user.
        """
        res = {
            "id": self.id,
            "uuid": self.uuid,
            "age": self.age,
            "media_assigned": self.media_assigned,
            "accepted_conditions": self.accepted_conditions,
            "language": self.language,
            "finished": self.finished,
            "passed_attention": self.passed_attention,
            "start_time": str(self.start_time),
            "last_seen": str(self.last_seen),
            "end_time": str(self.end_time),
            "total_time": str(self.end_time - self.start_time) if self.start_time and self.end_time else None,
        }

        res["timers"] = [timer.export() for timer in self.questionnaire_timer]
        res["answers"] = []
        res["answers"].extend([answer.export()
                              for answer in self.option_answers])
        res["answers"].extend([answer.export()
                              for answer in self.integer_answers])
        res["answers"].extend([answer.export()
                              for answer in self.float_answers])
        res["answers"].extend([answer.export()
                              for answer in self.text_answers])

        res["ratings"] = [rating.export() for rating in self.ratings]
        res["guesses"] = [guess.export() for guess in self.guesses]

        return res

    def __repr__(self):
        return f"<User {self.id} {self.uuid}>"
