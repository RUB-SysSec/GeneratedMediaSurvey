"""Hold the experiment data."""
from typing import Dict

from dfsurvey.models import db
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql import func


class ExperimentBase(object):
    """Base class for all experiment models.
    """
    id = db.Column(db.Integer, primary_key=True)

    @declared_attr
    def user_id(cls):
        return db.Column(db.Integer, db.ForeignKey("users.id"))

    @declared_attr
    def user(cls):
        return db.relationship(
            "User",
            back_populates="ratings",
            lazy="select",
            uselist=False,
        )

    index_of_experiment = db.Column(db.Integer, nullable=False)
    finished = db.Column(db.Boolean, default=False)

    start_time = db.Column(db.DateTime, server_default=func.now())
    end_time = db.Column(db.DateTime, onupdate=func.now())

    @classmethod
    def create(cls, user_id: int, index_of_experiment: str, **kwargs) -> 'ExperimentBase':
        """Create a new experiment.
        """
        experiment = cls(
            user_id=user_id,
            index_of_experiment=index_of_experiment,
            **kwargs,
        )
        db.session.add(experiment)
        db.session.commit()
        return experiment

    def finish_experiment(self, **kwargs):
        """Method for finishing the experiment and cleaning up the session.
        """
        self.finished = True
        self._finish_experiment(**kwargs)
        db.session.commit()

    def _finish_experiment(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise KeyError("Trying set a non-experiment variable!")
            setattr(self, key, val)

    def export(self) -> Dict:
        """Export this entry to a dict.
        """
        return {
            "id": self.id,
            "index_of_experiment": self.index_of_experiment,
            "finished": self.finished,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "total_time": str(self.end_time - self.start_time) if self.end_time else None,
        }


class Rating(db.Model, ExperimentBase):
    """Realistic ratings for files.
    """
    __tablename__ = "ratings"
    file_path = db.Column(db.Text, nullable=True)
    rating = db.Column(db.Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<Rating for file {self.file_path} by user {self.user_id}: {self.rating}>"

    def export(self) -> Dict:
        """Export this function to a dict.
        """
        res = super().export()
        res.update({
            "file_path": self.file_path,
            "rating": self.rating,
        })

        return res


class Guess(db.Model, ExperimentBase):
    """Guess result between two files.
    """
    __tablename__ = "guesses"
    correct = db.Column(db.Boolean, nullable=True)
    file_path_real = db.Column(db.Text, nullable=True)
    file_path_fake = db.Column(db.Text, nullable=True)

    user = db.relationship(
        "User",
        back_populates="guesses",
        lazy="select",
        uselist=False,
    )

    def export(self) -> Dict:
        """Export this function to a dict.
        """
        res = super().export()
        res.update({
            "correct": self.correct,
            "file_path_real": self.file_path_real,
            "file_path_fake": self.file_path_fake,
        })

        return res

    def __repr__(self) -> str:
        return f"<Guess between real {self.file_path_real} and fake {self.file_path_fake} by user {self.user_id}: {self.correct}>"


class ExperimentCounter(db.Model):
    """Counter for experiments.
    """
    __tablename__ = "experiment_counters"
    id = db.Column(db.Integer, primary_key=True)
    index_of_experiment = db.Column(db.Integer, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def export(self) -> Dict:
        """Export this counter to a dict.
        """
        res = {
            "id": self.id,
            "index_of_experiment": self.index_of_experiment,
            "count": self.count,
        }

        return res

    def __repr__(self) -> str:
        return f"<Counter for Experiment {self.index_of_experiment}: {self.count}>"
