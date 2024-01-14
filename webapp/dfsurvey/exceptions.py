class DfSurveyException(Exception):
    """Base class for all exceptions.
    """


class NoLikertScaleFound(DfSurveyException):
    """Likert scale not found during startup.
    """


class UserAlreadyExists(DfSurveyException):
    """User id already in db.
    """
