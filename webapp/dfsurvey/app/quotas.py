import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from dfsurvey.app.questions import build_questions_and_options_mapping
from dfsurvey.models.question import OptionAnswer
from dfsurvey.models.user import User
from flask import current_app
from flask_sqlalchemy import BaseQuery
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import expression
from sqlalchemy.types import DateTime


class now_minus(expression.FunctionElement):
    """An sql function for computing a timestamp x seconds ago.
    """
    type = DateTime()
    name = "last_seconds"
    inherit_cache = True


@compiles(now_minus, "sqlite")
def sqlite_now_minus(element, compiler, **kw):
    """SQLite implementation of now_minus.
    """
    return "DATETIME('now', '-' || %s || ' seconds')" % (
        compiler.process(element.clauses, **kw),
    )


@compiles(now_minus, "postgresql")
def pg_now_minus(element, compiler, **kw):
    """PostgreSQL implementation of now_minus.
    """
    return "NOW() - (%s * INTERVAL '1 SECOND')" % (
        compiler.process(element.clauses, **kw),
    )


@lru_cache()
def quota_to_idx_mapping() -> Dict[str, Tuple[int]]:
    """This function computes a mapping from the different quota options to the corresponding 
    option ids.

    Example:
        {
            "Male": (10, ),
            "Low": (11, 12),
        }
    """
    quota_to_option_id = {}
    _, options_to_id, _ = build_questions_and_options_mapping()

    raw_data = current_app.quotas_raw

    # everything but demographis
    for category in filter(lambda x: x != "general" and x != "demographics", raw_data.keys()):
        category_data = raw_data[category]
        for key in category_data["quota"]:
            key = key[0]
            quota_to_option_id[key] = (options_to_id[key], )

    for key in raw_data["demographics"]["quota"]:
        key = key[0]
        data = current_app.questions["demographics"]["demographics"]["questions"]["education"][key]
        quota_to_option_id[key] = tuple(map(lambda x: options_to_id[x], data))

    return quota_to_option_id


@lru_cache()
def remap_mapping() -> Dict[int, int]:
    """Construct a mapping over all remapped ids.
    """
    _, options_to_id, _ = build_questions_and_options_mapping()
    raw_data = current_app.quotas_raw

    remap = {}

    for category in filter(lambda x: x != "general", raw_data.keys()):
        category_data = raw_data[category]
        if "remap" in category_data:
            for src, dst in category_data["remap"]:
                remap[options_to_id[src]] = options_to_id[dst]

    return remap


@lru_cache()
def create_mapping_from_quota_to_remapped_option_ids() -> Dict[int, List[int]]:
    """Create a mapping from a quota id to option ids which get remapped to that id.
    """
    remap = {}
    for src, dst in remap_mapping().items():
        entry = remap.get(dst, [])
        entry.append(src)
        remap[dst] = entry

    return remap


RANGE_RE = re.compile(r"""([0-9]{2})\-([0-9]{2})""")


@lru_cache()
def map_age_to_option_id(age: int) -> int:
    """This function provides a mapping from a continous age variable to the option_id
    representing the corresponding bucket.
    """
    raw_data = current_app.quotas_raw["age"]["quota"]
    _, options_to_id, _ = build_questions_and_options_mapping()

    # go through quotas
    for age_bin, _ in raw_data[:-1]:
        match = RANGE_RE.search(age_bin)
        assert match is not None

        lower = int(match[1])
        upper = int(match[2])

        if lower <= age and upper >= age:
            return options_to_id[age_bin]

    return options_to_id[raw_data[-1][0]]


def quota_to_amount_mapping(quotas: Optional[Dict] = None) -> Dict[str, int]:
    """Compute a mapping from quotas to the amount needed.
    """
    raw_data = quotas or current_app.quotas_raw
    total_amount = int(raw_data["general"]["n"])
    quota_to_amount = {}

    for category in filter(lambda x: x != "general", raw_data.keys()):
        running_percentage = 0
        for key, percentage in raw_data[category]["quota"]:
            percentage = float(percentage)
            assert percentage < 1.
            running_percentage += percentage

            # at least 1 per category
            quota_to_amount[key] = max(1, int(total_amount * percentage))

        assert running_percentage <= 1.05

    return quota_to_amount


def valid_user_ids() -> BaseQuery:
    """Return a query which selects all user ids which should be considered to be active in the survey.
    """
    timeout = current_app.config["TIMEOUT"]

    return User.query.with_entities(User.id).filter((User.language == current_app.config["LANGUAGE"]) & (User.accepted_conditions == True) & (
        (User.finished == True) | (User.last_seen >= now_minus(timeout)))
    )


def option_to_query_and_amount() -> Dict[int, Tuple[BaseQuery, int]]:
    """Compute a mapping from all demographic questions to a query calculating the current amount and the amount needed.

    Cannot be cached since queries have a session attached.
    """
    # retrieve all possible user ids as a subquery
    possible_user_ids = valid_user_ids().scalar_subquery()

    # compute quotas
    quota_to_amount = quota_to_amount_mapping()

    # compute quota to id mapping
    # this includes multiple keys like education
    # "low" = (11, 12)
    quota_to_ids = quota_to_idx_mapping()

    # create a mapping from a quota id to ids which get remapped to that key
    remapping = create_mapping_from_quota_to_remapped_option_ids()

    # compute remappings + queries
    quota_to_query_and_amount = {}

    for key, amount in quota_to_amount.items():
        ids = quota_to_ids[key]

        # check for remappings
        tmp = []
        for idx in ids:
            if idx in remapping:
                tmp.extend(remapping[idx])

            tmp.append(idx)  # also include original key

        ids = set(tmp)

        # build the query
        query = OptionAnswer.query.with_entities(OptionAnswer.user_id).filter(
            # all possible ids in the set
            OptionAnswer.option_id.in_(ids)
            # all possible allowed users
            & OptionAnswer.user_id.in_(possible_user_ids)
        ).group_by(
            OptionAnswer.user_id
        )

        # now create an entry for every found option
        for idx in ids:
            quota_to_query_and_amount[idx] = (query, amount)

    return quota_to_query_and_amount
