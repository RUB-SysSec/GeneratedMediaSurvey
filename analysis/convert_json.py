"""This file contains functions which are used to convert the json format provided by the application.
"""
import argparse
import functools
import json
import pathlib
import re
import uuid
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import toml as tl

from model_utils import standardize

COUNTER = count()

MEDIA_RE = re.compile(r"(audio|image|text)")

QUESTIONS = "../webapp/questions"

LANG_MAP = {
    "english": "USA",
    "german": "Germany",
    "chinese": "China",
}


INGLE_DICT = {
    "Maintaining order in the nation": "Postmaterialist",  # Postmaterialist
    "Giving people more say in important government decisions": "Materialist",  # Materialist
    "Fighting rising prices": "Postmaterialist",  # Postmaterialist
    "Protecting freedom of speech": "Materialist",  # Materialist
    "Aufrechterhaltung von Ruhe und Ordnung in diesem Lande (Deutschland)": "Postmaterialist",
    "Mehr Einfluss der Bürger*innen auf die Entscheidungen der Regierung": "Materialist",
    "Kampf gegen steigende Preise": "Postmaterialist",
    "Schutz des Rechts auf freie Meinungsäußerung": "Materialist",
    "保持国内（本国）的和平与秩序。": "Postmaterialist",
    "公民对于政府的决定更能够施加影响。": "Materialist",
    "抗争价格上涨。": "Postmaterialist",
    "保护言论自由的权力。": "Materialist",
}

INGLE_CODE_DICT = {
    "PostmaterialistPostmaterialist": 0,
    "PostmaterialistMaterialist": 1,
    "MaterialistPostmaterialist": 2,
    "MaterialistMaterialist": 3,
}


AUDIO_DICT = {
    "Mit Kopfhörern": "I used headphones",
    "Mit dem Lautsprecher meines Smartphones": "I used the speaker of my smartphone",
    "Mit dem Lautsprecher meines Computers/Laptops": "I used the speaker of my computer/laptop",
    "Mit einem externen Lautsprecher": "I used an external speaker",
    "Mit keinem der genannten Geräte ": np.nan,
    "以耳机": "I used headphones",
    "以手机扬声器": "I used the speaker of my smartphone",
    "以电脑扬声器": "I used the speaker of my computer/laptop",
    "以外置扬声器": "I used an external speaker",
    "以上述以外的音频设备": np.nan,
}


class ParseError(Exception):
    """Error parsing user entry.
    """

def _extract_likert_questions(toml: Dict) -> List[Tuple[str, str]]:
    """Extract likert questions from questions dict.
    """
    result = []
    if "question_type" in toml:
        if toml["question_type"] == "likert":
            return [(toml["question"], toml["options"])]
        else:
            return []

    for possible_dict in toml.values():
        if isinstance(possible_dict, Dict):
            result.extend(_extract_likert_questions(possible_dict))

    return result


@functools.lru_cache()
def _generate_code_book() -> Dict:
    """Generate a codebook for the different likert questions.
    """
    tomls = [tl.load(path)
             for path in pathlib.Path(QUESTIONS).glob("**/*.toml")]

    code_book = {}

    for toml in tomls:
        tmp = {}
        # extract likert questions
        for question, likert_type in _extract_likert_questions(toml):
            tmp[question] = likert_type

        # build likert codes
        likert_mapping = {}
        for name, scale in toml["likert_scales"].items():
            likert_mapping[name] = dict(
                map(lambda x: (x[1], x[0]), enumerate(scale["options"], 1)))

        # replace with actual code dict
        for question, scale in tmp.items():
            code_book[question] = likert_mapping[scale]

    return code_book


@functools.lru_cache()
def _generate_education_code_book() -> Dict[str, List[str]]:
    """Generate a codebook for the different education questions.
    """
    tomls = [tl.load(path)
             for path in pathlib.Path(QUESTIONS).glob("**/*.toml")]

    code_book: Dict[str, List[str]] = {}

    for toml in tomls:
        edu_questions = toml["demographics"]["demographics"]["questions"]["education"]

        for cat in ["low", "medium", "high"]:
            for quest in edu_questions[cat]:
                code_book[quest] = cat

    return code_book


@functools.lru_cache()
def _generate_question_code_book() -> Dict:
    """Generate a codebook for questions.
    """
    tomls = [tl.load(path)
             for path in pathlib.Path(QUESTIONS).glob("**/*.toml")]

    code_book = {}

    for toml in tomls:
        for category, data in toml.items():
            if category not in ["demographics", "presurvey", "postsurvey"]:
                continue

            for questions in map(lambda x: x["questions"], data.values()):
                code_book.update({
                    val["question"]: i for i, val in enumerate(questions.values(), 1)
                })

    return code_book


def clean_data(path: str):
    """Clean the data by removing every participant who did not finish.
    """
    with open(path, encoding="utf-8") as src:
        data_raw = json.load(src)

    tmp = []
    for data in data_raw:
        if data["finished"]:
            tmp.append(data)

    new_file_path = path[:path.rfind(".")] + "_cleaned.json"
    with open(new_file_path, "w+", encoding="utf-8") as dst:
        json.dump(tmp, dst)


def _parse_answers(answers: Dict) -> Dict:
    """Parses the question array into the different categories
    """
    tmp: Dict[str, List[Dict]] = {}
    for answer in answers:
        category = answer["category"]

        entry: List[Dict] = tmp.get(category, [])
        entry.append(answer)
        tmp[category] = entry

    result = {}
    for key, val in tmp.items():
        val = list(sorted(val, key=lambda x: x["question_id"]))
        result[key] = val

    return result


def _add_timers(timer: Dict, category: str, user_entry: pd.Series):
    """Add specific timers to the row and columns List.
    """
    start_time = timer["start_time"]
    end_time = timer["end_time"]
    total_time = timer["total_time"]

    user_entry = pd.concat(
        (
            user_entry,
            pd.Series([
                start_time,
                end_time,
                total_time,
            ], index=[
                f"{category}_start_time",
                f"{category}_end_time",
                f"{category}_total_time",
            ])
        ), axis=0)


class InvalidDataPointError(Exception):
    """An invalid data point discovered in the presurvey.
    """


def _add_answers(answers: Dict, categories: List, idx: str, questions_seen: Set, df: Optional[pd.DataFrame] = None, ):
    """Add answers based on categories provided.
    We sort by the question_id to ensure the same ordering.
    """
    tmp = []
    likert_code_book = _generate_code_book()
    question_code_book = _generate_question_code_book()
    for category in categories:
        if category not in answers:
            continue

        for question in sorted(answers[category], key=lambda x: int(x["question_id"])):
            question_id = question["question_id"]

            if question_id in questions_seen:
                continue

            questions_seen.add(question_id)

            quest = question["question"]
            if quest in likert_code_book:
                answer = likert_code_book[quest][question["answer"]]
            else:
                answer = question["answer"]

            if quest == "保护言论自由。":
                raise InvalidDataPointError(
                    "Second Inglehardt question was bugged.")
            else:
                i = question_code_book[quest]

            tmp.append((idx, f"{category}_{i}", answer))

    tmp_df = pd.DataFrame(
        tmp, columns=["id", "category", "answer"]
    )

    if df is None:
        return tmp_df

    return pd.concat([df, tmp_df])


def _add_ratings(ratings: Dict, idx: str) -> Tuple[str, pd.DataFrame]:
    """Extract the ratings, the type of the experiment and the time taken for the experiment.
    """
    # add media type
    media_type = None
    for rating in ratings:
        if rating["file_path"]:
            media_type = MEDIA_RE.search(rating["file_path"])[1]
            break

    tmp = []
    for rating in ratings:
        try:
            user_rated = int(rating["rating"])
            real = "fake" if "fake" in rating["file_path"] else "real"

            if user_rated == 0:
                correct = np.NaN
            elif real == "real":
                correct = user_rated > 0
            else:
                correct = user_rated < 0

            tmp.append([
                idx,
                rating["index_of_experiment"],
                rating["start_time"],
                rating["end_time"],
                rating["total_time"],
                rating["file_path"],
                user_rated,
                real,
                correct,
                media_type,
            ])
        except TypeError:
            continue

    rating_df = pd.DataFrame(
        tmp,
        columns=[
            "id",
            "index",
            "start_time",
            "end_time",
            "total_time",
            "file_path",
            "rating",
            "type",
            "correct",
            "media_type",
        ],
    )

    rating_df["start_time"] = pd.to_datetime(rating_df["start_time"])
    rating_df["end_time"] = pd.to_datetime(rating_df["end_time"])
    rating_df["total_time"] = pd.to_timedelta(rating_df["total_time"])

    return media_type, rating_df


def _standardize(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        df[f"{col}_s"] = standardize(df[col])


def _calculate_scores(questions: pd.DataFrame) -> pd.DataFrame:
    """Calculate the scores used for features in the regression.
    """
    quest_pivot = questions.pivot(
        index="id", columns="category", values="answer").reset_index()

    # CRT
    crt = ((quest_pivot["CRT_1"] == 5) | (quest_pivot["CRT_1"] == 0.05)).astype(int) \
        + (quest_pivot["CRT_2"] == 5) \
        + (quest_pivot["CRT_3"] == 47)

    # Inglehart
    inglehart = quest_pivot["Ingelhardt_1"].apply(
        INGLE_DICT.get) + quest_pivot["Ingelhardt_2"].apply(INGLE_DICT.get)
    inglehart = inglehart.apply(INGLE_CODE_DICT.get)

    # familiarity
    familiarity = quest_pivot["deepfakes_1_1"] - 1  # change to 0 indexing

    # AHS
    ahs = quest_pivot[[f"AHS_{i}" for i in range(1, 7)]].sum(axis=1)

    # reverse code 7-9
    ahs += (8 - quest_pivot[[f"AHS_{i}" for i in range(7, 10)]]).sum(axis=1)

    ahs += quest_pivot[[f"AHS_{i}" for i in range(10, 13)]].sum(axis=1)

    # GTS
    gts = quest_pivot[list(
        filter(lambda x: "GTS" in x, quest_pivot.columns))].sum(axis=1)

    # NMLS_FC
    nmls_fc = quest_pivot[list(
        filter(lambda x: "NMLS_FC" in x, quest_pivot.columns))].sum(axis=1)

    # NMLS_CC
    nmls_cc = quest_pivot[list(
        filter(lambda x: "NMLS_CC" in x, quest_pivot.columns))].sum(axis=1)

    # NMLS_FP
    nmls_fp = quest_pivot[list(
        filter(lambda x: "NMLS_FP" in x, quest_pivot.columns))].sum(axis=1)

    # NMLS_CP
    nmls_cp = quest_pivot[list(
        filter(lambda x: "NMLS_CP" in x, quest_pivot.columns))].sum(axis=1)

    # NMLS
    nmls = nmls_fc + nmls_cc + nmls_fp + nmls_cp

    df = pd.DataFrame({
        "id": quest_pivot["id"],
        "CRT": crt,
        "PO": inglehart,
        "FAM": familiarity,
        "AHS": ahs,
        "GTS": gts,
        "NMLS_FC": nmls_fc,
        "NMLS_CC": nmls_cc,
        "NMLS_FP": nmls_fp,
        "NMLS_CP": nmls_cp,
        "NMLS": nmls,
    })

    _standardize(
        df, ["AHS", "GTS", "NMLS_FC", "NMLS_CC", "NMLS_FP", "NMLS_CP", "NMLS"])

    return df


def _parse_entry(data: Dict) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    assert data["finished"]
    # ==========================
    # general
    # ==========================
    # assign unique id for user (to blind the UUID used by the provider)
    idx = next(COUNTER)


    age = data["age"]
    language = data["language"]
    country = LANG_MAP.get(language)
    passed_attention = data["passed_attention"]
    start_time = data["start_time"]
    end_time = data["end_time"]
    total_time = data["total_time"]

    user_entry = pd.Series([
        idx,
        age,
        language,
        country,
        passed_attention,
        start_time,
        end_time,
        total_time,
    ], index=[
        "id",
        "age",
        "language",
        "country",
        "passed_attention",
        "start_time",
        "end_time",
        "total_time",
    ])

    # ==========================
    # questionnaire
    # ==========================

    # extract timers
    timers = data["timers"]
    demographics_timer = timers[0]
    presurvey_timer = timers[1]
    postsurvey_timer = timers[2]

    # extract answers
    answers = _parse_answers(data["answers"])

    question_seen = set()
    # demographics
    _add_timers(demographics_timer, "demographics", user_entry)
    questions_df = _add_answers(answers, ["demographics"], idx, question_seen)

    # presurvey
    _add_timers(presurvey_timer, "presurvey", user_entry)
    questions_df = _add_answers(
        answers, ["deepfakes_1"], idx, question_seen, df=questions_df)
    questions_df = _add_answers(
        answers, ["deepfakes_2"], idx, question_seen, df=questions_df)
    questions_df = _add_answers(
        answers, ["deepfakes_3"], idx, question_seen, df=questions_df)

    # postsurvey
    _add_timers(postsurvey_timer, "postsurvey", user_entry)
    questions_df = _add_answers(answers, ["concern", "CRT", "AHS", "GTS", "NMLS_FC",
                                          "NMLS_CC", "NMLS_FP", "NMLS_CP", "Ingelhardt", "Audio"], idx, question_seen, df=questions_df)

    # add education and median time to user data
    education_code_book = _generate_education_code_book()
    edu_raw = questions_df[questions_df["category"]
                           == "demographics_3"]["answer"]
    education = education_code_book[edu_raw.iloc[0]]

    user_entry = pd.concat((user_entry, pd.Series(
        [education],
        index=["education"],
    )), axis=0)

    # ==========================
    # ratings
    # ==========================
    media, ratings_df = _add_ratings(data["ratings"], idx)

    median_time = (ratings_df["end_time"] - ratings_df["start_time"]).median()
    user_entry = pd.concat((user_entry, pd.Series(
        [media, median_time],
        index=["media_type", "median_time"],
    )), axis=0)

    return user_entry, questions_df, ratings_df


def read_and_parse_data(path: str, device_path: Optional[Union[Path, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and parse the data to DataFrames.
    """
    with open(path, "r", encoding="utf-8") as src:
        data_raw = json.load(src)

    # parse each entry
    users = []
    questions = []
    ratings = []

    for data in data_raw:
        try:
            user_entry, question_entry, rating_entry = _parse_entry(data)
            users.append(user_entry)
            questions.append(question_entry)
            ratings.append(rating_entry)
        except ParseError:
            pass

    user_df = pd.DataFrame(users)
    questions_df = pd.concat(questions).reset_index(drop=True)
    score_df = _calculate_scores(questions_df)

    ratings_df = pd.concat(ratings).reset_index(drop=True)
    _standardize(
        user_df, ["median_time"])

    if device_path is not None:
        device_df = pd.read_csv(device_path).rename(
            columns={"Device": "device"})
        device_df[device_df.isna()] = np.nan

        user_df = user_df.merge(device_df[["id", "device"]], on="id")
        user_df["device_enc"], _ = pd.factorize(
            user_df["device"], sort=True)

    audio_device = questions_df[questions_df["category"]
                                == "Audio_1"][["id", "answer"]]

    audio_device["hearing"] = audio_device["answer"].apply(AUDIO_DICT.get)
    audio_device["hearing_enc"], _ = pd.factorize(
        audio_device["hearing"], sort=True)

    user_df = user_df.merge(
        audio_device[["id", "hearing", "hearing_enc"]], on="id", how="outer")

    return user_df, questions_df, ratings_df, score_df

def main(args: argparse.Namespace):
    user_df, questions_df, ratings_df, score_df = read_and_parse_data(args.FILE)
    root = Path(args.dir)
    if not root.exists():
        root.mkdir(parents=True)

    user_df.to_csv(root.joinpath(Path("user.csv")))
    questions_df.to_csv(root.joinpath(Path("questions.csv")))
    ratings_df.to_csv(root.joinpath(Path("ratings.csv")))
    score_df.to_csv(root.joinpath(Path("scores.csv")))
    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse results and save to csv.")

    parser.add_argument("FILE", type=str, help="JSON file to parse.")
    parser.add_argument("-d", "--dir", type=str, help="Directory for saving CSVs.", default="data")

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())