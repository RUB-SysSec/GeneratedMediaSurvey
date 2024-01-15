from statistics import median
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

AGE_LABELS = [
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85-89",
    "90-94",
    "95-99",
]

EDU_LABELS = ["Low", "Medium", "High"]

COUNTRIES = ["USA", "Germany", "China"]
MEDIA_TYPES = ["audio", "image", "text"]


def _compute_row(data: pd.DataFrame, x: str, y: str, index: str, corr_func: Callable = pearsonr, label: str = "Pearson's r") -> pd.DataFrame:
    """Compute correlations for a single row.
    """
    columns = pd.MultiIndex.from_product(
        [["Overall", *COUNTRIES], [label, "p-value"]])

    row = []
    # compute all
    stat, p_value = corr_func(data[x], data[y])
    row.extend((stat, p_value))

    for country in COUNTRIES:
        dat = data[data.country == country]
        stat, p_value = corr_func(dat[x], dat[y])
        row.extend((stat, p_value))

    return pd.DataFrame(np.asarray(row).reshape(1, 8), columns=columns, index=[index])


def compute_correlations(data, x: str, y: str, corr_func: Callable = pearsonr, corr_label: str = "Pearson's r"):
    """Compute correlations for the data.
    """
    overall = _compute_row(data, x, y, "Overall",
                           corr_func=corr_func, label=corr_label)
    audio = _compute_row(data[data.media_type == "audio"],
                         x, y, "Audio", corr_func=corr_func, label=corr_label)
    image = _compute_row(data[data.media_type == "image"],
                         x, y, "Image", corr_func=corr_func, label=corr_label)
    text = _compute_row(data[data.media_type == "text"],
                        x, y, "Text", corr_func=corr_func, label=corr_label)

    return pd.concat(
        [overall, audio, image, text]
    )


def _compute_allowed_indexes(user_df: pd.DataFrame, filter_fn: Callable) -> List[int]:
    """Compute a list of allowed indexes, given the function.
    """
    all_times = pd.to_timedelta(user_df.total_time)
    ids = []
    for country in COUNTRIES:
        for media_type in MEDIA_TYPES:
            times_per_country = all_times[(user_df.country == country) & (
                user_df.media_type == media_type)]
            times_per_country = times_per_country.sort_values()

            # compute threshold and retain only valid indexes
            threshold = filter_fn(times_per_country)
            ids.extend(
                times_per_country.iloc[threshold:-threshold].index.to_list())

    return ids


def filter_ratings(
    ratings_df: pd.DataFrame,
    user_df: pd.DataFrame,
    drop_zero: bool = True,
    filter_type: str = "interval",
    filter_obvious: bool = True,
) -> pd.DataFrame:
    """Filter ratings by the provided type + drop 0 ratings on default. 
    Additionally filter people who always pick the same value.
    """
    # Compute all users in the 95% interval of finishing times
    ratings_filtered = ratings_df.copy()

    if filter_type == "none":
        pass
    elif filter_type == "interval":
        # discard everything but .95 interval in each country
        ids = _compute_allowed_indexes(
            user_df, lambda x: int((.025 * len(x)) + .5))
        allowed_users = user_df.iloc[ids]

        print(
            f"Filtering: {len(user_df) - len(allowed_users)} user using the 95% interval!")
        ratings_filtered = ratings_filtered[ratings_filtered.id.isin(
            allowed_users.id)]

    elif filter_type == "half-median":
        # for each country discard every user which is slower than half the median time
        def _median_filter(data: pd.Series) -> int:
            median_time = data.median()
            return data.reset_index(drop=True).gt(median_time / 2).idxmax()

        ids = _compute_allowed_indexes(user_df, _median_filter)
        allowed_users = user_df.iloc[ids]

        print(
            f"Filtering: {len(user_df) - len(allowed_users)} user using median time!")
        ratings_filtered = ratings_filtered[ratings_filtered.id.isin(
            allowed_users.id)]
    else:
        raise ValueError(f"Provided filter type not known: {filter_type}")

    if filter_obvious:
        for media_type in MEDIA_TYPES:
            ratings_per_media = ratings_filtered[ratings_filtered.media_type == media_type]
            amount_of_option_picked_per_user = ratings_per_media.groupby("id")[
                "rating"].value_counts().unstack().fillna(0)

            amount_of_option_picked_per_user["max"] = amount_of_option_picked_per_user.max(
                numeric_only=True, axis=1)

            if media_type == "image":
                banned = amount_of_option_picked_per_user[amount_of_option_picked_per_user["max"] == 32]
            else:
                banned = amount_of_option_picked_per_user[amount_of_option_picked_per_user["max"] == 30]

            ratings_filtered = ratings_filtered[~ratings_filtered.id.isin(
                banned.index)]
            print(f"Filtered another {len(banned)} users!")

    # filter zero ratings
    if drop_zero:
        ratings_filtered = ratings_filtered[ratings_filtered.rating != 0]

    ratings_filtered.rating = ratings_filtered.rating.astype(int)

    return ratings_filtered


def encode_lang_and_age(data_all: pd.DataFrame) -> pd.DataFrame:
    """Encode the education and age of the given dataframe.
    """
    # encode language
    lang_code = {
        "english": 0,
        "german": 1,
        "chinese": 2,
    }

    data_all["country_enc"] = data_all["language"].apply(lang_code.get)

    # encode education
    edu_code = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }
    data_all["edu_enc"] = data_all["education"].apply(edu_code.get)

    # encode age
    data_all["age_bin"] = pd.cut(data_all.age, np.arange(
        14, 101, 5), labels=AGE_LABELS)

    data_all["age_enc"], _ = pd.factorize(
        data_all["age_bin"], sort=True)

    # tidy up
    data_all = data_all.copy()
    data_all.PO = data_all.PO.astype(int)
    data_all.FAM = data_all.FAM.astype(int)

    return data_all.reset_index(drop=True)


def calculate_acc_per_user(ratings_df: pd.DataFrame, user_df: pd.DataFrame, filter_few: bool = False, **kwargs) -> pd.DataFrame:
    """Calculate a dataframe with the accuracy for each user and additional information.
    """
    ratings_filtered = filter_ratings(
        ratings_df=ratings_df, user_df=user_df, **kwargs)

    # aggregate by user
    acc_per_user = ratings_filtered.groupby("id")["correct"].agg(["count", "sum", "mean"]).rename(columns={
        "count": "N_trials",
        "sum": "N_correct",
        "mean": "Acc",
    })

    acc_per_user = acc_per_user.merge(user_df, on="id")

    # calc real/fake individually
    acc_fake_per_user = ratings_filtered[(ratings_filtered.type == "fake") & (ratings_filtered.id.isin(acc_per_user.id))].groupby("id")["correct"].agg(["count", "sum", "mean"]).rename(columns={
        "count": "N_trials_fake",
        "sum": "N_correct_fake",
        "mean": "Acc_fake",
    })

    acc_real_per_user = ratings_filtered[(ratings_filtered.type == "real") & (ratings_filtered.id.isin(acc_per_user.id))].groupby("id")["correct"].agg(["count", "sum", "mean"]).rename(columns={
        "count": "N_trials_real",
        "sum": "N_correct_real",
        "mean": "Acc_real",
    })

    acc_per_user = acc_per_user.merge(
        acc_fake_per_user, on="id").merge(acc_real_per_user, on="id")

    if filter_few:
        # filter user with few ratings
        before = len(acc_per_user)
        acc_per_user = acc_per_user[(acc_per_user["N_trials_real"] > 5) & (
            acc_per_user["N_trials_fake"] > 5)].reset_index()
        print(
            f"Filtered {before - len(acc_per_user)} user with too few ratings!")

    return acc_per_user
