from pathlib import Path
from typing import Iterable, List, Tuple, Union

import aesara.tensor as at
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit as logistic

from utils import calculate_acc_per_user, encode_lang_and_age, filter_ratings

TRACE_DIR = "traces/"


def standardize(series: pd.Series) -> pd.Series:
    """Standardize a series by subtracting the mean and scaling by standard deviation.
    """
    return (series - series.mean()) / series.std()


def simulate_user(
    n_user: int = 33,
    inglehart_size: int = 4,
    fam_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate user data with known ground truth
    """
    rng = np.random.default_rng(seed=seed)
    inglehart_index = list(range(inglehart_size))
    fam_index = list(range(fam_size))

    df_us = pd.DataFrame({
        "country": np.zeros(n_user, dtype=int),
        # https://reader.elsevier.com/reader/sd/pii/S0191886921007017?token=FD860DCCBC6590617BE9F420C7E19332DD3CD996BBBBA6AD822243492B3106EBE2DAE77B382259A857965B7A22E3498A&originRegion=eu-west-1&originCreation=20220509084239
        "AHS": rng.normal(loc=4.7, scale=1.4, size=n_user),
        # https://link.springer.com/content/pdf/10.1007/s12144-019-00435-2.pdf
        "GTS": rng.normal(loc=4.10, scale=1.14, size=n_user),
        # https://www.sciencedirect.com/science/article/pii/S0747563216304630
        "NMLS_FC": rng.normal(loc=25.85, scale=4.93, size=n_user),
        # https://www.sciencedirect.com/science/article/pii/S0747563216304630
        "NMLS_CC": rng.normal(loc=43.11, scale=6.51, size=n_user),
        # https://www.sciencedirect.com/science/article/pii/S0747563216304630
        "NMLS_FP": rng.normal(loc=26.72, scale=5.73, size=n_user),
        # https://www.sciencedirect.com/science/article/pii/S0747563216304630
        "NMLS_CP": rng.normal(loc=32.22, scale=8.97, size=n_user),
        # US skew's towards extreme
        "PO": rng.choice(inglehart_index, size=n_user, p=[0.4, 0.1, 0.1, 0.4]),
        # assume most people do not know
        "FAM": rng.choice(fam_index, size=n_user, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
    })

    df_zh = pd.DataFrame({
        "country": np.ones(n_user, dtype=int),
        # https://www.researchgate.net/profile/Helen-Boucher-2/publication/51158317_Language_and_the_Bicultural_Dialectical_Self/links/541586880cf2788c4b35b367/Language-and-the-Bicultural-Dialectical-Self.pdf
        "AHS": rng.normal(loc=5.55, scale=0.52, size=n_user),
        # https://journals.sagepub.com/doi/pdf/10.1177/1948550613514456
        "GTS": rng.normal(loc=4.68, scale=.87, size=n_user),
        "NMLS_FC": rng.normal(loc=22.85, scale=2.93, size=n_user),
        "NMLS_CC": rng.normal(loc=40.11, scale=6.51, size=n_user),
        "NMLS_FP": rng.normal(loc=21.72, scale=8.73, size=n_user),
        "NMLS_CP": rng.normal(loc=27.22, scale=6.97, size=n_user),
        # CN skews towards Materialism
        "PO": rng.choice(inglehart_index, size=n_user, p=[0.3, 0.35, 0.25, 0.1]),
        # assume most people do not know
        "FAM": rng.choice(fam_index, size=n_user, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
    })

    df_de = pd.DataFrame({
        "country": np.ones(n_user, dtype=int) * 2,
        # https://onlinelibrary.wiley.com/doi/pdf/10.1002/sdr.1702
        "AHS": rng.normal(loc=4.94, scale=0.42, size=n_user),
        "GTS": rng.normal(loc=4.31, scale=.51, size=n_user),  # Made-up
        "NMLS_FC": rng.normal(loc=26.85, scale=8.93, size=n_user),
        "NMLS_CC": rng.normal(loc=35.11, scale=6.51, size=n_user),
        "NMLS_FP": rng.normal(loc=17.72, scale=8.73, size=n_user),
        "NMLS_CP": rng.normal(loc=30.22, scale=6.97, size=n_user),
        # DE skews towards postmaterialism
        "PO": rng.choice(inglehart_index, size=n_user, p=[0.2, 0.25, 0.35, 0.2]),
        # assume most people do not know
        "FAM": rng.choice(fam_index, size=n_user, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
    })

    data = pd.concat((df_us, df_zh, df_de), ignore_index=True)
    for cat in ["AHS", "GTS", "NMLS_FC", "NMLS_CC", "NMLS_FP", "NMLS_CP"]:
        data[f"{cat}_s"] = standardize(data[cat])

    # sample CRT score based on AHS
    data["CRT"] = rng.binomial(3, p=logistic(data["AHS_s"]))

    alpha_c = [-.2, .1, .2]  # skew DE/China towards higher detection rate
    beta_AHS_c = [-0.05, 0.1, -0.1]  # assume lower slope for US/DE
    beta_GTS_c = [0.1, 0.05, -0.2]

    beta_NMLS_FC_c = [0.1, 0.5, -0.2]
    beta_NMLS_CC_c = [-0.1, 0.05, 0.2]
    beta_NMLS_FP_c = [0.1, 0.5, -0.2]
    beta_NMLS_CP_c = [-0.1, 0.05, 0.2]

    inglehart_c = [-0.4, -0.1, 0.1, 0.4]

    beta_crt_c = [-0.3, 0.4, 0.1]
    beta_fam_c = [-0.3, 0.4, 0.1]

    crt_delta = [0, 0.3, 0.4, 1.]
    fam_delta = [.2, 0.4, 0.5, 1., 1.]

    # for each user simulate correct picks from random possible trials (we drop 0 trials)
    p_log = np.asarray([alpha_c[x]
                        + ((0.3 + beta_AHS_c[x]) * data["AHS_s"].iloc[i])
                        + ((-0.5 + beta_GTS_c[x]) * data["GTS_s"].iloc[i])
                        + ((-0.3 + beta_NMLS_FC_c[x])
                           * data["NMLS_FC_s"].iloc[i])
                        + ((-.5 + beta_NMLS_CC_c[x])
                           * data["NMLS_CC_s"].iloc[i])
                        + ((1 + beta_NMLS_FP_c[x])
                           * data["NMLS_FP_s"].iloc[i])
                        + ((-.4 + beta_NMLS_CP_c[x])
                           * data["NMLS_CP_s"].iloc[i])
                        + ((0.5 + beta_crt_c[x]) *
                           crt_delta[data["CRT"].iloc[i]])
                        + ((.8 + beta_fam_c[x]) *
                           fam_delta[data["FAM"].iloc[i]])
                        + inglehart_c[data["PO"].iloc[i]]
                        for i, x in enumerate(data["country"])])
    p = logistic(p_log)

    n_trials = 20 + rng.binomial(5, p=.5, size=3*n_user)

    guess = rng.binomial(n_trials, p=p)
    data["p_log"] = p_log
    data["p"] = p
    data["N_correct"] = guess
    data["N_trials"] = n_trials

    return data


def pooled_normal(name: str, dims: Iterable[str], lam: float = 1., centered: bool = False) -> at.TensorVariable:
    """Create a pooled normal variable
    """
    sigma = pm.Exponential(f"{name}_sigma", lam=lam)

    if centered:
        var = pm.Normal(name, mu=0, sigma=sigma, dims=dims)
    else:
        z_var = pm.Normal(f"{name}_z", mu=0, sigma=1., dims=dims)
        var = pm.Deterministic(name, z_var * sigma, dims=dims)

    return var


def mv_normal(name: str, dims: Iterable[str], n_sigma: int, lam: float = 1., eta: float = 2.0, centered: bool = False) -> at.TensorVariable:
    """Create a multidimensional normal variable.
    """
    # prior stdev
    sd_dist = pm.Exponential.dist(lam=lam, shape=n_sigma)

    # cholesky decomposition of cov matrix
    chol, _, _ = pm.LKJCholeskyCov(
        f"{name}_chol", n=n_sigma, eta=eta, sd_dist=sd_dist)

    if centered:
        var = pm.MvNormal(name, mu=0, chol=chol, dims=dims)
    else:
        z_var = pm.Normal(f"{name}_z", mu=0.0, sigma=1.0, dims=dims)
        var = pm.Deterministic(name, at.dot(chol, z_var), dims=dims)

    return var


def l2_gp_with_sigma(
    name: str,
    data: np.ndarray,
    etasq_prior: Union[float, at.TensorVariable] = 1,
    ls_inv_prior: Union[float, at.TensorVariable] = 2.5,
    sigma_prior: Union[float, at.TensorVariable] = 1,
    dims: Union[str, List[str]] = "Countries",
) -> at.TensorVariable:
    """Create a gaussian process with L2-Kernel.
    """
    etasq = pm.Exponential(f"{name}_etasq", etasq_prior)
    ls_inv = pm.HalfNormal(f"{name}_ls_inv", ls_inv_prior)
    sigma = pm.Exponential(f"{name}_sigma", sigma_prior)
    _ = pm.Deterministic(f"{name}_rhosq", .5 * ls_inv ** 2)

    # sigma when x=x'
    in_group = at.eye(data.shape[0]) * sigma
    in_group = in_group * etasq

    # etasq * (ExpQuad + in_group) tries to use flatiter which is not supported by jax
    # so we do it this way ¯\_(ツ)_/¯
    cov = etasq * pm.gp.cov.ExpQuad(input_dim=1, ls_inv=ls_inv)
    cov = cov + in_group
    gp = pm.gp.Latent(cov_func=cov)

    return gp.prior(name, X=data[:, None], dims=dims)


def encode_ordinal(delta: at.TensorVariable) -> at.TensorVariable:
    """Encode ordered levels.
    """
    # 0 for first level
    zeros = np.zeros(1)
    delta_j = pm.math.concatenate([zeros, delta])
    delta_j_cumulative = at.cumsum(delta_j)

    return delta_j_cumulative


def _clean_for_controls(data: pd.DataFrame, hearing: bool = False) -> pd.DataFrame:
    # remove missing values
    if hearing:
        data = data[data.hearing_enc == -1]
    else:
        data = data[data.device_enc == -1]

    return data


def load_long_data(drop_zero: bool = True, controls: bool = False, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an encode data for analysis.
    """
    ratings_df = pd.read_csv("data/ratings.csv")
    score_df = pd.read_csv("data/scores.csv")
    user_df = pd.read_csv("data/user.csv")

    ratings_filtered = filter_ratings(
        ratings_df=ratings_df, user_df=user_df, drop_zero=drop_zero, **kwargs).reset_index(drop=True)
    data_all = ratings_filtered.merge(user_df, on="id")
    data_all = data_all.merge(score_df, on="id")

    # filter faulty PO ratings
    data_all = data_all[~data_all.PO.isna()]

    data_all = encode_lang_and_age(data_all)

    data_all["idx"] = data_all["index"].astype(int) - 1
    data_all.correct = data_all.correct.astype(float)
    data_all["type_enc"], _ = pd.factorize(data_all["type"], sort=True)
    data_all["type_enc"], _ = pd.factorize(data_all["type"], sort=True)
    data_all["rating_s"] = standardize(data_all["rating"])

    text = data_all[data_all.media_type_x ==
                    "text"].copy().reset_index(drop=True)
    image = data_all[data_all.media_type_x ==
                     "image"].copy().reset_index(drop=True)
    audio = data_all[data_all.media_type_x ==
                     "audio"].copy().reset_index(drop=True)

    audio["id_enc"], _ = pd.factorize(audio["id"], sort=True)
    image["id_enc"], _ = pd.factorize(image["id"], sort=True)
    text["id_enc"], _ = pd.factorize(text["id"], sort=True)

    audio["item_enc"], _ = pd.factorize(audio["file_path"], sort=True)
    image["item_enc"], _ = pd.factorize(image["file_path"], sort=True)
    text["item_enc"], _ = pd.factorize(text["file_path"], sort=True)

    if controls:
        audio = _clean_for_controls(audio, hearing=True)
        image = _clean_for_controls(image)
        text = _clean_for_controls(text)

    return audio, image, text


def load_acc_data(real_fake: bool = False, controls: bool = False, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an encode data for analysis.
    """
    ratings_df = pd.read_csv("data/ratings.csv")
    score_df = pd.read_csv("data/scores.csv")
    user_df = pd.read_csv("data/user.csv")

    data_all = calculate_acc_per_user(
        ratings_df=ratings_df, user_df=user_df, **kwargs)

    # merge scores
    data_all = data_all.merge(score_df, on="id")

    # filter faulty PO ratings
    data_all = data_all[~data_all.PO.isna()]

    data_all = encode_lang_and_age(data_all)

    data_all.N_trials = data_all.N_trials.astype(int)
    data_all.N_correct = data_all.N_correct.astype(int)
    data_all.N_correct_real = data_all.N_correct_real.astype(int)
    data_all.N_correct_fake = data_all.N_correct_fake.astype(int)

    audio = data_all[data_all.media_type == "audio"].reset_index(drop=True)
    image = data_all[data_all.media_type == "image"].reset_index(drop=True)
    text = data_all[data_all.media_type == "text"].reset_index(drop=True)

    if controls:
        audio = _clean_for_controls(audio, hearing=True)
        image = _clean_for_controls(image)
        text = _clean_for_controls(text)

    return audio, image, text


def analyze_trace(
    path: Path,
    media_to_consider: Iterable[str] = ("audio", "image", "text"),
) -> List[pd.DataFrame]:
    """Analyze traces and collect data on predictors.
    """
    pred_data = []
    for media in media_to_consider:
        traces = []
        for p in path.joinpath(Path(media)).glob("*.nc"):
            try:
                traces.append(az.from_netcdf(p))
            except:
                raise ValueError(f"Could not load: {p}")

        data = []
        for trace in traces:
            predictors = [
                var for var in trace.posterior.data_vars if "beta" in var and "_z" not in var and "_sigma" not in var and "PO" not in var]
            hdi = az.hdi(trace).get(predictors)
            for pred in predictors:
                pred_hdi = hdi.get(pred)
                pred_name = pred.replace("beta_", "").replace("_", " ")
                if "Country" in pred_hdi.coords:
                    for country in pred_hdi["Country"]:
                        lower, higher = pred_hdi.sel(Country=country).values

                        post_data = trace.posterior[pred].sel(Country=country)
                        mean = post_data.mean()

                        if mean > 0:
                            prob_significant = (post_data > 0.).sum()
                        else:
                            prob_significant = (post_data < 0.).sum()

                        prob_significant = prob_significant / post_data.count()

                        data.append([f"{pred_name} {str(country.values)}",
                                    mean.values, lower, higher, prob_significant.values])
                else:
                    mean = trace.posterior[pred].mean().values

                    if mean > 0:
                        prob_significant = (trace.posterior[pred] > 0.).sum()
                    else:
                        prob_significant = (trace.posterior[pred] < 0.).sum()

                    prob_significant = prob_significant / \
                        trace.posterior[pred].count()

                    data.append(
                        [pred_name, mean, *pred_hdi.values, prob_significant.values])

        data = pd.DataFrame(
            data, columns=["name", "mean", "low", "high", "prob_significant"])
        pred_data.append(data)

    return pred_data
