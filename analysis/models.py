"""Models used in the analysis.
"""
import os
from copy import deepcopy
from dataclasses import astuple, dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm

import utils
from model_utils import (encode_ordinal, l2_gp_with_sigma, mv_normal,
                         pooled_normal)

COORDS = {
    "Country": ["USA", "Germany", "China"],
    "Education": ["Low", "Medium", "High"],
    "Inglehart": ["Postmaterialism", "Postmaterialism-M", "Materialism-M", "Materialism"],
}

RANDOM_SEED = 42


class VarType(Enum):
    """The different variables we support.
    """
    CONTINOUS = auto()
    INDEX = auto()
    ORDINAL = auto()


@dataclass
class Variable:
    """The definition for a model variable.
    """
    var_type: VarType
    name: str
    data: np.ndarray
    dims: Optional[Tuple[str]] = None

    def __iter__(self):
        return iter(astuple(self))


def _mv_normal_per_type(name: str, model: at.TensorVariable, dims: Tuple[str], **kwargs) -> at.TensorVariable:
    n_dims = len(dims)
    assert n_dims <= 2 and n_dims != 0
    if n_dims == 1:
        return mv_normal(name, dims=("Type", *dims), n_sigma=2, **kwargs)

    var = []
    for media_type in model.coords["Type"]:
        var_type = mv_normal(f"{name}_{media_type}", dims=dims, n_sigma=len(
            model.coords[dims[0]]), **kwargs)
        var.append(var_type)

    var = pm.Deterministic(name, at.stack(var), dims=("Type", *dims))
    return var


def _pooled_normal_per_type(name: str, model: at.TensorVariable, dims: Tuple[str], **kwargs) -> at.TensorVariable:
    var = []
    for media_type in model.coords["Type"]:
        var_type = pooled_normal(f"{name}_{media_type}", dims=dims, **kwargs)
        var.append(var_type)

    var = pm.Deterministic(name, at.stack(var), dims=("Type", *dims))
    return var


def _age_by_type(age_ids: np.ndarray, model: at.TensorVariable) -> at.TensorVariable:
    age_effect = []
    for media_type in model.coords["Type"]:
        age_effect_c = []

        # we partially pool parameter on population level (split by type)
        etasq_prior = pm.HalfCauchy(f"gp_etasq_{media_type}", beta=1.)
        ls_inv_prior = pm.HalfCauchy(f"gp_ls_inv_{media_type}", beta=1.)
        sigma_prior = pm.Exponential(f"gp_sigma_{media_type}", lam=1.)

        for country in model.coords["Country"]:
            age = l2_gp_with_sigma(
                f"age_{country}_{media_type}",
                age_ids,
                dims="Age",
                etasq_prior=etasq_prior,
                ls_inv_prior=ls_inv_prior,
                sigma_prior=sigma_prior,
            )
            age_effect_c.append(age)

        age_effect.append(age_effect_c)

    age_effect = pm.Deterministic("age", at.stack(
        age_effect), dims=("Type", "Country", "Age"))

    return age_effect


def _fixed_effects(fixed: List[Variable], country_idx: at.TensorVariable, by_type: bool = False, type_idx: Optional[at.TensorVariable] = None) -> at.TensorVariable:
    effects = []
    for var_type, name, data, dims in fixed:
        if by_type:
            if dims:
                dims = ("Type", *dims)
            else:
                dims = ("Type", )

        dat = pm.Data(name, data, dims="obs_idx", mutable=True)
        var = pm.Normal(f"beta_{name}", mu=0., sigma=1., dims=dims)

        # create variable
        if var_type == VarType.CONTINOUS:
            if by_type:
                if len(dims) == 2:
                    assert dims[1] == "Country", "We only support country fixed effects atm."
                    contrib = var[:, country_idx].T
                else:
                    contrib = var

                contrib = (contrib * dat[:, None]).T
            else:
                if dims:
                    assert len(
                        dims) == 1 and dims[0] == "Country", "We only support no-pooling country wise atm."
                    contrib = var[country_idx]
                else:
                    contrib = var

                contrib = contrib * dat

        elif var_type == VarType.INDEX:
            if by_type:
                assert dims is not None
                if len(dims) == 3:
                    assert dims[
                        1] == "Country", f"We only support country fixed effects atm: {dims}"
                    contrib = var[:, country_idx, dat]
                else:
                    contrib = var[:, dat]
            else:
                assert dims is not None
                if len(dims) == 2:
                    assert dims[0] == "Country", "We only support no-pooling country wise atm."
                    contrib = var[country_idx, dat]
                else:
                    contrib = var[dat]

        elif var_type == VarType.ORDINAL:
            n_level = len(np.unique(data))
            delta = pm.Dirichlet(
                f"delta_{name}", np.repeat(2., n_level-1), shape=n_level-1)

            delta_cumulative = encode_ordinal(delta)

            if by_type:
                if len(dims) == 2:
                    assert dims[1] == "Country", "We only support country fixed effects atm."
                    contrib = var[:, country_idx].T
                else:
                    contrib = var

                contrib = (contrib * delta_cumulative[dat, None]).T
            else:
                if dims:
                    assert len(
                        dims) == 1 and dims[0] == "Country", "We only support no-pooling country wise atm."
                    contrib = var[country_idx]
                else:
                    contrib = var

                contrib = contrib * delta_cumulative[dat]

        else:
            raise NotImplementedError(
                f"Var type {var_type} not implemetned for fixed variables ({name})")

        if type_idx is None:
            contrib = pm.Deterministic(f"contrib_{name}", contrib)
        else:
            contrib = pm.Deterministic(f"contrib_{name}", at.where(
                type_idx, contrib[1], contrib[0]))
        effects.append(contrib)

    return effects


def mixed_effect_model(
    data: pd.DataFrame,
    time_correction: bool = True,
    edu_correction: bool = True,
    age_correction: bool = True,
    fixed: Optional[List[Variable]] = None,
    dimensions: Optional[Dict[Any, Any]] = None,
    control: Optional[str] = None,
):
    """Mixed effect model for estimating real/fake probabilites.
    """
    fixed = [] if fixed is None else fixed

    coords = deepcopy(COORDS)
    coords.update({
        "Type": ["Fake", "Real"],
        "Age": utils.AGE_LABELS,
    })

    if dimensions is not None:
        coords.update(dimensions)

    with pm.Model(coords=coords) as model:
        y_obs_real = pm.Data("y_obs_real", data.N_correct_real.values,
                             dims="obs_idx", mutable=True)
        y_obs_fake = pm.Data("y_obs_fake", data.N_correct_fake.values,
                             dims="obs_idx", mutable=True)

        N_trials_real = pm.Data("N_trials_real", data.N_trials_real.values,
                                dims="obs_idx", mutable=True)
        N_trials_fake = pm.Data("N_trials_fake", data.N_trials_fake.values,
                                dims="obs_idx", mutable=True)

        country_idx = pm.Data("country_idx", data.country_enc.values,
                              dims="obs_idx", mutable=True)

        median_times = pm.Data("median_time", data.median_time_s.values,
                               dims="obs_idx", mutable=True)

        predictors = []
        # ================================================================
        # Intercept
        # ================================================================
        alpha = _pooled_normal_per_type(
            "alpha", model=model, dims=("Country", ), lam=2.)
        predictors.append(alpha[:, country_idx])

        # ================================================================
        # Median Time
        # ================================================================
        if time_correction:
            time = _pooled_normal_per_type(
                "time", model=model, dims=("Country", ), lam=1.)
            predictors.append(time[:, country_idx] * median_times)

        # ================================================================
        # Education
        # ================================================================
        if edu_correction:
            edu_idx = pm.Data("edu_idx", data.edu_enc.values,
                              dims="obs_idx", mutable=True)
            edu = _mv_normal_per_type(
                "edu", model=model, dims=("Country", "Education"), lam=1.)
            predictors.append(edu[:, country_idx, edu_idx])

        # ================================================================
        # Age
        # ================================================================
        if age_correction:
            age_idx = pm.Data("age_idx", data.age_enc.values,
                              dims="obs_idx", mutable=True)

            age_ids = np.arange(1, len(utils.AGE_LABELS) + 1)
            assert len(age_ids) == len(utils.AGE_LABELS)
            age_ids.sort()

            age_ids = age_ids / age_ids.max()
            age_effect = _age_by_type(age_ids, model)

            predictors.append(age_effect[:, country_idx, age_idx])

        # ================================================================
        # Fixed variables
        # ================================================================
        fixed_effects = _fixed_effects(fixed, country_idx, by_type=True)
        predictors.extend(fixed_effects)

        # ================================================================
        # Inference
        # ================================================================
        pi_log = pm.Deterministic("pi_log", at.sum(
            predictors, axis=0), dims=("Type", "obs_idx"))

        if control is None:
            pi_log_fake = pm.Deterministic(
                "pi_log_fake", pi_log[0, :], dims="obs_idx")
            pi_log_real = pm.Deterministic(
                "pi_log_real", pi_log[1, :], dims="obs_idx")

            _ = pm.Binomial("y_fake", logit_p=pi_log_fake, n=N_trials_fake,
                            observed=y_obs_fake, dims="obs_idx")
            _ = pm.Binomial("y_real", logit_p=pi_log_real, n=N_trials_real,
                            observed=y_obs_real, dims="obs_idx")
        elif control == "device":
            n_level = data.device.dropna().nunique()
            device_idx = data.device_enc.values

            beta_device = pm.Normal(
                "beta_device", mu=1., sigma=1.5, dims=("Type", "Device"))
            device_prior = pm.Dirichlet(
                "device", np.repeat(2., n_level), size=2)

            assert tuple(device_prior.shape.eval()) == (2, 3)

            known_mask = device_idx != -1
            n_unkown = sum(~known_mask)
            pi_unkown = pi_log[:, ~known_mask]

            data_unkown = np.vstack([
                data.N_correct_fake.values[~known_mask],
                data.N_correct_real.values[~known_mask],
            ])

            custom_logp = pm.math.logsumexp(
                [
                    pm.math.log(device_prior[:, 0])[:, None] *
                    at.ones(n_unkown),
                    pm.logp(pm.Binomial.dist(
                        n=data.N_trials.values[~known_mask],
                        logit_p=pi_unkown + beta_device[:, 0][:, None]), data_unkown),
                    pm.math.log(device_prior[:, 1])[:, None] *
                    at.ones(n_unkown),
                    pm.logp(pm.Binomial.dist(
                        n=data.N_trials.values[~known_mask],
                        logit_p=pi_unkown + beta_device[:, 1][:, None]), data_unkown),
                    pm.math.log(device_prior[:, 2])[:, None] *
                    at.ones(n_unkown),
                    pm.logp(pm.Binomial.dist(
                        n=data.N_trials.values[~known_mask],
                        logit_p=pi_unkown + beta_device[:, 2][:, None]), data_unkown),
                ]
            )

            y_unkown = pm.Potential("y|device not known", custom_logp)

            # known
            pi_known = pi_log[:, known_mask] + \
                beta_device[:, device_idx[known_mask]]

            pi_known_fake = pm.Deterministic(
                "pi_log_fake", pi_known[0], dims="obs_idx")
            pi_known_real = pm.Deterministic(
                "pi_log_real", pi_known[1], dims="obs_idx")

            y_known_fake = pm.Binomial("y_fake|device known", logit_p=pi_known_fake, n=N_trials_fake[known_mask],
                                       observed=data.N_correct_fake.values[known_mask])
            y_known_real = pm.Binomial("y_real|device known", logit_p=pi_known_real, n=N_trials_real[known_mask],
                                       observed=data.N_correct_real.values[known_mask])

    return model


def mixed_effect_time_rating(
    data: pd.DataFrame,
    time_correction: bool = True,
    edu_correction: bool = True,
    age_correction: bool = True,
    fixed: Optional[List[Variable]] = None,
    no_data: bool = False,


) -> at.TensorVariable:
    """Mixed effect model for modelling rating.
    """
    fixed = [] if fixed is None else fixed
    index_ids = list(sorted(data.idx.unique()))
    user_ids = list(sorted(data.id_enc.unique()))
    item_idx = list(sorted(data.item_enc.unique()))

    coords = deepcopy(COORDS)
    coords.update({
        "Type": ["Fake", "Real"],
        "Index": index_ids,
        "Age": utils.AGE_LABELS,
        "User": user_ids,
        "Item": item_idx,
        "obs_idx": data.index,
    })

    with pm.Model(coords=coords) as model:
        if no_data:
            rating = data.rating_s.values
            index = data.idx.values
            country_idx = data.country_enc.values
            user_idx = data.id_enc.values
            type_idx = data.type_enc.values
            item_idx = data.item_enc.values
        else:
            rating = pm.Data("rating", data.rating_s.values,
                             dims="obs_idx", mutable=True)
            index = pm.Data("index", data.idx.values,
                            dims="obs_idx", mutable=True)
            country_idx = pm.Data("country_idx", data.country_enc.values,
                                  dims="obs_idx", mutable=True)
            user_idx = pm.Data("participant_idx", data.id_enc.values,
                               dims="user_idx", mutable=True)
            type_idx = pm.Data("type_idx", data.type_enc.values,
                               dims="obs_idx", mutable=True)
            item_idx = pm.Data("item_idx", data.item_enc.values,
                               dims="obs_idx", mutable=True)

        predictors = []
        # ========================================
        # participant intercept per type
        # ========================================
        alpha = pm.Normal("alpha", mu=0., sigma=1., dims=("Type", "User"))
        predictors.append(alpha[type_idx, user_idx])

        # ========================================
        # per item uncertainty
        # ========================================
        item_sigma = pm.Exponential("item_sigma", lam=2.)
        item = pm.Normal("item", mu=0., sigma=item_sigma, dims=("Item"))
        predictors.append(item[item_idx])

        # ========================================
        # fixed term per timestep and country
        # ========================================
        if time_correction:
            time = _mv_normal_per_type(
                "time", model=model, dims=("Country", "Index"))
            predictors.append(time[type_idx, country_idx, index])

        # ========================================
        # fixed term per education and country
        # ========================================
        if edu_correction:
            if no_data:
                edu_idx = data.edu_enc.values
            else:
                edu_idx = pm.Data("edu_idx", data.edu_enc.values,
                                  dims="obs_idx", mutable=True)

            edu = _mv_normal_per_type(
                "edu", model=model, dims=("Country", "Education"))
            predictors.append(edu[type_idx, country_idx, edu_idx])

        # ========================================
        # fixed term per age and country
        # ========================================
        if age_correction:
            if no_data:
                age_idx = data.age_enc.values
            else:
                age_idx = pm.Data("age_idx", data.age_enc.values,
                                  dims="obs_idx", mutable=True)

            age_ids = np.arange(1, len(utils.AGE_LABELS) + 1)
            assert len(age_ids) == len(utils.AGE_LABELS)
            age_ids.sort()

            age_ids = age_ids / age_ids.max()
            age_effect = _age_by_type(age_ids, model)

            predictors.append(age_effect[type_idx, country_idx, age_idx])

        # ================================================================
        # Fixed variables
        # ================================================================
        fixed_effects = _fixed_effects(
            fixed, country_idx, by_type=True, type_idx=type_idx)
        predictors.extend(fixed_effects)

        # ========================================
        # inference
        # ========================================
        if no_data:
            mu = at.sum(predictors, axis=0)
        else:
            mu = pm.Deterministic(
                "mu",
                at.sum(predictors, axis=0),
                dims=("obs_idx", ),
            )
        assert tuple(mu.shape.eval()) == (len(data), )

        sigma = pm.Exponential(
            "sigma",
            lam=2.,
        )

        _ = pm.Normal(
            "y", mu=mu, sigma=sigma, observed=rating, dims=("obs_idx", ))

    return model


def mixed_effect_time_model(
    data: pd.DataFrame,
    time_correction: bool = True,
    edu_correction: bool = True,
    age_correction: bool = True,
    fixed: Optional[List[Variable]] = None,
) -> at.TensorVariable:
    """Defines a mixed effect models which work in the time domain.
    """
    fixed = [] if fixed is None else fixed

    index_ids = list(sorted(data.idx.unique()))
    user_ids = list(sorted(data.id_enc.unique()))

    coords = deepcopy(COORDS)
    coords.update({
        "Type": ["Fake", "Real"],
        "Index": index_ids,
        "Age": utils.AGE_LABELS,
        "User": user_ids,
    })
    with pm.Model(coords=coords) as model:
        # ========================================
        # Data
        # ========================================
        correct = pm.Data("correct", data.correct.values,
                          dims="obs_idx", mutable=True)
        index = pm.Data("index", data.idx.values, dims="obs_idx", mutable=True)
        country_idx = pm.Data("country_idx", data.country_enc.values,
                              dims="obs_idx", mutable=True)
        user_idx = pm.Data("participant_idx", data.id_enc.values,
                           dims="user_idx", mutable=True)
        type_idx = pm.Data("type_idx", data.type_enc.values,
                           dims="obs_idx", mutable=True)

        predictors = []
        # ========================================
        # participant intercept
        # ========================================
        alpha = pm.Normal("alpha", mu=0., sigma=1.5, dims=("Type", "User"))
        predictors.append(alpha[type_idx, user_idx])

        # ========================================
        # country fixed term
        # ========================================
        country = _mv_normal_per_type(
            "country", model=model, dims=("Country", ))
        predictors.append(country[type_idx, country_idx])

        # ========================================
        # fixed term per timestep and country
        # ========================================
        if time_correction:
            delta = _mv_normal_per_type(
                "delta", model=model, dims=("Country", "Index"))
            predictors.append(delta[type_idx, country_idx, index])

        # ========================================
        # fixed term per education and country
        # ========================================
        if edu_correction:
            edu_idx = pm.Data("edu_idx", data.edu_enc.values,
                              dims="obs_idx", mutable=True)

            edu = _mv_normal_per_type(
                "edu", model=model, dims=("Country", "Education"))
            predictors.append(edu[type_idx, country_idx, edu_idx])

        # ========================================
        # fixed term per age and country
        # ========================================
        if age_correction:
            age_idx = pm.Data("age_idx", data.age_enc.values,
                              dims="obs_idx", mutable=True)

            age_ids = data.age_enc.unique() + 1
            age_ids.sort()

            age_ids = age_ids / age_ids.max()
            age_effect = _age_by_type(age_ids, model)

            predictors.append(age_effect[type_idx, country_idx, age_idx])

        # ================================================================
        # Fixed variables
        # ================================================================
        fixed_effects = _fixed_effects(fixed, country_idx, by_type=True)
        predictors.extend(fixed_effects)

        # ========================================
        # inference
        # ========================================
        pi_log = pm.Deterministic(
            "pi_log",
            at.sum(predictors, axis=0),
            dims=("obs_idx", ),
        )
        pi = pm.Deterministic(
            "pi",
            pm.math.invlogit(pi_log),
            dims=("obs_idx", ),
        )

        _ = pm.Binomial("y", p=pi, n=1, observed=correct, dims="obs_idx")

    return model


def define_model(
    data: pd.DataFrame,
    time_correction: bool = True,
    edu_correction: bool = True,
    age_correction: bool = True,
    fixed: Optional[List[Variable]] = None,
) -> at.TensorVariable:
    """Define a model based on the vars provided.
    """
    fixed = [] if fixed is None else fixed

    coords = deepcopy(COORDS)
    coords.update({
        "Type": ["Fake", "Real"],
        "Age": utils.AGE_LABELS,
    })

    with pm.Model(coords=COORDS) as model:
        # ================================================================
        # Data section
        # ================================================================
        y_obs = pm.Data("y_obs", data.N_correct.values,
                        dims="obs_idx", mutable=True)
        N_trials = pm.Data("N_trials", data.N_trials.values,
                           dims="obs_idx", mutable=True)
        country_idx = pm.Data("country_idx", data.country_enc.values,
                              dims="obs_idx", mutable=True)

        # ================================================================
        # Intercept + Offsets
        # ================================================================
        predictors = []

        alpha_c = pooled_normal(
            "alpha_c", lam=2., dims="Country")
        predictors.append(alpha_c[country_idx])

        # ================================================================
        # Median Time
        # ================================================================
        if time_correction:
            median_times = pm.Data("median_time", data.median_time_s.values,
                                   dims="obs_idx", mutable=True)

            time = pooled_normal(
                "time", dims=("Country", ), lam=1.)
            predictors.append(time[country_idx] * median_times)

        # ================================================================
        # Education
        # ================================================================
        if edu_correction:
            edu_idx = pm.Data("edu_idx", data.edu_enc.values,
                              dims="obs_idx", mutable=True)
            edu = mv_normal("edu", dims=("Country", "Education"), n_sigma=3,)

            predictors.append(edu[country_idx, edu_idx])

        # ================================================================
        # Age
        # ================================================================
        if age_correction:
            age_idx = pm.Data("age_idx", data.age_enc.values,
                              dims="obs_idx", mutable=True)

            age_ids = np.arange(1, len(utils.AGE_LABELS) + 1)
            assert len(age_ids) == len(utils.AGE_LABELS)
            age_ids.sort()

            age_ids = age_ids / age_ids.max()

            # we partially pool parameter on population level (split by type)
            etasq_prior = pm.HalfCauchy("gp_etasq", beta=1.)
            ls_inv_prior = pm.HalfCauchy("gp_ls_inv", beta=1.)
            sigma_prior = pm.Exponential("gp_sigma", lam=1.)

            age_effect_c = []
            for country in model.coords["Country"]:
                age = l2_gp_with_sigma(
                    f"age_{country}",
                    age_ids,
                    dims="Age",
                    etasq_prior=etasq_prior,
                    ls_inv_prior=ls_inv_prior,
                    sigma_prior=sigma_prior,
                )
                age_effect_c.append(age)

            age_effect = pm.Deterministic("age", at.stack(
                age_effect_c), dims=("Country", "Age"))

            predictors.append(age_effect[country_idx, age_idx])

        # ================================================================
        # Fixed variables
        # ================================================================
        effects = _fixed_effects(fixed, country_idx)
        predictors.extend(effects)

        # ================================================================
        # Outcome
        # ================================================================
        pi_log = pm.Deterministic(
            "pi_log",
            at.sum(predictors, axis=0),
            dims="obs_idx",
        )
        pi = pm.Deterministic("pi", pm.math.invlogit(pi_log), dims="obs_idx")

        _ = pm.Binomial("y", n=N_trials,
                        p=pi, observed=y_obs, dims="obs_idx")

    return model


def train_model(
        model: at.TensorVariable,
        jax_cpu: bool = False,
        gpu: bool = False,
        sampler_kwargs: Optional[Dict] = None) -> Dict:
    """Function for training all models in one go.
    """
    # trace
    sampler_args = sampler_kwargs or {}
    with model:
        if gpu:
            from pymc.sampling_jax import sample_numpyro_nuts
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
            trace = sample_numpyro_nuts(**sampler_args)
        elif jax_cpu:
            import jax
            from pymc.sampling_jax import sample_numpyro_nuts
            jax.config.update('jax_platform_name', 'cpu')
            trace = sample_numpyro_nuts(**sampler_args)
        else:
            trace = pm.sample(**sampler_args)

    # sample predictions
    with model:
        sample = pm.sample_posterior_predictive(trace)

    trace.extend(sample)

    return {
        "model": model,
        "trace": trace,
    }


MODELS_AND_CATEGORY = [("rating", partial(mixed_effect_time_rating, no_data=True)),
                       ("accuracy", define_model), ("real_fake", mixed_effect_model)]


def _build_demographic_mode() -> Dict[str, Dict[str, object]]:
    demo_models = {}
    for (category, model_fn) in MODELS_AND_CATEGORY:
        demo_models[category] = {
            "base": partial(model_fn, time_correction=False, edu_correction=False, age_correction=False),
            "time": partial(model_fn, edu_correction=False, age_correction=False),
            "edu": partial(model_fn, time_correction=False, age_correction=False),
            "time_edu": partial(model_fn, age_correction=False),
            "age": partial(model_fn, time_correction=False, edu_correction=False),
            "time_age": partial(model_fn, edu_correction=False),
            "age_edu": partial(model_fn, time_correction=False),
            "time_age_edu": model_fn,
        }

    return demo_models


DEMOGRAPHIC_MODELS: Dict[str, Dict[str, object]] = _build_demographic_mode()

AUDIO_MODEL = "time_age_edu"
IMAGE_MODEL = "time_age_edu"
TEXT_MODEL = "time_age"


def _build_selected_models() -> Dict[str, Dict[str, Tuple[Callable]]]:
    sel_models = {}
    for (category, _) in MODELS_AND_CATEGORY:
        sel_models[category] = {
            "audio": (AUDIO_MODEL, DEMOGRAPHIC_MODELS[category][AUDIO_MODEL]),
            "image": (IMAGE_MODEL, DEMOGRAPHIC_MODELS[category][IMAGE_MODEL]),
            "text": (TEXT_MODEL, DEMOGRAPHIC_MODELS[category][TEXT_MODEL]),
        }

    return sel_models


SELECTED_MODELS: Dict[str, Dict[str, Tuple[str, Callable]]
                      ] = _build_selected_models()
