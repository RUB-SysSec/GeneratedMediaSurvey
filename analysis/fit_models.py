"""This script fits all models used in the analysis and saves their traces."""
import argparse
from functools import partial
from pathlib import Path
from typing import Callable, List

import pandas as pd

import model_utils
import models

AVAILABLE_MODELS = ["rating", "accuracy", "real_fake"]

def _compute_single_pairs(predictors: List[List[models.Variable]]):
    res = list()
    for i in range(len(predictors) - 1):
      for j in range(i+1, len(predictors)):
        res.append(predictors[i] + predictors[j])

    return res

def _train_and_save_model(
    data: pd.DataFrame,
    model_fn: Callable,
    save_dir: Path,
    model_name: str,
    args: argparse.Namespace,
):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    res = models.train_model(
        model_fn(data),
        sampler_kwargs={
            # at least as much samples as tuning steps
            "draws": max(args.tune, 2_000),
            "tune": args.tune,
            "target_accept": args.target_accept,
        },
        gpu=args.gpu,
        jax_cpu=args.cpu,
    )

    trace = res["trace"]
    model_path = save_dir.joinpath(Path(f"{model_name}.nc"))

    print(f"Saving trace to {model_path}...")
    trace.to_netcdf(model_path)


def main(args: argparse.Namespace):
    """Main.
    """
    model_to_train = args.model
    if model_to_train not in AVAILABLE_MODELS:
        raise ValueError("Selected model not supported.")

    print("# ========================================")
    print("Loading data....")
    if model_to_train == "real_fake":
        audio, image, text = model_utils.load_acc_data(
            real_fake=True, filter_type=args.filtering, controls=args.control or args.complete)
    elif model_to_train == "accuracy":
        audio, image, text = model_utils.load_acc_data(
            filter_type=args.filtering, controls=args.control or args.complete)
    elif model_to_train == "rating":
        audio, image, text = model_utils.load_long_data(
            drop_zero=False, filter_type=args.filtering, controls=args.control or args.complete)

    merged = pd.concat((audio, image, text))
    print(merged.groupby(["country"])["id"].nunique())

    print("Loading data done!")
    print("# ========================================")
    base_path = Path(
        f"{model_utils.TRACE_DIR}/{model_to_train}/{args.filtering}/")

    if args.combinations:
      predictors = _compute_single_pairs(predictors)
      base_path = base_path.joinpath("combinations/")

    for data_name, data in [("audio", audio), ("image", image), ("text", text)]:
        if args.demographics:
            for model_name, model_fn in models.DEMOGRAPHIC_MODELS[model_to_train].items():
                print("# ========================================")
                print(f"Fitting {model_name} model on {data_name}")
                print("# ========================================")
                demo_save_dir = base_path.joinpath(
                    Path(f"demographics/{data_name}"))

                if model_to_train == "rating":
                    model_fn = partial(
                        model_fn, time_correction=False, no_data=True)

                _train_and_save_model(
                    data, model_fn, demo_save_dir, model_name, args)

        predictors = [
            [
                models.Variable(models.VarType.CONTINOUS,
                                "AHS", data.AHS_s.values)
            ],
            [
                models.Variable(models.VarType.CONTINOUS,
                                "GTS", data.GTS_s.values)
            ],
            [
                models.Variable(models.VarType.CONTINOUS,
                                "NMLS_FC", data.NMLS_FC_s.values),
                models.Variable(models.VarType.CONTINOUS,
                                "NMLS_CC", data.NMLS_CC_s.values),
                models.Variable(models.VarType.CONTINOUS,
                                "NMLS_FP", data.NMLS_FP_s.values),
                models.Variable(models.VarType.CONTINOUS,
                                "NMLS_CP", data.NMLS_CP_s.values)
            ],
            [
                models.Variable(models.VarType.ORDINAL, "CRT", data.CRT.values)
            ],
            [
                models.Variable(models.VarType.ORDINAL, "FAM", data.FAM.values)
            ],
            [
                models.Variable(models.VarType.ORDINAL, "PO", data.PO.values),
            ],
        ]

        if args.fixed or args.fixed_country:
            (model_name,
             model_fn) = models.SELECTED_MODELS[model_to_train][data_name]

            for predictor in predictors:
                name = "_".join([pred.name for pred in predictor])
                print("# ========================================")
                print(
                    f"Fitting {model_name} model on {data_name} and {name}")
                print("# ========================================")

                path = "fixed"
                if args.fixed_country:
                    path += "_country"

                    for pred in predictor:
                        pred.dims = ("Country", *(pred.dims or ()))

                if args.complete:
                    path += "_complete"

                if args.control:
                    path += "_control"

                    if data_name == "audio":
                        predictor.append(
                            models.Variable(models.VarType.INDEX, "hearing",
                                            data.hearing_enc.values.astype(int), ("Hearing", )),
                        )
                        model_fn = partial(model_fn, dimensions={
                            "Hearing": list(sorted(data.hearing.unique())),
                        })
                    else:
                        predictor.append(
                            models.Variable(models.VarType.INDEX, "device",
                                            data.device_enc.values, ("Device", )),
                        )
                        model_fn = partial(model_fn, dimensions={
                            "Device": list(sorted(data.device.unique())),
                        })

                effect_fn = partial(model_fn, fixed=predictor)

                if model_to_train == "rating":
                    effect_fn = partial(
                        effect_fn, time_correction=False, no_data=True)

                save_dir = base_path.joinpath(Path(f"{path}/{data_name}"))

                _train_and_save_model(
                    data=data,
                    model_fn=effect_fn,
                    save_dir=save_dir,
                    model_name=f"{model_name}_{name}",
                    args=args,
                )

        print("# ========================================\n")


def parse_args() -> argparse.Namespace:
    """Arguments for running the script.
    """
    parser = argparse.ArgumentParser(
        description="Fit all models used in the analyiss.")

    parser.add_argument(
        "model", help=f"Select the models to train. Available: {' '.join(AVAILABLE_MODELS)}", type=str
    )

    parser.add_argument(
        "--cpu", "-c", help="Fit models on cpu but use jax backend.", action="store_true")
    parser.add_argument(
        "--gpu", "-g", help="Fit models on gpu.", action="store_true")
    parser.add_argument(
        "--tune", "-t", help="Amount of tuning steps.", default=2_000, type=int)
    parser.add_argument(
        "--target_accept", "-a", help="Target accept probability.", default=.95, type=float)
    parser.add_argument(
        "--demographics", "-d", help="Train all the demographics models.", action="store_true")
    parser.add_argument(
        "--fixed", "-f", help="Train models with fixed effects.", action="store_true")
    parser.add_argument(
        "--fixed-country", "-r", help="Train models with fixed effects per country (partially pooled).", action="store_true")
    parser.add_argument(
        "--filtering", help="Filtering to apply to the data.", type=str, default="interval")
    parser.add_argument(
        "--control", help="Include control variables.", action="store_true")
    parser.add_argument(
        "--combinations", help="Compute all model combinations.", action="store_true")
    parser.add_argument(
        "--complete", help="Complete case analysis (remove same as control).", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
