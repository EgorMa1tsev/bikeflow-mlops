"""Command line entry points: python -m bikeflow <command>."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def _cmd_download(args: argparse.Namespace) -> int:
    from .data.download import download_raw

    download_raw(force=args.force)
    return 0


def _cmd_preprocess(_: argparse.Namespace) -> int:
    from .data.preprocess import preprocess

    preprocess()
    return 0


def _cmd_split(_: argparse.Namespace) -> int:
    from .data.split import run_split

    run_split()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .training.train import run_training

    run_training(save=not args.dry_run, figures=not args.no_figures)
    return 0


def _cmd_cv(_: argparse.Namespace) -> int:
    from .training.cv import run_cv, summarise_cv

    scores = run_cv()
    print()
    print(scores.round(1).to_string(index=False))
    print()
    print(summarise_cv(scores).round(1).to_string(index=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import pandas as pd

    from .data.split import load_splits
    from .features import TARGET
    from .metrics import evaluate, evaluate_by_slice, rain_flag
    from .models.registry import load_bundle

    bundle = load_bundle(args.model)
    frame = load_splits()[args.split]
    frame = frame[frame["is_functioning"]].reset_index(drop=True)
    frame["is_rain"] = rain_flag(frame)

    actual = frame[TARGET].to_numpy(dtype="float64")
    predicted = bundle["pipeline"].predict(frame)

    print(f"model: {bundle['kind']}  trained_at: {bundle['metadata']['trained_at']}")
    print(f"split: {args.split}  rows: {len(frame)}\n")
    print(pd.Series(evaluate(actual, predicted)).to_string())
    for column in ("season", "is_weekend", "is_rain"):
        print(f"\n-- by {column} --")
        print(evaluate_by_slice(frame, actual, predicted, column).to_string(index=False))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import numpy as np

    from .inference import Predictor, predict_file

    if args.json:
        payload = json.loads(args.json)
        predictor = Predictor.load(args.model)
        result = predictor.predict(payload, return_frame=True)
        for _, row in result.iterrows():
            print(
                f"{row['date'].date()} {int(row['hour']):02d}:00 -> "
                f"{row['predicted_demand']:.1f} аренд"
            )
        return 0

    if args.input:
        result = predict_file(args.input, args.output, args.model)
        if args.output is None:
            print(result[["date", "hour", "predicted_demand"]].to_string(index=False))
        else:
            values = result["predicted_demand"].to_numpy()
            print(f"mean {np.mean(values):.1f}  min {np.min(values):.1f}  max {np.max(values):.1f}")
        return 0

    print("Provide either --json or --input.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bikeflow", description="BikeFlow data/model pipeline (role A)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch the raw dataset from UCI")
    download.add_argument("--force", action="store_true", help="re-download even if present")
    download.set_defaults(func=_cmd_download)

    preprocess = subparsers.add_parser("preprocess", help="raw csv -> processed parquet")
    preprocess.set_defaults(func=_cmd_preprocess)

    split = subparsers.add_parser("split", help="temporal train/validation/test split")
    split.set_defaults(func=_cmd_split)

    train = subparsers.add_parser("train", help="train all models and write reports")
    train.add_argument("--dry-run", action="store_true", help="do not write artifacts")
    train.add_argument("--no-figures", action="store_true", help="skip plots")
    train.set_defaults(func=_cmd_train)

    cross_val = subparsers.add_parser(
        "cv", help="rolling-origin cross-validation used to pick the champion"
    )
    cross_val.set_defaults(func=_cmd_cv)

    evaluate = subparsers.add_parser("evaluate", help="score a saved model on a split")
    evaluate.add_argument("--model", default=None, help="path to a .joblib artifact")
    evaluate.add_argument("--split", default="test", choices=["train", "validation", "test"])
    evaluate.set_defaults(func=_cmd_evaluate)

    predict = subparsers.add_parser("predict", help="predict for one or many observations")
    predict.add_argument("--model", default=None, help="path to a .joblib artifact")
    predict.add_argument("--json", help="a single observation, or a list, as JSON")
    predict.add_argument("--input", help="csv/parquet file with observations")
    predict.add_argument("--output", help="where to write the scored csv")
    predict.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "model", None) is None and args.command in {"evaluate", "predict"}:
        args.model = load_config()["paths"]["production_model"]
    return args.func(args)
