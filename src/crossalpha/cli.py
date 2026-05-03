"""Command-line interface for CrossAlpha."""

from __future__ import annotations

import argparse
import logging

from crossalpha.config import load_config
from crossalpha.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrossAlpha pipeline CLI")
    parser.add_argument("command", choices=[
        "ingest",
        "features",
        "labels",
        "train",
        "validate",
        "recommend",
        "backtest",
        "run-all",
    ])
    parser.add_argument("--config", default="configs/base.toml", help="Path to the project config TOML.")
    parser.add_argument("--start-date", default=None, help="Optional override for date-range based commands.")
    parser.add_argument("--end-date", default=None, help="Optional override for date-range based commands.")
    parser.add_argument("--latest-only", action="store_true", help="Only emit the latest recommendation date.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg.logs_dir)

    if args.command == "ingest":
        from crossalpha.data.pipeline import run_data_ingestion

        run_data_ingestion(cfg)
        return 0

    if args.command == "features":
        from crossalpha.features.engine import build_feature_store

        build_feature_store(cfg)
        return 0

    if args.command == "labels":
        from crossalpha.labels.engine import build_label_store

        build_label_store(cfg)
        return 0

    if args.command == "train":
        from crossalpha.models.training import train_final_models

        train_final_models(cfg)
        return 0

    if args.command == "validate":
        from crossalpha.validation.walkforward import run_validation

        run_validation(cfg)
        return 0

    if args.command == "recommend":
        from crossalpha.engine.recommender import generate_recommendations

        generate_recommendations(
            cfg,
            start_date=args.start_date,
            end_date=args.end_date,
            latest_only=args.latest_only,
        )
        return 0

    if args.command == "backtest":
        from crossalpha.backtest.simulator import run_backtest

        run_backtest(cfg)
        return 0

    if args.command == "run-all":
        from crossalpha.data.pipeline import run_data_ingestion
        from crossalpha.features.engine import build_feature_store
        from crossalpha.labels.engine import build_label_store
        from crossalpha.validation.walkforward import run_validation
        from crossalpha.models.training import train_final_models
        from crossalpha.engine.recommender import generate_recommendations
        from crossalpha.backtest.simulator import run_backtest

        LOGGER.info("Running full CrossAlpha pipeline.")
        run_data_ingestion(cfg)
        build_feature_store(cfg)
        build_label_store(cfg)
        run_validation(cfg)
        train_final_models(cfg)
        generate_recommendations(cfg, start_date=args.start_date, end_date=args.end_date, latest_only=args.latest_only)
        run_backtest(cfg)
        return 0

    parser.print_help()
    return 1
