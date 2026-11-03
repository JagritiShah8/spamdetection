"""Command Line Interface for Spam SMS Detector.

Usage:
    python -m spamdetector train [--data-path PATH]
    python -m spamdetector predict "Your text message"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spamdetector.predict import predict
from spamdetector.train import train_and_evaluate


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spamdetector",
        description="Production-grade spam SMS detection without third-party ML frameworks.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models and generate reports")
    train_parser.add_argument(
        "--data-path",
        default="data/SMSSpamCollection",
        help="Path to SMSSpamCollection file",
    )
    train_parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Output directory for saved models and vectorizers",
    )
    train_parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Output directory for metrics and confusion matrix plots",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict class for an SMS message")
    predict_parser.add_argument(
        "message",
        nargs="?",
        default="",
        type=str,
        help="The SMS message text to evaluate",
    )
    predict_parser.add_argument(
        "--model",
        default="best_model",
        help="Model artifact name (e.g. best_model, logistic_regression)",
    )
    predict_parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory containing trained model artifacts",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "train":
        train_and_evaluate(
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            reports_dir=args.reports_dir,
        )
        return 0

    if args.command == "predict":
        result = predict(args.message, artifacts_dir=args.artifacts_dir, model_name=args.model)
        print("\n" + "=" * 50)
        print(f"Prediction Result: {result['label']}")
        print("=" * 50)
        print(f"Spam Probability: {result['spam_probability'] * 100:.2f}%")
        print(f"Confidence Level: {result['confidence'].upper()}")
        print("\nTop Contributing Words:")
        if result["top_words"]:
            for word, contrib in result["top_words"][:8]:
                direction = "Spam +" if contrib > 0 else "Ham  -"
                print(f"  {direction}  {word:<15} ({contrib:+.4f})")
        else:
            print("  (No vocabulary words recognized)")
        print("=" * 50 + "\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
