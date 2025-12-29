"""Main CLI entry point for SFTQwen project.

This module provides a command-line interface for running different modes:
- annotate: Generate synthetic summaries using a teacher model
- train: Fine-tune a student model on the annotated data
- eval: Evaluate the trained model on the test set
- generate: Generate a summary for a single article
"""

import argparse
import random
from pathlib import Path

from datasets import Dataset

from src.annotator import run_annotation
from src.config import Config
from src.dataset import create_train_val_test_splits
from src.eval import run_evaluation
from src.generate import generate_summary
from src.train import run_training


def main() -> None:
    """Main entry point for the SFTQwen pipeline."""
    parser = argparse.ArgumentParser(
        description="SFTQwen: Supervised Fine-Tuning for News Summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["annotate", "train", "eval", "generate"],
        required=True,
        help=(
            "Mode to run:\n"
            "  annotate - Generate synthetic summaries using teacher model\n"
            "  train    - Fine-tune student model on annotated data\n"
            "  eval     - Evaluate trained model on test set\n"
            "  generate - Generate a summary for a single article"
        ),
    )

    parser.add_argument(
        "--path_to_article",
        type=str,
        default=None,
        help=(
            "Path to a text file containing the article to summarize. "
            "If omitted, a random article is sampled from the dataset."
        ),
    )

    args = parser.parse_args()
    cfg = Config()

    if args.mode == "annotate":
        print("\n" + "=" * 80)
        print("Running annotation pipeline...")
        print("=" * 80 + "\n")
        run_annotation(cfg.annotator)
        print("\n" + "=" * 80)
        print("Annotation completed successfully!")
        print("=" * 80 + "\n")

    elif args.mode == "train":
        print("\n" + "=" * 80)
        print("Running training pipeline...")
        print("=" * 80 + "\n")
        run_training(cfg)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80 + "\n")

    elif args.mode == "eval":
        print("\n" + "=" * 80)
        print("Running evaluation pipeline...")
        print("=" * 80 + "\n")

        # Load dataset and get test split
        raw = Dataset.from_file(cfg.model.dataset_path)
        splits = create_train_val_test_splits(
            raw,
            val_size=0.05,
            test_size=0.10,
            seed=cfg.training.seed,
        )
        test_dataset = splits["test"]

        print(f"Test dataset size: {len(test_dataset)} examples\n")

        # Run evaluation
        metrics = run_evaluation(cfg, test_dataset)

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        for name, value in metrics.items():
            print(f"{name:20s}: {value:.4f}")
        print("=" * 80 + "\n")

    elif args.mode == "generate":
        print("\n" + "=" * 80)
        print("Running generation...")
        print("=" * 80 + "\n")

        if args.path_to_article is not None:
            article_path = Path(args.path_to_article)
            article = article_path.read_text(encoding="utf-8")
        else:
            raw = Dataset.from_file(cfg.model.dataset_path)
            example = raw[random.randrange(len(raw))]
            article = example["document"]

        summary = generate_summary(cfg, article)
        print(summary)


if __name__ == "__main__":
    main()
