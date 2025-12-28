"""Evaluation module for summarization quality metrics.

This module implements comprehensive evaluation using ROUGE, METEOR, and BERTScore
metrics to assess the quality of generated summaries against reference summaries.
"""

import gc

import evaluate
import nltk
import torch
from bert_score import score
from datasets import Dataset
from tqdm import tqdm

from .config import Config
from .generate import generate_summaries


def run_evaluation(cfg: Config, test_dataset: Dataset, batch_size: int = 4) -> dict:
    """Run comprehensive evaluation on the test dataset.

    Args:
        cfg: Configuration object.
        test_dataset: Test dataset with 'document' and 'summary' fields.
        batch_size: Batch size for generation.

    Returns:
        Dictionary containing evaluation metrics.
    """
    # Download required NLTK data
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Load metrics
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    predictions = []
    references = []

    docs_buffer = []
    refs_buffer = []

    # Generate summaries in batches
    for ex in tqdm(test_dataset, desc="Evaluating on test set (batched)"):
        docs_buffer.append(ex["document"])
        refs_buffer.append(ex["summary"].strip())

        if len(docs_buffer) == batch_size:
            preds = generate_summaries(cfg, docs_buffer)
            predictions.extend(preds)
            references.extend(refs_buffer)

            docs_buffer = []
            refs_buffer = []

    # Process remaining documents
    if len(docs_buffer) > 0:
        preds = generate_summaries(cfg, docs_buffer)
        predictions.extend(preds)
        references.extend(refs_buffer)

    # Compute ROUGE scores
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
    )

    # Compute METEOR score
    meteor_score = meteor.compute(
        predictions=predictions,
        references=references,
    )

    # Prepare for BERTScore
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    torch.set_grad_enabled(False)

    # Compute BERTScore
    _, _, f1 = score(
        predictions,
        references,
        lang="en",
        model_type="roberta-base",
        device="cuda",
        rescale_with_baseline=True,
        verbose=True,
    )

    bert_f1 = f1.mean().item()

    # Compile results
    results = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "meteor": meteor_score["meteor"],
        "bertscore_f1": bert_f1,
    }

    return results
