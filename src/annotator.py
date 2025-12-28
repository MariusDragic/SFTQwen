"""Teacher annotation pipeline for generating synthetic summaries.

This module implements a pipeline that uses a larger teacher model to generate
high-quality synthetic summaries for news articles, which are then used to
fine-tune a smaller student model.
"""

import shutil
import time
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

from .config import AnnotatorConfig
from .utils import clean_text, filter_summary, infer_compute_dtype


def build_prompt(cfg: AnnotatorConfig, tokenizer, article: str) -> str:
    """Build a chat prompt for article summarization.

    Args:
        cfg: Annotator configuration.
        tokenizer: Tokenizer for the model.
        article: Article text to summarize.

    Returns:
        Formatted chat prompt.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional news summarization system."
        },
        {
            "role": "user",
            "content": cfg.prompt_template.format(article=article)
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(
    docs: List[str],
    cfg: AnnotatorConfig,
    model,
    tokenizer,
) -> List[str]:
    """Generate summaries for a batch of documents.

    Args:
        docs: List of document texts.
        cfg: Annotator configuration.
        model: Model for generation.
        tokenizer: Tokenizer for the model.

    Returns:
        List of generated summaries.
    """
    prompts = [build_prompt(cfg, tokenizer, d) for d in docs]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.prompt_max_length,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )

    return [clean_text(d) for d in decoded]


def run_annotation(cfg: AnnotatorConfig) -> None:
    """Run the teacher annotation pipeline.

    Args:
        cfg: Annotator configuration.
    """
    dtype = infer_compute_dtype()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.teacher_model,
        max_seq_length=cfg.prompt_max_length + cfg.max_new_tokens,
        dtype=dtype,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    stream = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
    )

    selected = []
    for row in stream:
        txt = clean_text(row["article"])
        if cfg.min_chars <= len(txt) <= cfg.max_chars:
            selected.append({"id": str(len(selected)), "document": txt})
        if len(selected) >= cfg.n_docs_stream:
            break

    raw_docs = Dataset.from_list(selected)

    out_dir = Path(cfg.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    done_ids = set()
    cursor = 0
    batches_since_save = 0

    pbar = tqdm(total=cfg.n_docs_dataset, desc="Annotating", unit="doc")

    while len(rows) < cfg.n_docs_dataset and cursor < len(raw_docs):
        batch = raw_docs[cursor: cursor + cfg.batch_size]
        cursor += cfg.batch_size

        summaries = generate_batch(
            batch["document"], cfg, model, tokenizer
        )

        for ex_id, doc, summ in zip(batch["id"], batch["document"], summaries):
            if ex_id in done_ids or not filter_summary(summ):
                continue

            rows.append(
                {"id": ex_id, "document": doc, "summary": summ.strip()}
            )
            done_ids.add(ex_id)
            pbar.update(1)

            if len(rows) >= cfg.n_docs_dataset:
                break

        batches_since_save += 1

        if batches_since_save >= cfg.save_every_batches:
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            Dataset.from_list(rows).save_to_disk(str(out_dir))
            batches_since_save = 0

        torch.cuda.empty_cache()

    pbar.close()

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    Dataset.from_list(rows).save_to_disk(str(out_dir))
