"""Text generation module for inference and summary generation.

This module provides functionality for loading trained models and generating
summaries from documents using the fine-tuned model with custom stopping criteria.
"""

import re
import textwrap
from typing import List

import torch
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel, get_chat_template

from .config import Config


class StopAfterNSentences(StoppingCriteria):
    """Custom stopping criterion that halts generation after N sentences.

    This prevents the model from generating overly long outputs or entering
    repetitive loops.
    """

    def __init__(self, tokenizer, n_sentences: int = 3):
        """Initialize the stopping criterion.

        Args:
            tokenizer: Tokenizer used for decoding.
            n_sentences: Maximum number of sentences to generate.
        """
        self.tokenizer = tokenizer
        self.n_sentences = n_sentences

        self.sentence_regex = re.compile(
            r"(?<!\b[A-Z])([.!?])(?=\s|$)"
        )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Check whether generation should stop.

        Args:
            input_ids: Generated token IDs.
            scores: Model scores (unused).

        Returns:
            True if generation should stop, False otherwise.
        """
        decoded = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=False
        )

        if "<|im_start|>assistant\n" in decoded:
            decoded = decoded.split("<|im_start|>assistant\n", 1)[1]

        sentence_count = len(self.sentence_regex.findall(decoded))
        return sentence_count >= self.n_sentences


def load_model_for_inference(
    base_model_name_or_path: str,
    lora_path: str | None = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a base model with optional LoRA adapters for inference.

    Args:
        base_model_name_or_path: Base model name or path.
        lora_path: Path to LoRA adapters, if any.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to load the model in 4-bit precision.

    Returns:
        Tuple of (model, tokenizer).
    """
    torch.set_grad_enabled(False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        torch_dtype=None,
        device_map="auto",
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=False,
        )

    FastLanguageModel.for_inference(model)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_summaries(cfg: Config, documents: List[str]) -> List[str]:
    """Generate summaries for a batch of documents.

    Args:
        cfg: Configuration object.
        documents: List of document texts to summarize.

    Returns:
        List of generated summaries.
    """

    model, tokenizer = load_model_for_inference(
        base_model_name_or_path=cfg.model.base_model,
        lora_path=cfg.lora.lora_dir,
        max_seq_length=cfg.generation.max_input_length,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=cfg.model.chat_template,
    )

    batch_messages = [
        [
            {"role": "system", "content": cfg.prompt.system_prompt},
            {"role": "user", "content": (
                cfg.prompt.user_prompt +
                f"ARTICLE:\n{doc}"
            )},
        ]
        for doc in documents
    ]

    input_ids = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.generation.max_input_length,
    ).to("cuda")

    stopping_criteria = StoppingCriteriaList([
        StopAfterNSentences(
            tokenizer,
            n_sentences=cfg.generation.n_sentences,
        ),
    ])

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=cfg.generation.max_new_tokens,
            do_sample=cfg.generation.do_sample,
            repetition_penalty=cfg.generation.repetition_penalty,
            no_repeat_ngram_size=cfg.generation.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    preds = []
    for out in outputs:
        decoded = tokenizer.decode(out, skip_special_tokens=False)
        decoded = decoded.split("<|im_start|>assistant\n", 1)[-1]
        decoded = decoded.split(cfg.model.eos_token, 1)[0]
        preds.append(decoded.strip())

    return preds


def generate_summary(cfg: Config, article: str) -> str:
    """Generate a single summary for one article.

    Args:
        cfg: Configuration object.
        article: Article text to summarize.

    Returns:
        Generated summary.
    """
    return generate_summaries(cfg, [article])[0]


def format_generation_output(*, article: str, summary: str, width: int = 100) -> str:
    """Format a generation result for terminal display.

    This is presentation-only (no change to generation logic).

    Args:
        article: Input article text.
        summary: Generated summary.
        width: Line wrap width.

    Returns:
        Formatted string.
    """
    sep = "=" * 100
    article_preview = textwrap.fill(article.strip(), width=width)
    summary_wrapped = textwrap.fill(summary.strip(), width=width)

    return (
        f"{sep}\n"
        f"ARTICLE\n"
        f"{sep}\n"
        f"{article_preview}\n"
        f"{sep}\n"
        f"SUMMARY\n"
        f"{sep}\n"
        f"{summary_wrapped}\n"
        f"{sep}"
    )
