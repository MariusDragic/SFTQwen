"""Utility functions for SFTQwen project.

This module contains utility functions for GPU management, text processing,
and data type inference.
"""

import gc
import re

import torch


EMAIL_FOOTER_PATTERN = re.compile(
    r"E[-–—]?mail\s+to\s+a\s+friend\s*\..*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def setup_gpu() -> None:
    """Set up GPU environment and clear cache.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()


def infer_compute_dtype() -> torch.dtype:
    """Infer the optimal compute dtype based on GPU capability.

    Returns:
        torch.dtype: torch.bfloat16 if GPU supports it, otherwise torch.float16.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text with normalized whitespace and removed footers.
    """
    if not text:
        return text

    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if "--" in text:
        text = text.split("--", 1)[1].strip()

    text = EMAIL_FOOTER_PATTERN.sub("", text)
    return text.strip()


def filter_summary(
    summary: str,
    min_words: int = 35,
    max_words: int = 100,
) -> bool:
    """Validate a summary based on word count.

    Args:
        summary: Summary text to validate.
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.

    Returns:
        True if summary word count is within bounds, False otherwise.
    """
    if not summary:
        return False

    words = re.findall(r"\b\w+\b", summary.strip())
    return min_words <= len(words) <= max_words
