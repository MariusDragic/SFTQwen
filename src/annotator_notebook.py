# %% [markdown]
# # High-Quality Teacher Annotation Pipeline for News Summarization

# %% [markdown]
# ## Installations and dependancies

# %%
!pip install \
  "datasets>=3.4.1,<4.4.0" \
  "trl>=0.18.2,<=0.24.0" \
  "transformers>=4.44.0" \
  "accelerate>=0.33.0" \
  "peft>=0.12.0" \
  "bitsandbytes>=0.43.0" \
  unsloth \
  wandb \
  evaluate \
  rouge-score \
  bert-score \
  nltk \
  tqdm 


# %% [markdown]
# ## Imports

# %% [markdown]
# To optimize GPU utilization and speed up training, we import Unsloth, a library designed for efficient fine-tuning of large language models.

# %%
import os
import re
import gc
import time
from pathlib import Path
from typing import Literal, List
from pprint import pprint
import shutil

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import BaseModel, Field
from tqdm import tqdm


# %% [markdown]
# ## GPU Requirements

# %% [markdown]
# We verifie that the allocated GPU corresponds to a “small” GPU (with less than 16 GB of VRAM), in line with the project requirement to run on a lightweight Google Colab GPU. 

# %%
assert torch.cuda.is_available(), "Non detected GPU."

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.synchronize()
torch.set_grad_enabled(False)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

use_bf16 = is_bfloat16_supported()
print("bf16 supported:", use_bf16)

!nvidia-smi

# %% [markdown]
# ## Configuration

# %% [markdown]
# Centralized configuration using Pydantic to make the code modular, readable, and reproducible.
# This structure allows easy adaptation of the model, training, generation, and prompts without changing core logic.

# %%
class AnnotatorConfig(BaseModel):
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"

    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    dataset_split: str = "train"

    n_docs_stream: int = 9000
    n_docs_dataset: int = 5000
    min_chars: int = 800
    max_chars: int = 1700

    prompt_max_length: int = 1024
    max_new_tokens: int = 160

    batch_size: int = 10
    save_every_batches: int = 1000

    output_dir: str = "data/cnn_dataset.arrow"
    resume: bool = True

    prompt_template: str = (
        "You are an expert news summarization assistant. "
        "Summarize the following news article in at most 3 sentences.\n\n"
        "Constraints:\n"
        "- Maximum 3 sentences.\n"
        "- Use only information explicitly stated in the article.\n"
        "- Stop immediately after the last sentence.\n\n"
        "ARTICLE:\n{article}"
    )


cfg = AnnotatorConfig()
print("\n===== ANNOTATOR CONFIG =====")
pprint(cfg.model_dump())
print("============================\n")


# %% [markdown]
# ## Utils

# %%
EMAIL_FOOTER_PATTERN = re.compile(
    r"E[-–—]?mail\s+to\s+a\s+friend\s*\..*$",
    flags=re.IGNORECASE | re.DOTALL
)

def clean_text(x: str) -> str:
    """
    Clean and normalize text.

    Args:
        x (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not x:
        return x

    x = x.replace("\n", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()

    # 2. Remove everything before the first double dash `--`
    if "--" in x:
        x = x.split("--", 1)[1].strip()

    # 3. Remove "E-mail to a friend" AND everything after it
    x = EMAIL_FOOTER_PATTERN.sub("", x)

    return x.strip()

def build_prompt(cfg: AnnotatorConfig, article: str) -> str:
    """
    Build a chat prompt for article summarization.

    Args:
        cfg: Configuration containing the prompt template.
        article (str): Article text to summarize.

    Returns:
        str: Formatted chat prompt.
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

def infer_compute_dtype() -> torch.dtype:
    """
    Infer the optimal compute dtype based on CUDA availability and GPU capability.

    Returns:
        torch.dtype: Recommended compute dtype.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16

def filter_summary(summary: str, min_words: int = 35, max_words: int = 100) -> bool:
    """
    Validate a summary based on word count.

    Args:
        summary (str): Summary text.
        min_words (int): Minimum allowed word count.
        max_words (int): Maximum allowed word count.

    Returns:
        bool: True if the summary is valid.
    """
    if summary is None:
        return False

    summary = summary.strip()
    if not summary:
        return False

    words = re.findall(r"\b\w+\b", summary)
    n_words = len(words)

    if n_words < min_words:
        return False
    if n_words > max_words:
        return False

    return True


# %% [markdown]
# ## Load Model

# %%
compute_dtype = infer_compute_dtype()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = cfg.teacher_model,
    max_seq_length = cfg.prompt_max_length + cfg.max_new_tokens,
    dtype = compute_dtype,
    load_in_4bit = True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

print("Teacher model loaded.")
print("Compute dtype:", compute_dtype)
print("eos_token:", tokenizer.eos_token, "| eos_token_id:", tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token, "| pad_token_id:", tokenizer.pad_token_id)

# %% [markdown]
# ## CNN Daily Dataset Loading & Filtering

# %% [markdown]
# We only keep articles with lengths between 800 and 1700 characters to slightly simplify the task.

# %%
stream = load_dataset(
    cfg.dataset_name,
    cfg.dataset_config,
    split=cfg.dataset_split,
    streaming=True,
)

selected = []
for row in tqdm(stream, desc="processing"):
    txt = clean_text(row["article"])
    if cfg.min_chars <= len(txt) <= cfg.max_chars:
        selected.append({
            "id": str(row.get("id", len(selected))),
            "document": txt,
        })
    if len(selected) >= cfg.n_docs_stream:
        break

raw_docs = Dataset.from_list(selected)

print("Selected documents:", len(raw_docs))
print("Document example:", raw_docs[0]) 

# %% [markdown]
# ## High-Quality Summary Generation and Filtering Loop

# %%
@torch.inference_mode()
def generate_batch(docs: List[str], cfg: AnnotatorConfig, tokenizer, model):
    """
    Generate summaries for a batch of documents.

    Args:
        docs (List[str]): Input documents.
        cfg (AnnotatorConfig): Generation configuration.
        tokenizer (PreTrainedTokenizerBase): Tokenizer.
        model (PreTrainedModel): Language model.

    Returns:
        List[str]: Generated and cleaned summaries.
    """

    prompts = [build_prompt(cfg, doc) for doc in docs]

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

    generated = outputs[:, prompt_len:]

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    decoded = [clean_text(d) for d in decoded]

    return decoded

# %% [markdown]
# This loop generates summaries in batches and applies strict quality filtering, as very short summaries (under 35 words) or overly long ones (over 100 words) are typically low-quality labels.

# %%
out_dir = Path(cfg.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
done_ids = set()

cursor = 0
t0 = time.time()
batches_since_save = 0

pbar = tqdm(
    total=cfg.n_docs_dataset,
    desc="Annotating",
    unit="doc",
)

while len(rows) < cfg.n_docs_dataset and cursor < cfg.n_docs_stream:
    batch = raw_docs[cursor : cursor + cfg.batch_size]
    cursor += cfg.batch_size

    docs = batch["document"]
    ids  = batch["id"]

    summaries = generate_batch(
        docs,
        cfg=cfg,
        tokenizer=tokenizer,
        model=model,
    )

    new_added = 0

    for ex_id, doc, summ in zip(ids, docs, summaries):
        if ex_id in done_ids:
            continue

        summ = summ.strip()

        if not filter_summary(summ):
            continue

        rows.append({
            "id": ex_id,
            "document": doc,
            "summary": summ,
        })
        done_ids.add(ex_id)
        new_added += 1
        pbar.update(1)

        if len(rows) >= cfg.n_docs_dataset:
            break

    batches_since_save += 1

    elapsed = (time.time() - t0) / 60
    pbar.set_postfix(
        batch_size=cfg.batch_size,
        added=new_added,
        total=len(rows),
        time=f"{elapsed:.1f}m",
    )

    if batches_since_save >= cfg.save_every_batches:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        Dataset.from_list(rows).save_to_disk(out_dir)
        batches_since_save = 0

    torch.cuda.empty_cache()

pbar.close()

if out_dir.exists():
    shutil.rmtree(out_dir)
Dataset.from_list(rows).save_to_disk(out_dir)

print(f"Dataset saved : {len(rows)} documents")


