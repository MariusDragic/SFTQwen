# %% [markdown]
# # SFTQwen: Supervised Fine-Tuning of a Small Language Model for Document Summarization

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

import os
os.environ["WANDB_PROJECT"] = "qwen2.5-summarization"


# %% [markdown]
# ## Imports 

# %% [markdown]
# To optimize GPU utilization and speed up training, we import Unsloth, a library designed for efficient fine-tuning of large language models.

# %%
import random
import re
import gc
from textwrap import fill
from typing import List, Optional, Any
from pprint import pprint

from unsloth import FastLanguageModel, is_bfloat16_supported, get_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import nltk
import numpy as np
from datasets import Dataset
from peft import LoftQConfig
from trl import SFTTrainer
from peft import PeftModel
import wandb
import evaluate
from tqdm import tqdm
from pydantic import BaseModel, Field
from bert_score import score

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

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

use_bf16 = is_bfloat16_supported()
print("bf16 supported:", use_bf16)

!nvidia-smi

# %% [markdown]
# ## Configurations

# %% [markdown]
# Centralized configuration using Pydantic to make the code modular, readable, and reproducible.
# This structure allows easy adaptation of the model, training, generation, and prompts without changing core logic.

# %%
class ModelConfig(BaseModel):
    base_model: str = "unsloth/Qwen2.5-1.5B"
    dataset_path: str = "cnn_dataset.arrow"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    chat_template: str = "qwen-2.5"
    eos_token: str = "<|im_end|>"

class LoRAConfig(BaseModel):
    lora_dir: str = "./model"
    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.20
    bias: str = "none"

    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42
    use_rslora: bool = False

    loftq_bits: int = 4
    loftq_iter: int = 1

class TrainingConfig(BaseModel):
    output_dir: str = "/content/qwen_summarizer_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    logging_strategy: str = "steps"
    logging_steps: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 10

    save_steps: int = 100
    save_total_limit: int = 3

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    optim: str = "adamw_8bit"
    report_to: str = "wandb"

class GenerationConfig(BaseModel):
    n_sentences: int = 3
    max_new_tokens: int = 160
    max_input_length: int = 2048

    do_sample: bool = False
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

class PromptConfig(BaseModel):
    system_prompt: str = (
        "You are a professional news summarization assistant. "
    )

    user_prompt: str = (
        "Summarize the following news article in at most 3 sentences. "
        "Rewrite the information concisely in your own words. "
        "Focus on the main events and key facts. "
    
    )

class Configuration(BaseModel):
    model: ModelConfig = ModelConfig()
    lora: LoRAConfig = LoRAConfig()
    training: TrainingConfig = TrainingConfig()
    generation: GenerationConfig = GenerationConfig()
    prompt: PromptConfig = PromptConfig()

config = Configuration()

# %% [markdown]
# ## Model Initialization and Tokenizer Configuration

# %% [markdown]
# Ensure the EOS token matches the Qwen chat message terminator (<|im_end|>) to guarantee correct sequence termination.

# %%
model_cfg = config.model

print("\n===== Model CONFIG =====")
pprint(model_cfg.model_dump())
print("=========================\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = model_cfg.base_model,
    max_seq_length = model_cfg.max_seq_length,
    dtype          = model_cfg.dtype,
    load_in_4bit   = model_cfg.load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=model_cfg.chat_template,
)

IM_END = model_cfg.eos_token
im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

if tokenizer.eos_token_id is None or tokenizer.eos_token_id != im_end_id:
    tokenizer.eos_token = IM_END
    tokenizer.eos_token_id = im_end_id

print(
    "eos_token:", tokenizer.eos_token,
    "| eos_token_id:", tokenizer.eos_token_id
)

# %%
lora_cfg = config.lora

print("\n===== LoRA CONFIG =====")
pprint(lora_cfg.model_dump())
print("=========================\n")

loftq_config = LoftQConfig(
    loftq_bits=lora_cfg.loftq_bits,
    loftq_iter=lora_cfg.loftq_iter,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_cfg.r,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_cfg.lora_alpha,
    lora_dropout=lora_cfg.lora_dropout,
    bias=lora_cfg.bias,
    use_gradient_checkpointing=lora_cfg.use_gradient_checkpointing,
    random_state=lora_cfg.random_state,
    use_rslora=lora_cfg.use_rslora,
    loftq_config=loftq_config,
)

model.print_trainable_parameters()

# %% [markdown]
# This block applies parameter-efficient fine-tuning (PEFT) using LoRA on top of the frozen base model. LoRA adapters are injected into the main attention and MLP projection layers, drastically reducing the number of trainable parameters. LoftQ is enabled to better initialize LoRA weights when training on a 4-bit quantized model, improving stability and convergence while keeping GPU memory usage low.

# %% [markdown]
# ## Supervised Fine-Tuning Dataset Preprocessing

# %% [markdown]
# This cell builds the supervised fine-tuning dataset in chat format. The prompt (system + user) is masked so that the loss is computed only on the assistant’s summary. Sequences are tokenized, truncated to a fixed length, and split into train, validation, and test sets.
# 
# You must run Annotator-notebook.ipynb before to generate the synthetic summaries used for supervised fine-tuning.

# %%
prompt_cfg = config.prompt

print("\n===== PROMPT CONFIG =====")
pprint(prompt_cfg.model_dump())
print("===========================\n")

MAX_SEQ_LENGTH = config.model.max_seq_length
EOS_TOKEN = config.model.eos_token
SYSTEM_PROMPT = config.prompt.system_prompt
USER_PROMPT = config.prompt.user_prompt

def process_func(example: dict[str, str]) -> dict[str, list[int]]:
    """
    Tokenize a single training example into model inputs and labels.

    Args:
        example (dict[str, str]): Dataset example containing document and summary.

    Returns:
        dict[str, list[int]]: Tokenized inputs with labels.
    """
    ignore_index = -100
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            USER_PROMPT +
            f"ARTICLE:\n{example['document']}"
        )},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    answer_text = example["summary"].rstrip() + "\n" + EOS_TOKEN

    prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
    answer_tok = tokenizer(answer_text, add_special_tokens=False)

    input_ids = prompt_tok["input_ids"] + answer_tok["input_ids"]
    attention_mask = prompt_tok["attention_mask"] + answer_tok["attention_mask"]

    labels = [ignore_index] * len(prompt_tok["input_ids"]) + answer_tok["input_ids"]

    input_ids = input_ids[:MAX_SEQ_LENGTH]
    attention_mask = attention_mask[:MAX_SEQ_LENGTH]
    labels = labels[:MAX_SEQ_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize(ds: Dataset) -> Dataset:
    """
    Tokenize a dataset for supervised fine-tuning.

    Args:
        ds (Dataset): Input dataset.

    Returns:
        Dataset: Tokenized dataset.
    """
    return ds.map(
        process_func,
        remove_columns=ds.column_names,
        num_proc=2,
    )


raw = Dataset.from_file(model_cfg.dataset_path)

raw_split = raw.train_test_split(test_size=0.10, seed=42)
raw_train_val = raw_split["train"]
raw_test = raw_split["test"]

raw_train_val = raw_train_val.train_test_split(test_size=0.05, seed=42)
raw_train = raw_train_val["train"]
raw_val = raw_train_val["test"]

train_dataset = tokenize(raw_train)
val_dataset   = tokenize(raw_val)
test_dataset  = tokenize(raw_test)

print("Example of input ids: ", train_dataset[0]['input_ids'][:10])
print("Example of attention mask: ", train_dataset[0]['attention_mask'][:10])
print("Example of label ids: ", train_dataset[0]['labels'][-100:])

# %% [markdown]
# ## Supervised Fine-Tuning Configuration and Training

# %% [markdown]
# Training configuration : the setup is optimized for small GPUs through gradient accumulation, 8-bit optimization, and mixed-precision training, while monitoring progress with Weights & Biases. Evaluation and checkpointing are performed regularly to track convergence and prevent overfitting.

# %%
train_cfg = config.training

print("\n===== TRAINING CONFIG =====")
pprint(train_cfg.model_dump())
print("==============================\n")

wandb.finish()
wandb.init(
    name=train_cfg.output_dir.split("/")[-1], 
)

training_args = TrainingArguments(
    output_dir=train_cfg.output_dir,

    num_train_epochs=train_cfg.num_train_epochs,
    per_device_train_batch_size=train_cfg.per_device_train_batch_size,
    per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

    learning_rate=train_cfg.learning_rate,
    lr_scheduler_type=train_cfg.lr_scheduler_type,
    warmup_ratio=train_cfg.warmup_ratio,

    logging_strategy="steps",
    logging_steps=train_cfg.logging_steps,

    eval_strategy="steps",
    eval_steps=train_cfg.eval_steps,

    save_strategy="steps",
    save_steps=train_cfg.save_steps,
    save_total_limit=train_cfg.save_total_limit,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to=train_cfg.report_to,

    bf16=use_bf16,
    fp16=not use_bf16,

    optim=train_cfg.optim,
    weight_decay=train_cfg.weight_decay,
    max_grad_norm=train_cfg.max_grad_norm,
    seed=train_cfg.seed,
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=model_cfg.max_seq_length,
    data_collator=data_collator,
    packing=False,
)

# %%
from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer)

print(trainer_stats)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory during training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")

# %%
lora_cfg = config.lora

trainer.model.save_pretrained(lora_cfg.lora_dir)
tokenizer.save_pretrained(lora_cfg.lora_dir)

print("LoRA adapter saved to:", lora_cfg.lora_dir)

# %% [markdown]
# ## Inference for summary generation 

# %%
generation_cfg = config.generation

print("\n===== PROMPT CONFIG =====")
pprint(generation_cfg.model_dump())
print("===========================\n")

def load_model_for_inference(
    base_model_name_or_path: str,
    lora_path: str | None = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """
    Load a base model with optional LoRA adapters for optimized inference.

    Args:
        base_model_name_or_path (str): Base model name or path.
        lora_path (str | None): Path to LoRA adapters, if any.
        max_seq_length (int): Maximum sequence length.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.

    Returns:
        tuple: Loaded model and tokenizer.
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


model, tokenizer = load_model_for_inference(
    base_model_name_or_path=model_cfg.base_model,
    lora_path=lora_cfg.lora_dir,
    max_seq_length=generation_cfg.max_input_length,
)


# %% [markdown]
# This custom stopping criterion is used because the model sometimes loops or fails to emit the <|im_end|> token during generation. To ensure concise outputs and prevent runaway generations, decoding is explicitly stopped once three sentences have been produced.

# %%
class StopAfterNSentences(StoppingCriteria):
    """
    Stop text generation after a fixed number of sentences.
    """

    def __init__(self, tokenizer, n_sentences: int = 3):
        """
        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer used for decoding.
            n_sentences (int): Maximum number of sentences to generate.
        """
        self.tokenizer = tokenizer
        self.n_sentences = n_sentences

        self.sentence_regex = re.compile(
            r"(?<!\b[A-Z])([.!?])(?=\s|$)"
        )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        Check whether generation should stop.

        Args:
            input_ids (torch.Tensor): Generated token IDs.
            scores (torch.Tensor): Model scores (unused).

        Returns:
            bool: True if generation should stop.
        """
        decoded = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=False
        )

        if "<|im_start|>assistant\n" in decoded:
            decoded = decoded.split("<|im_start|>assistant\n", 1)[1]

        sentence_count = len(self.sentence_regex.findall(decoded))
        return sentence_count >= self.n_sentences
        

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

# %% [markdown]
# Function performs batched inference by formatting inputs in the Qwen chat style and generating summaries in evaluation mode without gradient computation. Generation is constrained using decoding penalties and a custom stopping criterion to ensure concise, non-repetitive summaries limited to a fixed number of sentences.

# %%
def generate_summary(documents):
    """
    Generate summaries for a batch of documents using config-driven generation
    and prompt settings.
    """

    batch_messages = [
        [
            {"role": "system", "content": prompt_cfg.system_prompt},
            {"role": "user", "content": (
                prompt_cfg.user_prompt +
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
        max_length=generation_cfg.max_input_length,
    ).to("cuda")

    stopping_criteria = StoppingCriteriaList([
        StopAfterNSentences(
            tokenizer,
            n_sentences=generation_cfg.n_sentences,
        ),
    ])

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=generation_cfg.max_new_tokens,
            do_sample=generation_cfg.do_sample,
            repetition_penalty=generation_cfg.repetition_penalty,
            no_repeat_ngram_size=generation_cfg.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    preds = []
    for out in outputs:
        decoded = tokenizer.decode(out, skip_special_tokens=False)
        decoded = decoded.split("<|im_start|>assistant\n", 1)[-1]
        decoded = decoded.split(model_cfg.eos_token, 1)[0]
        preds.append(decoded.strip())

    return preds

# %% [markdown]
# ## Random summary example

# %%
ex = random.choice(raw_test)

article = ex["document"]
gold = ex["summary"]

pred = generate_summary([article])[0]

print("="*110)
print("ARTICLE:\n", fill(article, 110))
print("="*110)
print("GOLD SUMMARY:\n", fill(gold, 110))
print("="*110)
print("MODEL SUMMARY:\n", fill(pred, 110))
print("="*110)

# %% [markdown]
# ## Evaluation Strategy

# %% [markdown]
# We evaluate the summarization quality using standard, widely adopted metrics. ROUGE measures n-gram overlap, METEOR accounts for synonymy and linguistic variation, and BERTScore captures semantic similarity using contextual embeddings, providing a balanced assessment of lexical and semantic fidelity.

# %%
nltk.download("wordnet")
nltk.download("omw-1.4")

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")


# %%
BATCH_SIZE = 4

predictions = []
references = []

docs_buffer = []
refs_buffer = []

for ex in tqdm(raw_test, desc="Evaluating on val (batched)"):
    docs_buffer.append(ex["document"])
    refs_buffer.append(ex["summary"].strip())

    if len(docs_buffer) == BATCH_SIZE:
        preds = generate_summary(docs_buffer)
        predictions.extend(preds)
        references.extend(refs_buffer)

        docs_buffer = []
        refs_buffer = []

if len(docs_buffer) > 0:
    preds = generate_summary(docs_buffer)
    predictions.extend(preds)
    references.extend(refs_buffer)


# %%
rouge_scores = rouge.compute(
    predictions=predictions,
    references=references,
)

print("=== ROUGE ===")
for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
    print(f"{k}: {rouge_scores[k]:.4f}")


# %%
meteor_score = meteor.compute(
    predictions=predictions,
    references=references,
)

print("\n=== METEOR ===")
print(f"meteor: {meteor_score['meteor']:.4f}")

# %%
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

torch.cuda.synchronize()
torch.set_grad_enabled(False)

P, R, F1 = score(
    predictions,
    references,
    lang="en",
    model_type="roberta-base",   
    device="cuda",           
    rescale_with_baseline=True,
    verbose=True,
)

bert_p = P.mean().item()
bert_r = R.mean().item()
bert_f1 = F1.mean().item()

print("\n=== BERTScore (roberta-base, GPU) ===")
print(f"Precision: {bert_p:.4f}")
print(f"Recall:    {bert_r:.4f}")
print(f"F1:        {bert_f1:.4f}")


# %%
results = {
    "rouge1": rouge_scores["rouge1"],
    "rouge2": rouge_scores["rouge2"],
    "rougeL": rouge_scores["rougeL"],
    "meteor": meteor_score["meteor"],
    "bertscore_f1": bert_f1,
}

print("\n=== FINAL VALIDATION RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")


# %% [markdown]
# ## Conclusion

# %% [markdown]
# The fine-tuned Qwen2.5-1.5B model achieves reasonable summarization performance, with a ROUGE-1 score of 0.36 indicating acceptable content coverage under tight length constraints. However, the relatively low ROUGE-2 (0.12) suggests limited modeling of longer phrasal dependencies, which is expected for a small model trained on noisy, automatically generated labels. The METEOR score of 0.29 shows some paraphrasing ability, but also highlights remaining lexical and stylistic inconsistencies. The BERTScore F1 of 0.30 reflects moderate semantic alignment, constrained by both model capacity and teacher-label imperfections. Overall, the results confirm the feasibility of the teacher–student approach, while leaving clear room for improvement through cleaner supervision and stronger distillation strategies.


