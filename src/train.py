"""Training module for supervised fine-tuning.

This module handles the complete training pipeline including model loading,
LoRA configuration, dataset preparation, and training with Unsloth and TRL.
"""

from pprint import pprint

import torch
import wandb
from datasets import Dataset
from peft import LoftQConfig
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import (
    FastLanguageModel,
    get_chat_template,
    is_bfloat16_supported,
    unsloth_train,
)

from .config import Config
from .dataset import create_train_val_test_splits, tokenize_dataset
from .utils import setup_gpu


def run_training(cfg: Config) -> None:
    """Run the supervised fine-tuning training pipeline.

    Args:
        cfg: Configuration object containing all settings.
    """
    print("\n" + "=" * 80)
    print("TRAINING CONFIG")
    print("=" * 80)
    pprint(cfg.training.model_dump(), sort_dicts=False)
    print("=" * 80 + "\n")

    setup_gpu()
    use_bf16 = is_bfloat16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=cfg.model.chat_template,
    )

    im_end = cfg.model.eos_token
    im_end_id = tokenizer.convert_tokens_to_ids(im_end)

    if tokenizer.eos_token_id is None or tokenizer.eos_token_id != im_end_id:
        tokenizer.eos_token = im_end
        tokenizer.eos_token_id = im_end_id

    loftq_config = LoftQConfig(
        loftq_bits=cfg.lora.loftq_bits,
        loftq_iter=cfg.lora.loftq_iter,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
        use_rslora=cfg.lora.use_rslora,
        loftq_config=loftq_config,
    )

    raw = Dataset.from_file(cfg.model.dataset_path)
    splits = create_train_val_test_splits(
        raw,
        val_size=0.05,
        test_size=0.10,
        seed=cfg.training.seed,
    )

    train_dataset = tokenize_dataset(
        splits["train"],
        tokenizer,
        cfg.model.max_seq_length,
        cfg.model.eos_token,
        cfg.prompt.system_prompt,
        cfg.prompt.user_prompt,
    )
    val_dataset = tokenize_dataset(
        splits["val"],
        tokenizer,
        cfg.model.max_seq_length,
        cfg.model.eos_token,
        cfg.prompt.system_prompt,
        cfg.prompt.user_prompt,
    )

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        report_to=cfg.training.report_to,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=cfg.training.optim,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        seed=cfg.training.seed,
    )

    wandb.finish()
    wandb.init(
        name=cfg.training.output_dir.split("/")[-1],
        project=cfg.training.report_to,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=cfg.model.max_seq_length,
        data_collator=data_collator,
        packing=False,
    )

    trainer_stats = unsloth_train(trainer)

    trainer.model.save_pretrained(cfg.lora.lora_dir)
    tokenizer.save_pretrained(cfg.lora.lora_dir)

    print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"LoRA adapter saved to: {cfg.lora.lora_dir}")
