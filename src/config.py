"""Configuration module for SFTQwen project.

This module contains all configuration classes used across the project,
including settings for annotation, model, LoRA, training, generation, and prompts.
"""

from typing import Optional

from pydantic import BaseModel


class AnnotatorConfig(BaseModel):
    """Configuration for the teacher annotation pipeline."""

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


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    base_model: str = "unsloth/Qwen2.5-1.5B"
    dataset_path: str = "data/cnn_dataset.arrow"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    chat_template: str = "qwen-2.5"
    eos_token: str = "<|im_end|>"


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

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
    """Configuration for model training."""

    output_dir: str = "./outputs"
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

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    n_sentences: int = 3
    max_new_tokens: int = 160
    max_input_length: int = 2048

    do_sample: bool = False
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    system_prompt: str = (
        "You are a professional news summarization assistant."
    )

    user_prompt: str = (
        "Summarize the following news article in at most 3 sentences. "
        "Rewrite the information concisely in your own words. "
        "Focus on the main events and key facts. "
    )


class Config(BaseModel):
    """Main configuration class combining all sub-configurations."""

    annotator: AnnotatorConfig = AnnotatorConfig()
    model: ModelConfig = ModelConfig()
    lora: LoRAConfig = LoRAConfig()
    training: TrainingConfig = TrainingConfig()
    generation: GenerationConfig = GenerationConfig()
    prompt: PromptConfig = PromptConfig()
