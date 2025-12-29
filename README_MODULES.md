# SFTQwen: Supervised Fine-Tuning for News Summarization

A modular Python project for fine-tuning small language models on news summarization using high quality annotated dataset by a Qwen2.5-7B.

## Project Structure

```
SFTQwen/
├── main.py                 # CLI entry point
├── src/
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration classes (Pydantic)
│   ├── utils.py           # Utility functions (GPU, text processing)
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── annotator.py       # Teacher annotation pipeline
│   ├── train.py           # Student model training
│   ├── generate.py        # Inference and generation
│   └── eval.py            # Evaluation metrics (ROUGE, METEOR, BERTScore)
├── data/                  # Dataset storage
├── model/                 # Trained LoRA adapters
├── outputs/               # Training checkpoints
└── notebooks/             # Original Jupyter notebooks (archived)
```

## Modules

### `config.py`
Centralized configuration using Pydantic models:
- `AnnotatorConfig`: Teacher annotation settings
- `ModelConfig`: Base model configuration
- `LoRAConfig`: LoRA fine-tuning parameters
- `TrainingConfig`: Training hyperparameters
- `GenerationConfig`: Inference settings
- `PromptConfig`: System and user prompts
- `Config`: Main configuration combining all sub-configs

### `utils.py`
Utility functions:
- `setup_gpu()`: Initialize GPU environment
- `infer_compute_dtype()`: Determine optimal dtype based on GPU capability
- `clean_text()`: Text normalization and cleaning
- `filter_summary()`: Validate summary quality by word count

### `dataset.py`
Dataset handling:
- `process_example()`: Tokenize single example with prompt masking
- `tokenize_dataset()`: Batch tokenization with multiprocessing
- `create_train_val_test_splits()`: Split data into train/val/test (80/10/10)

### `annotator.py`
Teacher annotation pipeline:
- `build_prompt()`: Format articles into chat templates
- `generate_batch()`: Batched inference with teacher model
- `run_annotation()`: Complete annotation pipeline

### `train.py`
Student model training:
- Model initialization with Unsloth
- LoRA adapter configuration with LoftQ
- Dataset tokenization and splitting
- Training with SFTTrainer and Unsloth optimization
- Model saving

### `generate.py`
Inference and generation:
- `StopAfterNSentences`: Custom stopping criterion
- `load_model_for_inference()`: Load model with optional LoRA adapters
- `generate_summaries()`: Batched summary generation

### `eval.py`
Comprehensive evaluation:
- ROUGE (n-gram overlap)
- METEOR (semantic similarity with synonyms)
- BERTScore (contextual embedding similarity)
- Batched processing for efficiency

## Configuration

All configuration is centralized in `src/config.py`. Key settings:

```python
# Example: Modify training epochs
cfg = Config()
cfg.training.num_train_epochs = 3

# Example: Change LoRA rank
cfg.lora.r = 64
cfg.lora.lora_alpha = 64
```

## Requirements

See the original notebooks for the complete list of dependencies:
- `torch`, `transformers`, `datasets`
- `unsloth` (optimized training)
- `peft` (LoRA implementation)
- `trl` (SFTTrainer)
- `wandb` (experiment tracking)
- `evaluate`, `rouge-score`, `bert-score`, `nltk`
- `pydantic` (configuration)

## Author

Marius Dragic
