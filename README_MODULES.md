# SFTQwen: Supervised Fine-Tuning for News Summarization

A modular Python project for fine-tuning small language models on news summarization using teacher-student distillation.

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

## Usage

### 1. Annotation (Generate Training Data)
```bash
python main.py --mode annotate
```
Uses a 7B teacher model to generate synthetic summaries from CNN/DailyMail articles.

### 2. Training (Fine-tune Student Model)
```bash
python main.py --mode train
```
Fine-tunes a 1.5B student model using LoRA on the annotated data.

### 3. Evaluation (Test Set Metrics)
```bash
python main.py --mode eval
```
Evaluates the trained model on the held-out test set.

## Configuration

All configuration is centralized in `src/config.py`. Key settings:

```python
# Example: Modify training epochs
cfg = Config()
cfg.training.num_train_epochs = 5

# Example: Change LoRA rank
cfg.lora.r = 128
cfg.lora.lora_alpha = 128
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

## Design Principles

1. **Modularity**: Each module has a single, well-defined responsibility
2. **Type Safety**: Pydantic for configuration, type hints throughout
3. **PEP 8 Compliance**: Consistent style, docstrings, naming conventions
4. **Exact Logic Preservation**: All notebook logic maintained without modification
5. **Extensibility**: Easy to add new models, metrics, or datasets

## Notes

- The train/val/test split is 80/10/10 by default
- Validation set monitors overfitting during training
- Test set is completely unseen until evaluation
- LoRA adapters are saved separately from the base model
- All randomness is seeded for reproducibility

## Author

Refactored from Jupyter notebooks to production-quality Python modules.
