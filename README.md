
# SFTQwen

SFTQwen fine-tunes a small language model (SLM) to perform **news summarization** using **supervised fine-tuning (SFT)** under strict resource constraints.

## Objective

- Train a summarization-capable SLM from a larger instruction-tuned base model.
- Operate under **≤16 GB VRAM**.
- Use a compact training set of **~5,000 synthetic examples**.

## Data: teacher–student supervision

The training data is **automatically generated**:

- A larger **teacher** model produces summaries for news articles.
- The resulting (article, summary) pairs form the SFT dataset for the smaller **student** model.

This is a teacher–student setup: the student learns to imitate the teacher’s summaries rather than relying on human-labeled references.

## Fine-tuning approach (conceptual)

Training uses a parameter-efficient setup consistent with **QLoRA-style fine-tuning**:

- The base model is loaded in reduced precision (4-bit quantized weights).
- Only lightweight adapter parameters (LoRA) are trained.

Implementation details and the full methodology live in the accompanying report/notebooks; this README focuses on how to run the code.

## How to run

The CLI entry point is [main.py](main.py).

### 1) Generate synthetic training data (annotation)

Creates an on-disk dataset of synthetic summaries.

```bash
python main.py --mode annotate
```

### 2) Fine-tune the student model

Runs SFT on the generated dataset and saves the trained adapters.

```bash
python main.py --mode train
```

### 3) Evaluate on a held-out test split

Computes summarization metrics on a test split that is not used for training.

```bash
python main.py --mode eval
```

## Configuration

Configuration is centralized in [src/config.py](src/config.py). It includes:

- Teacher model and annotation settings
- Student model settings
- Training/evaluation settings
- Prompt templates

## Project layout

- [src/annotator.py](src/annotator.py): teacher-driven dataset generation
- [src/train.py](src/train.py): supervised fine-tuning (student)
- [src/generate.py](src/generate.py): inference helpers used by evaluation
- [src/eval.py](src/eval.py): evaluation loop and metrics
- [src/dataset.py](src/dataset.py): dataset splitting and tokenization helpers

For a more code-oriented module overview, see [README_MODULES.md](README_MODULES.md).

## Contact

For questions or collaboration: marius.dragic@gmail.com

## License

See [LICENSE](LICENSE).
