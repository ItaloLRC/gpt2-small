# GPT-2 Small (124M) trained from scratch

English | [Português (Brasil)](README.pt-BR.md)

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-itlrc%2Fgpt2--small-yellow)](https://huggingface.co/itlrc/gpt2-small)

Implementation and training pipeline for a causal language model with an architecture equivalent to GPT-2 Small. The model was trained from scratch on FineWeb-Edu's `sample-10BT`, using the GPT-2 tokenizer from `tiktoken`.

This repository contains the pipeline, code, and experiment metrics. Trained weights are distributed separately on the [Hugging Face Hub](https://huggingface.co/itlrc/gpt2-small), keeping large binary files out of the Git history.

## Published model

The inference checkpoint is available at [`itlrc/gpt2-small`](https://huggingface.co/itlrc/gpt2-small). The model repository contains FP32 Safetensors weights, the architecture configuration, and a custom Transformers integration.

| Resource | Location |
|---|---|
| Code and training pipeline | This repository |
| Inference weights | [Hugging Face](https://huggingface.co/itlrc/gpt2-small) |
| Dataset used | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |

## Results

The published checkpoint includes 42 logical training epochs.

| Item | Result |
|---|---:|
| Parameters | 124,439,808 |
| Optimizer updates | 10,500 |
| Tokens processed | 5,505,024,000 |
| Train loss | 3.3080 |
| Validation loss | 3.2915 |
| Validation perplexity | 26.88 |

![Training curves](artifacts/training_curves.png)

## Architecture

- 12 Transformer blocks;
- 12 attention heads per block;
- embedding dimension of 768;
- maximum context of 1,024 tokens;
- vocabulary of 50,257 tokens;
- weights shared between token embeddings and the language modeling head;
- causal attention with PyTorch's `scaled_dot_product_attention`.

## Training configuration

- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT` configuration;
- streaming corpus loading;
- GPT-2 tokenizer;
- effective batch of 524,288 tokens per update;
- 250 updates per logical epoch;
- AdamW with betas `(0.9, 0.95)` and weight decay of `0.1`;
- gradient clipping at `1.0`;
- warmup followed by cosine decay;
- validation on a fixed partition of held-out documents.

Because the dataset is streamed, an "epoch" in this project means a logical interval of 250 updates rather than a full pass over the corpus.

## Repository contents

```text
.
├── artifacts/
│   ├── training_curves.png
│   ├── training_history.csv
│   └── training_history.json
├── notebooks/
│   ├── gpt2small.ipynb
│   └── gpt2small_kaggle_quickstart.ipynb
├── scripts/
│   ├── export_checkpoint.py
│   └── plot_history.py
├── src/nano_gpt/
│   ├── generate.py
│   └── model.py
├── config.json
├── MODEL_CARD.md
├── pyproject.toml
└── requirements.txt
```

`notebooks/gpt2small.ipynb` documents the configuration used for the actual training run. `notebooks/gpt2small_kaggle_quickstart.ipynb` is a shorter variant for validating the pipeline on Kaggle before starting a long run. The reusable architecture implementation is in `src/nano_gpt/model.py`, and metrics extracted from training are in `artifacts/`.

## Installation

Requires Python 3.10 or newer and a PyTorch version compatible with the available hardware.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[training]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[training]"
```

To load the published model directly from the Hub, also install the optional integration:

```bash
pip install -e ".[hub]"
```

## Running the pipeline

To inspect the configuration that produced the published model, open `notebooks/gpt2small.ipynb`. For a quick Kaggle test, use `notebooks/gpt2small_kaggle_quickstart.ipynb`; this variant uses only 5 logical epochs, 100 steps per epoch, and temporarily limits HellaSwag to 100 examples.

Before starting a new training run, adjust the following in the selected notebook:

- the checkpoint directory;
- `B` according to GPU memory;
- `STEPS_PER_EPOCH` and `NUM_EPOCHS` for the desired token budget;
- `VAL_STEPS` for an acceptable validation cost.

The dataset is streamed directly from Hugging Face, so internet access is required during training.

## Metrics and plots

The complete per-epoch history is available in JSON and CSV formats. To regenerate the curves from this data:

```bash
python scripts/plot_history.py
```

## Using the published model

Because the architecture was implemented in this project, loading it through Transformers uses `trust_remote_code=True`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "itlrc/gpt2-small"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

inputs = tokenizer("Hi! I'm a", return_tensors="pt")
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

`trust_remote_code=True` is required because the Hub needs to load `configuration_nanogpt.py` and `modeling_nanogpt.py` along with the weights. Review remote code before enabling this option for third-party models.

For checkpoints produced locally by the notebook, the `meu-gpt2-generate` command remains available. The `scripts/export_checkpoint.py` utility separates inference weights, configuration, and history from the full checkpoint.
