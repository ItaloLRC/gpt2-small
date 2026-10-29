from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nano_gpt.model import GPT, GPTConfig, clean_state_dict  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separa pesos de inferência e métricas do checkpoint completo."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "weights")
    parser.add_argument("--artifacts-dir", type=Path, default=REPOSITORY_ROOT / "artifacts")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # weights_only evita desserializar objetos Python arbitrários. Ainda assim,
    # use somente um checkpoint que você criou ou recebeu de fonte confiável.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    required = {"config", "model", "history"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Chaves ausentes no checkpoint: {sorted(missing)}")

    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    if args.dtype == "float16":
        model.half()
    elif args.dtype == "bfloat16":
        model.bfloat16()

    weights_path = args.output_dir / "model.safetensors"
    save_model(
        model,
        weights_path,
        metadata={
            "format": "pt",
            "architecture": "GPT-2 Small",
            "tokenizer": "gpt2",
            "dtype": args.dtype,
        },
    )

    config_data = asdict(config) | {
        "architecture": "GPT",
        "tokenizer": "gpt2",
        "parameter_count": model.parameter_count,
        "dtype": args.dtype,
    }
    write_json(args.output_dir / "config.json", config_data)

    history = checkpoint["history"]
    training_data = {
        "source_checkpoint": args.checkpoint.name,
        "global_step": int(checkpoint.get("global_step", 0)),
        "saved_epoch_zero_based": int(checkpoint.get("epoch", len(history) - 1)),
        "completed_epochs": len(history),
        "history": history,
    }
    write_json(args.artifacts_dir / "training_history.json", training_data)

    with (args.artifacts_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_loss", "val_loss", "val_perplexity"),
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"Pesos de inferência: {weights_path}")
    print(f"Configuração: {args.output_dir / 'config.json'}")
    print(f"Histórico: {args.artifacts_dir / 'training_history.json'}")
    print("O estado do otimizador não foi copiado para os pesos de inferência.")


if __name__ == "__main__":
    main()
