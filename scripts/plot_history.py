from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plota loss e perplexidade por época.")
    parser.add_argument(
        "--history",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "training_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "training_curves.png",
    )
    args = parser.parse_args()

    payload = json.loads(args.history.read_text(encoding="utf-8"))
    history = payload["history"] if isinstance(payload, dict) else payload
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    perplexity = [item["val_perplexity"] for item in history]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (loss_axis, ppl_axis) = plt.subplots(1, 2, figsize=(13, 5.2))
    figure.suptitle("GPT-2 Small (124M) — evolução do treinamento", fontsize=16, weight="bold")

    loss_axis.plot(epochs, train_loss, label="Treino", color="#2563eb", linewidth=2.2)
    loss_axis.plot(epochs, val_loss, label="Validação", color="#f97316", linewidth=2.2)
    loss_axis.set(title="Loss por época", xlabel="Época", ylabel="Cross-entropy loss")
    loss_axis.legend(frameon=True)

    ppl_axis.plot(epochs, perplexity, color="#7c3aed", linewidth=2.2)
    ppl_axis.set_yscale("log")
    ppl_axis.set(title="Perplexidade de validação", xlabel="Época", ylabel="Perplexidade (escala log)")

    final = history[-1]
    loss_axis.scatter([final["epoch"]], [final["val_loss"]], color="#f97316", zorder=3)
    loss_axis.annotate(
        f"{final['val_loss']:.3f}",
        (final["epoch"], final["val_loss"]),
        xytext=(-32, 12),
        textcoords="offset points",
    )
    ppl_axis.scatter([final["epoch"]], [final["val_perplexity"]], color="#7c3aed", zorder=3)
    ppl_axis.annotate(
        f"{final['val_perplexity']:.2f}",
        (final["epoch"], final["val_perplexity"]),
        xytext=(-36, 12),
        textcoords="offset points",
    )

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    print(f"Gráficos salvos em {args.output} e {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
