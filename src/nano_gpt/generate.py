from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
import torch

from .model import GPT, GPTConfig, clean_state_dict


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    config_path: Path | None,
    device: torch.device,
) -> GPT:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_model as load_safetensors_model

        if config_path is None:
            config_path = checkpoint_path.with_name("config.json")
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        config = GPTConfig(
            **{key: raw_config[key] for key in GPTConfig.__dataclass_fields__}
        )
        model = GPT(config)
        load_safetensors_model(model, checkpoint_path, strict=True, device="cpu")
    else:
        # Carregue somente checkpoints próprios ou de uma fonte confiável.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("Formato esperado: dicionário contendo as chaves 'config' e 'model'.")
        config = GPTConfig(**checkpoint["config"])
        model = GPT(config)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature deve ser maior que zero")
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.eot_token]
    inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = inputs[:, -model.config.block_size :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            cutoff = torch.topk(next_logits, k).values[:, [-1]]
            next_logits = next_logits.masked_fill(next_logits < cutoff, -torch.inf)
        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        inputs = torch.cat((inputs, next_token), dim=1)
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(inputs[0].tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera texto com o meu GPT-2 Small.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prompt", default="Hi! I'm a")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model = load_model(args.checkpoint, args.config, device)
    print(
        generate(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
    )


if __name__ == "__main__":
    main()
