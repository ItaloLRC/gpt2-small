---
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - nanogpt
  - causal-lm
  - text-generation
  - from-scratch
datasets:
  - HuggingFaceFW/fineweb-edu
language:
  - en
---

# Meu GPT-2 Small — 124M

Modelo causal de linguagem com arquitetura equivalente ao GPT-2 Small, treinado do zero sobre o `sample-10BT` do FineWeb-Edu usando o tokenizer GPT-2 do `tiktoken`.

Pesos e configuração: [`itlrc/gpt2-small`](https://huggingface.co/itlrc/gpt2-small).

## Arquitetura

- 124.439.808 parâmetros
- 12 camadas, 12 cabeças de atenção e embedding de 768 dimensões
- contexto máximo de 1.024 tokens
- vocabulário de 50.257 tokens
- pesos compartilhados entre embedding de tokens e cabeça de linguagem

## Treinamento

- 10.500 atualizações do otimizador
- 42 épocas lógicas concluídas
- batch efetivo de 524.288 tokens por atualização
- aproximadamente 5,51 bilhões de tokens processados
- AdamW, gradient clipping em 1.0 e cosine decay com warmup

## Resultados registrados

| Métrica | Resultado final |
|---|---:|
| Train loss | 3,3080 |
| Validation loss | 3,2915 |
| Validation perplexity | 26,88 |
| HellaSwag | 27,68% (10.042 exemplos) |

Os números são os resultados registrados pelo notebook de treinamento. O modelo é experimental e não passou por avaliação ampla de segurança, factualidade ou vieses.

## Limitações

O modelo pode gerar informações incorretas, enviesadas, ofensivas ou repetitivas. Ele não recebeu instruction tuning nem alinhamento por preferência humana. Não deve ser usado para decisões de alto impacto.

## Uso

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "itlrc/gpt2-small",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

inputs = tokenizer("Hi! I'm a", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Formato dos arquivos

O Hub distribui `model.safetensors` com os pesos de inferência e `config.json` com a arquitetura. O checkpoint completo de treinamento, com otimizador, não é necessário para gerar texto e não é distribuído.
