import tiktoken
from typing import List

def approx_tokens(texts: List[str], model_name: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return sum(len(enc.encode(t)) for t in texts)
    except Exception:
        # prosta heurystyka (z grubsza): 1 token ~ 4 znaki (ang), PL bywa inaczej
        return sum(max(1, len(t)//4) for t in texts)

def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    return prompt_tokens * price_in + completion_tokens * price_out

# Użycie:
prompt = "Podaj 3 krótkie pomysły na aktywność fizyczną w domu."
completion = "1) ...\n2) ...\n3) ..."

pt = approx_tokens([prompt])
ct = approx_tokens([completion])

# UWAGA: tu podstaw realne ceny z cennika dostawcy/modelu
print("USD:", estimate_cost_usd(pt, ct, price_in=0.0, price_out=0.400))