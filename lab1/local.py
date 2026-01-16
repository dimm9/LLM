import os, torch, time
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = "Odpowiadaj po polsku i zwięźle."
USER_PROMPT = "Podaj 3 krótkie pomysły na szybkie dania."

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# 3. Ładowanie Tokenizera ("Tłumacza")
# Tokenizer zamienia tekst na liczby, które rozumie model.
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)

# 4. Ładowanie Modelu ("Mózgu")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    # Jeśli masz kartę graficzną NVIDIA (cuda), użyj lżejszego formatu float16 (szybsze, mniej RAMu).
    # Jeśli nie, użyj standardowego float32 dla procesora.
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    # Automatycznie dopasuj mapowanie (GPU jeśli jest, CPU jeśli nie ma).
    device_map="auto" if torch.cuda.is_available() else None,
)

# Sprawdzamy, na czym uruchomiliśmy model (cuda = karta graficzna, cpu = procesor)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Przenosimy model na urządzenie (choć device_map="auto" zazwyczaj już to robi)
model = model.to(device)

print("Model:", LOCAL_MODEL_NAME)
print("Device:", device)


def build_prompt(system: str, user: str) -> str:
    """Tworzy sformatowany tekst rozmowy (chat template), który rozumie model."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # Jeśli tokenizer ma wbudowany szablon czatu (większość nowych modeli ma), użyj go.
    if hasattr(tokenizer, "apply_chat_template"):
        # tokenize=False -> Zwróć nam sformatowany tekst (string), a nie liczby.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Awaryjny, ręczny format (gdyby tokenizer nie miał szablonu)
    return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"


def count_tokens_local(text: str) -> int:
    """Pomocnicza funkcja do liczenia tokenów w tekście."""
    return len(tokenizer.encode(text))


# @torch.inference_mode() -> Wyłącza obliczanie gradientów (uczenie).
# Dzięki temu kod działa szybciej i zużywa mniej pamięci RAM.
@torch.inference_mode()
def local_generate(
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_output_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()  # Start stopera

    # 1. Przygotowanie tekstu wejściowego
    text = build_prompt(system, prompt)

    # 2. Tokenizacja (Tekst -> Liczby) i wysłanie na urządzenie (GPU/CPU)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 3. Konfiguracja parametrów generowania
    # Jeśli temperatura > 0, włączamy losowanie (sampling). Jeśli 0, model jest deterministyczny.
    do_sample = temperature > 0.0

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_output_tokens,  # Limit długości odpowiedz
        do_sample=do_sample,  # Czy losować słowa?
        eos_token_id=tokenizer.eos_token_id,  # Znak "koniec wypowiedzi"
    )

    # Jeśli losujemy (temp > 0), dodajemy parametry sterujące kreatywnością
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

    # 4. Generowanie (To jest główny moment "myślenia" AI)
    # **inputs rozpakowuje słownik (input_ids, attention_mask)
    output_ids = model.generate(**inputs, **gen_kwargs)

    # 5. Obróbka wyniku
    # Model zwraca [Pytanie + Odpowiedź]. Musimy wyciąć Pytanie, żeby została sama Odpowiedź.
    # inputs["input_ids"].shape[-1] to długość pytania.
    gen_only = output_ids[0, inputs["input_ids"].shape[-1]:]

    # Dekodowanie (Liczby -> Tekst). skip_special_tokens ukrywa techniczne znaczniki.
    output_txt = tokenizer.decode(gen_only, skip_special_tokens=True)

    dt = time.perf_counter() - t0  # Stop stopera

    # Proste (przybliżone) liczenie tokenów do statystyk
    ptoks = count_tokens_local(prompt) + count_tokens_local(system)
    ctoks = count_tokens_local(output_txt)

    return {
        "text": output_txt,
        "latency_s": round(dt, 3),
        "usage": {
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "total_tokens": ptoks + ctoks,
        }
    }


out = local_generate(
    USER_PROMPT,
    system=SYSTEM_PROMPT,
    temperature=0.1,
    top_p=1.0,
    max_output_tokens=256,
)

print(out["text"])
print(out["latency_s"], "s")
print(out["usage"])