'''
Architektura: Router i Dispatcher
    User: "Ile to 2+2?".
    Router (LLM): Analizuje pytanie i zwraca JSON: {"tool": "calculator.add", "args": {"a": 2, "b": 2}}.
    Dispatcher (Python):
    – Sprawdza, czy calculator.add jest na liście dozwolonych narzędzi (allowlist).
    – Uruchamia funkcję Pythonową (np. math.add(2, 2)).
    – Zwraca wynik 4.
    Final Response (Python / LLM) – Opcjonalne: Model dostaje wynik i formuje odpowiedź dla użytkownika: "Wynik to 4"
'''

#-------------------KROK 1: Mini Baza Wiedzy (KB)
import json, os
from typing import List

KB_PATH = "./lab03_kb.json" # Ścieżka do pliku z "bazą wiedzy"
print("KB path:", KB_PATH, "exists:", os.path.exists(KB_PATH)) # czy plik faktycznie istnieje na dysku

def load_kb(path=KB_PATH) -> List[dict]:
    # Otwórz plik bezpiecznie w trybie odczytu (r) i kodowaniu UTF-8
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) # Zamień tekst JSON na listę słowników w Pythonie

# Wczytaj dane tylko jeśli plik istnieje, w przeciwnym razie ustaw pustą listę
KB = load_kb() if os.path.exists(KB_PATH) else []
print("KB size:", len(KB)) # Wypisz ile elementów (wierszy) udało się wczytać


import re
from typing import Dict, Any, List

def normalize(text: str) -> List[str]:
    """Czyści tekst: małe litery, usuwa interpunkcję, zostawia słowa."""
    text = text.lower()
    # Znajdź wszystkie ciągi znaków składające się z liter i cyfr.
    # regex [a-ząćęłńóśźż0-9]+ to "dopasuj jedną lub więcej liter/cyfr"
    tokens = re.findall(r"[a-ząćęłńóśźż0-9]+", text, flags=re.IGNORECASE)
    return tokens

def score_entry(query_tokens: List[str], entry: Dict[str, Any]) -> float:
    """Oblicza punkty dla jednego rekordu z bazy na podstawie zapytania."""
    # 1. Przygotowanie danych z rekordu (czyszczenie i podział na słowa)
    c_tokens = normalize(entry.get("content", ""))
    t_tokens = normalize(entry.get("title", ""))
    # Tagi są listą tekstów, więc normalizujemy każdy z osobna
    tags = [normalize(t) for t in entry.get("tags", [])]
    # Spłaszczamy listę list do jednej prostej listy słów z tagów
    tags_flat = [t for sub in tags for t in sub]

    # 2. Liczenie punktów (Heurystyka)
    # Sprawdzamy każde słowo z zapytania (q) czy występuje w danym polu
    # Za każde wystąpienie w treści: 1.0 punktu
    base = sum(1 for q in query_tokens if q in c_tokens) * 1.0
    # Za każde wystąpienie w tytule: 1.5 punktu (tytuł ważniejszy!)
    title_bonus = sum(1 for q in query_tokens if q in t_tokens) * 1.5
    # Za każde wystąpienie w tagach: 1.2 punktu
    tag_bonus = sum(1 for q in query_tokens if q in tags_flat) * 1.2
    return base + title_bonus + tag_bonus    # Zwracamy sumę wszystkich punktów

# Rekord ma w tytule "Kąty", "i" "liczby" (3 trafienia * 1.5)
# Rekord ma w treści "Ala", "ma", "i", "liczby" (4 trafienia * 1.0)
print(score_entry(normalize("Ala ma kąty i 123 liczby!"), {
    "title": "Kąty i liczby",
    "content": "Ala ma kota i lubi liczby.",
    "tags": ["zwierzęta", "matematyka"]
}))


# -------------------- KROK 2: Bezpieczeństwo (Dispatcher & Pydantic)
'''
Dwuwarstwowa ochrona:
    Allowlist: Model może wybrać tylko narzędzie z góry zdefiniowanej listy (ALLOWED_TOOLS).
    Pydantic: Każde narzędzie ma swój schemat argumentów. Jeśli model zwróci string tam, gdzie ma być float – Pydantic odrzuci to wywołanie.
'''




#---------------------- KROK 3: Router (LLM)
SYSTEM = """
    You are a routing controller.
    TASK:
    Pick EXACTLY ONE tool for the user request.
    Return ONLY valid minified JSON:
    {"tool": "<tool name>", "args": { ...required args... }}
    TOOLS AND WHEN TO USE THEM (STRICT):
    1. "kb.lookup"
       Use this if the user asks for any explanation, definition, concept, meaning, description, summary, background, or "what is ...".
       Examples: "co to jest embedding", "wyjaśnij...", "czym jest X", "jak działa Y".
       This is the DEFAULT tool unless another rule clearly matches.
       Required args:
       {"query": <the user's question as a short keyword or phrase>, "top_k": 5}
    2. "calculator.add" / "calculator.sub" / "calculator.mul" / "calculator.div"
       Use ONLY if the user explicitly asks you to compute a numeric result of two numbers.
       Required args: {"a": float, "b": float}
    3. "units.convert"
       Use ONLY if the user explicitly asks to convert units between km and mi, or between c and f.
       Required args:
       {"value": float, "from_unit": "km|mi|c|f", "to_unit": "km|mi|c|f"}

    4. "files.search"
       Use ONLY if the user asks to search local documents/files by name/pattern.
       Required args:
       {"pattern": string}
    RULES:
    - If none of the non-default rules match, use "kb.lookup".
    - Never invent numbers or units if the user didn't ask about numbers or units.
    - Never leave args incomplete.
    - No prose, no code fences, ONLY JSON.
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os, torch, time
from typing import Optional, Dict, Any, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

load_dotenv()

MODEL_MODE = os.getenv("MODEL_MODE", 'groq')

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

BASE_URL = os.getenv("GROQ_BASE_URL")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

SYSTEM_PROMPT = "Odpowiadaj po polsku i zwięźle."
USER_PROMPT = "Podaj 3 krótkie pomysły na szybkie dania."

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def build_prompt(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"


def count_tokens_local(text: str) -> int:
    return len(tokenizer.encode(text))

@torch.inference_mode()
def local_generate(
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_output_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        format_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    text = build_prompt(system, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_output_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
    if format_fn is not None:
        gen_kwargs["prefix_allowed_tokens_fn"] = format_fn
    output_ids = model.generate(**inputs, **gen_kwargs)
    gen_only = output_ids[0, inputs["input_ids"].shape[-1]:]
    output_txt = tokenizer.decode(gen_only, skip_special_tokens=True)
    dt = time.perf_counter() - t0
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

def gemini_generate(prompt : str, system : str = "You are a helpful assistant", temperature : float = 0.0, top_p : float = 1.0, top_k : int = 40, max_output_tokens : int = 256) -> dict:
    gclient = genai.Client(api_key=GOOGLE_API_KEY)
    t0 = time.perf_counter()
    response = gclient.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        ),
    )
    dt = time.perf_counter() - t0
    usage = getattr(response, "usage_metadata", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    } if usage is not None else None
    text = getattr(response, "text", None)
    gclient.close()
    return {
        "text": text if text is not None else str(response),
        "latency_s": round(dt, 3),
        "usage": usage_dict,
    }

def groq_generate(prompt : str, system : str = "You are a helpful assistant", temperature : float = 0.0, top_p : float = 1.0, max_output_tokens : int = 256) -> dict:
    client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)
    t0 = time.perf_counter()
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    dt = time.perf_counter() - t0
    #usage - informacja o zużyciu zasobów (tokenów) przez model podczas generowania odpowiedzi
    usage = getattr(response, "usage", None)
    usage_dict = None if usage is None else getattr(usage, "model_dump", lambda: usage)()
    client.close()
    return {
        "text": response.output_text,
        "latency_s": round(dt, 3),
        "usage": usage_dict,
    }


def chat_once_demo(
        prompt : str,
        system : str = "You are a helpful assistant",
        temperature : float = 0.0,
        top_p : float = 1.0,
        top_k : Optional[int] = None,
        max_output_tokens : int = 256,
        format_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
) -> dict:
    if MODEL_MODE == "gemini":
        return gemini_generate(
            prompt=prompt,
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    elif MODEL_MODE == "groq":
        return groq_generate(
            prompt=prompt,
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        return local_generate(
            prompt=prompt,
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            format_fn=format_fn
        )


