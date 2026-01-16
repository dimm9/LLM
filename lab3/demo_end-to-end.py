import json, os
import re
from typing import Dict, Any, List, Tuple

from transformers import AutoTokenizer

from lab3.agentic_ai import chat_once_demo

KB_PATH = "./lab03_kb.json"
print("KB path:", KB_PATH, "exists:", os.path.exists(KB_PATH))

def load_kb(path=KB_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) # Zamień tekst

KB = load_kb() if os.path.exists(KB_PATH) else []
print("KB size:", len(KB))

def normalize(text: str) -> List[str]:
    tokens = re.findall(r"[a-ząćęłńóśźż0-9]+", text, flags=re.IGNORECASE)
    return tokens

def score_entry(query_tokens: List[str], entry: Dict[str, Any]) -> float:
    c_tokens = normalize(entry.get("content", ""))
    t_tokens = normalize(entry.get("title", ""))
    tags = [normalize(t) for t in entry.get("tags", [])]
    tags_flat = [t for sub in tags for t in sub]
    base = sum(1 for q in query_tokens if q in c_tokens) * 1.0
    title_bonus = sum(1 for q in query_tokens if q in t_tokens) * 1.5
    tag_bonus = sum(1 for q in query_tokens if q in tags_flat) * 1.2
    return base + title_bonus + tag_bonus

def kb_lookup(query : str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not KB:
        return []
    qtok = normalize(query) # zamień pytanie użytkownika na listę czystych słów (np. "What is token?" -> ['what', 'is', 'token']).
    scored: List[Tuple[float, Dict[str, Any]]] = [] # pusta lista, w której będziemy trzymać pary: (liczba punktów, treść artykułu).
    for item in KB:
        s = score_entry(qtok, item) # Oblicz punkty dla tego artykułu
        if s > 0:
            # Jeśli artykuł zdobył jakiekolwiek punkty dodaj go do listy wyników jako parę (punkty, artykuł).
            scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True) # Posortuj listę wyników malejąco według punktów (x[0]), żeby najlepsze były na górze.
    # Weź 'top_k' najlepszych wyników, wyrzuć informację o punktach (zmienna _) i zwróć same artykuły.
    return [entry for _, entry in scored[:max(1, top_k)]]

kb_lookup("What is token?", top_k=2)

# -------------------- KROK 2: Bezpieczeństwo (Dispatcher & Pydantic)
import glob
from pydantic import BaseModel, Field
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

class CalcArgs(BaseModel):                        # Model danych dla kalkulatora (wymusza typy).
    a: float                                      # Pierwsza liczba (musi być zmiennoprzecinkowa).
    b: float                                      # Druga liczba.

class ConvertArgs(BaseModel):                     # Model danych dla konwersji jednostek.
    value: float                                  # Wartość do przeliczenia.
    from_unit: Literal["km","mi","c","f"]         # Jednostka źródłowa (wybór ze ścisłej listy).
    to_unit:   Literal["km","mi","c","f"]         # Jednostka docelowa.

class SearchArgs(BaseModel):                      # Model danych dla szukania plików.
    pattern: str = Field(..., max_length=64)      # Wzorzec wyszukiwania (max 64 znaki).

class KBArgs(BaseModel):                          # Model danych dla przeszukiwania bazy wiedzy.
    query: str                                    # Treść zapytania.
    top_k: int = Field(3, ge=1, le=10)            # Liczba wyników (domyślnie 3, min 1, max 10).

class ToolCall(BaseModel):                        # Główny model wywołania narzędzia.
    tool: Literal["calculator.add","calculator.sub","calculator.mul","calculator.div", "units.convert","files.search","kb.lookup"]         # Nazwa narzędzia (musi być jedną z wymienionych).
    args: Dict[str, Any]                          # Słownik z argumentami dla narzędzia.

ALLOWED_TOOLS = {                                  # Zbiór dozwolonych nazw narzędzi (do szybkiego sprawdzania).
    "calculator.add","calculator.sub","calculator.mul","calculator.div",
    "units.convert","files.search","kb.lookup"
}

def _run_tool_sync(tc: ToolCall) -> Dict[str, Any]: # Funkcja wykonująca logikę narzędzia (synchronicznie).
    if tc.tool not in ALLOWED_TOOLS:              # Sprawdź, czy narzędzie jest na liście dozwolonych.
        raise ValueError("Tool not allowed")      # Jeśli nie, zgłoś błąd.
    if tc.tool.startswith("calculator."):         # Jeśli to narzędzie kalkulatora...
        args = CalcArgs(**tc.args)                # Zweryfikuj i wczytaj argumenty do modelu CalcArgs.
        op = tc.tool.split(".")[1]                # Wyciągnij nazwę operacji (np. "add" z "calculator.add").
        if op == "add": res = args.a + args.b     # Obsługa dodawania.
        elif op == "sub": res = args.a - args.b   # Obsługa odejmowania.
        elif op == "mul": res = args.a * args.b   # Obsługa mnożenia.
        elif op == "div":                         # Obsługa dzielenia.
            if args.b == 0: raise ValueError("Division by zero")   # Zabezpieczenie przed dzieleniem przez zero.
            res = args.a / args.b                 # Wykonaj dzielenie.
        else:                                     # Jeśli operacja nieznana...
            raise ValueError("Unknown calc op")   # Zgłoś błąd.
        return {"result": res}                    # Zwróć wynik w słowniku.
    if tc.tool == "units.convert":                # Jeśli to konwersja jednostek...
        args = ConvertArgs(**tc.args)             # Walidacja argumentów.
        v, fr, to = args.value, args.from_unit, args.to_unit # Przypisanie zmiennych dla czytelności.
        if fr == to: return {"result": v}         # Jeśli jednostki te same, zwróć bez zmian.
        if fr == "km" and to == "mi": ...         # Przelicz km na mile.
        if fr == "mi" and to == "km": ...         # Przelicz mile na km.
        if fr == "c" and to == "f":  ...          # Przelicz Celsjusze na Fahrenheity.
        if fr == "f" and to == "c":  ...          # Przelicz Fahrenheity na Celsjusze.
        raise ValueError("Unsupported conversion")# Jeśli inna para jednostek, zgłoś błąd.
    if tc.tool == "files.search":                 # Jeśli to szukanie plików...
        args = SearchArgs(**tc.args)              # Walidacja argumentów.
        pat = args.pattern.replace("..","")[:64]  # Usuń ".." (zabezpieczenie przed wyjściem z katalogu).
        paths = glob.glob(pat)                    # Znajdź ścieżki pasujące do wzorca.
        files = [os.path.basename(p) for p in paths if os.path.isfile(p)]         # Pobierz same nazwy plików (ignoruj katalogi).
        return {"files": files[:50]}              # Zwróć max 50 pierwszych plików.
    if tc.tool == "kb.lookup":                    # Jeśli to baza wiedzy...
        args = KBArgs(**tc.args)                  # Walidacja argumentów.
        hits = kb_lookup(args.query, top_k=args.top_k)         # Wywołaj funkcję szukającą w bazie (zdefiniowaną wcześniej).
        return {"hits": hits}                     # Zwróć znalezione artykuły.
    raise ValueError("Unhandled tool")            # Jeśli narzędzie jest na liście ALLOWED, ale nie ma tu kodu 'if'.

def run_tool(tc: ToolCall, timeout_s: float = 2.0): # Wrapper uruchamiający narzędzie z limitem czasu.
    with ThreadPoolExecutor(max_workers=1) as ex:   # Utwórz pulę wątków (żeby móc przerwać działanie).
        fut = ex.submit(_run_tool_sync, tc)   # Uruchom właściwą funkcję w tle.
        try:                                        # Rozpocznij blok próbny.
            out = fut.result(timeout=timeout_s)     # Czekaj na wynik max 2 sekundy.
            return True, out, None                  # Sukces: (True, wynik, brak błędu).
        except FuturesTimeout:                      # Jeśli minął czas...
            return False, {}, "timeout"             # Zwróć błąd timeoutu.
        except Exception as e:                      # Jeśli wystąpił inny błąd (np. walidacji)...
            return False, {}, str(e)                # Zwróć treść błędu.

#---------------------- KROK 3: Router (LLM)
from pydantic import ValidationError
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

def extract_json_maybe(text: str) -> str:
    print("Raw LLM output:", text)
    # Szukaj JSON-a zamkniętego w bloku kodu Markdown (```json ... ```)
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m: return m.group(1)  # Jeśli znaleziono blok kodu, zwróć tylko jego zawartość
    m = re.search(r"(\{[\s\S]*\})", text) # Szukaj pierwszego otwarcie klamry '{' i ostatniego zamknięcia '}' To zadziała, jeśli model zapomni o ```backtickach```, ale poda poprawny JSON
    return m.group(1) if m else text

USE_LOCAL = False
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
GUIDED_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["kb.lookup", "calculator.add", "calculator.sub", "calculator.mul", "calculator.div", "units.convert", "files.search"]
        },
        "args": {
            "type": "object"
        }
    },
    "required": ["tool", "args"],
    "additionalProperties": False
}

def ask_for_tool_decision(user_prompt: str, retries: int = 3) -> ToolCall:
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
    INSTRUCTION = "Return ONLY valid JSON matching the schema. No extra text."
    last_err = None
    lp = None
    format_fn = None
    if USE_LOCAL:
        schema_dict = GUIDED_SCHEMA
        format_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema_dict))
    for i in range(retries):
        raw = chat_once_demo(
            f"{user_prompt}\n\n{INSTRUCTION}",
            system=SYSTEM,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=220,
            format_fn=format_fn
        )
        try:
            js = extract_json_maybe(raw['text'])
            data = json.loads(js)
            tc = ToolCall(**data)
            if tc.tool not in ALLOWED_TOOLS:
                raise ValidationError(f"Tool not allowed: {tc.tool}", ToolCall)
            return tc
        except Exception as e:
            last_err = str(e)
            user_prompt = f"{user_prompt}\nIf you fail, return minimal JSON only. Error: {last_err}"
    raise RuntimeError(f"Failed to obtain valid ToolCall JSON after {retries} tries. Last error: {last_err}")

def compose_final_with_llm(user_prompt: str, tool_output: Dict[str,Any]) ->  Dict[str, Any]:
    sysmsg = "You are a concise tutor. Use the provided knowledge base hits to answer briefly and accurately."
    prompt = f"User: {user_prompt}\nKB hits JSON: {json.dumps(tool_output, ensure_ascii=False)}\nAnswer in 2-3 sentences."
    return chat_once_demo(
        prompt,
        system=sysmsg,
        temperature=0.0,
        max_output_tokens=160
    )

def call_llm_with_kb_lookup(user_prompt: str) -> str:
    tc = ask_for_tool_decision(user_prompt)
    ok, out, err = run_tool(tc)
    if not ok:
        return f"Tool call failed: {err}"
    return compose_final_with_llm(user_prompt, out)['text']

user_prompt = "Wytłumacz krótko co to jest embedding (jeśli możesz, skorzystaj z bazy wiedzy)."
print(call_llm_with_kb_lookup(user_prompt))

print("----------------------------------------")

def compose_final_with_llm_v2(user_prompt: str, tool_output: Dict[str,Any]) -> str:
    sysmsg =  "You are a precise assistant. Use the KB hits to answer and list titles + ids as sources."
    prompt = f"User: {user_prompt}\nKB hits JSON: {json.dumps(tool_output, ensure_ascii=False)}\nFormat: paragraph + 'References: <title> (id), ...'"
    return chat_once_demo(
        prompt,
        system=sysmsg,
        temperature=0.0,
        max_output_tokens=160
    )['text']

def call_llm_with_kb_lookup_v2(user_prompt: str, extra_tool_args: Dict[str, Any]) -> str:
    tc = ask_for_tool_decision(user_prompt)
    # KROK 2: Nadpisanie parametrów (Ważne!)
    # Tutaj Twój kod ręcznie "wtrąca się" w decyzję AI.
    # Np. jeśli przekazałeś {'top_k': 3}, to nawet jak AI chciało 5 wyników, wymuszasz 3.
    for k, v in extra_tool_args.items():
        tc.args[k] = v    # np tc.args['top_k'] = 3
    ok, out, err = run_tool(tc)
    if not ok:
        return f"Tool call failed: {err}"
    return compose_final_with_llm(user_prompt, out)['text']

user_prompt = "Wyjaśnij w jednym akapicie, czym jest RAG i podaj źródła z KB."
print(call_llm_with_kb_lookup_v2(user_prompt, {'top_k': 3}))


# ----------------------- KROK 4 : Walka z halucynacjami (kompozycja odpowiedzi)
'''
# Rozwiązaniem będzie rozdzielenie generacji treści od źródeł.
#     Pobieramy hity z kb.lookup.
#     Wkładamy je do promptu: „Odpowiedz WYŁĄCZNIE na podstawie poniższego kontekstu”.
#     Sekcję „References:” doklejamy sami w Pythonie, używając ID i tytułów z bazy, a nie z generacji modelu.
'''

print("----------------------------------------")

def compose_final_with_llm_v3(user_prompt: str, tool_output: Dict[str, Any]) -> Dict[str, Any]:
    # bierzemy tylko top hity, żeby nie zalać modelu
    hits = tool_output.get("hits", [])
    kb_context = []
    for h in hits:
        kb_context.append(
            f"[ID: {h.get('id')}] {h.get('title')}: {h.get('content')}"
        )
    kb_context_str = "\n".join(kb_context)

    sysmsg = (
        "You are a precise assistant.\n"
        "Answer ONLY using the KB context.\n"
        "If the KB does not explicitly contain the answer, say exactly: 'Brak danych w KB.'\n"
        "STRICT RULES:\n"
        "- Do NOT add any facts, names, places, dates, papers, URLs, or acronyms that are not literally present in KB context.\n"
        "- Do NOT guess expansions of acronyms.\n"
        "- Do NOT invent references.\n"
        "- Output ONE short paragraph in Polish.\n"
        "- Do NOT output 'References:' section. That will be added later by the system."
    )

    prompt = (
        f"User question:\n{user_prompt}\n\n"
        f"KB context(only permitted knowledge :\n{kb_context_str}\n\n"
        f"One paragraph. No references section."
    )

    resp = chat_once_demo(
        prompt,
        system=sysmsg,
        temperature=0.0,
        max_output_tokens=160
    )
    paragraph = resp["text"].strip()

    # teraz źródła robimy sami, deterministycznie:
    refs_list = [f"{h.get('title')} ({h.get('id')})" for h in hits]
    refs_text = "References: " + ", ".join(refs_list) if refs_list else "References: brak"

    resp['text'] = paragraph + "\n\n" + refs_text
    return resp

user_prompt = "Wyjaśnij w jednym akapicie, czym jest RAG i podaj źródła z KB."
print(call_llm_with_kb_lookup_v2(user_prompt, {'top_k': 3}))

