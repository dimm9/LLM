from lab2.chat_once import chat_once
#=================jak treść polecenia wpływa na odpowiedź modelu językowego==========
prompts = [
    "Summarize this text.",
    "Summarize this text in one sentence.",
    "Summarize this text in one sentence using simple English and output as JSON {summary: ...}"
]
text = """
Artificial intelligence (AI) is a field of computer science that builds systems able to perform tasks that typically require human intelligence-such as understanding language, learning from data, reasoning, and perception.
What AI can do: perception (vision/speech), reasoning, learning, interaction (natural language), and planning/control.
How it works (at a glance):
- Symbolic AI: hand-written rules and logic.
- Machine learning: models learn patterns from data.
- Deep learning: multi-layer neural networks for images, speech, and text.
"""
for p in prompts:
    print("\n---\nPrompt:", p)
    response = chat_once(f"{p}\n\n{text}")
    print("Answer:", response['text'])
    print(f"---⏱ {response['latency_s']}s | Tokens: {response['usage']['total_tokens']}")
'''Traktuj prompt jak specyfikację funkcji. Musisz zdefiniować nie tylko „co” model ma zrobić, ale także „w jakim formacie” ma to zwrócić. '''


# ================= System Prompt („Osobowość”)
# System prompt - instrukcja „nadrzędna”, ustawia kontekst całej rozmowy. Tutaj definiujemy zasady, których użytkownik nie powinien móc nadpisać.
roles = [
    "You are a sarcastic assistant.",
    "You are a formal university lecturer.",
    "You are a motivational coach."
]
question = "Explain recursion in one sentence."
for r in roles:
    print("\n---\nRole:", r)
    print("Question:", question)
    response = chat_once(f"{question}", system=r, temperature=0.3)
    print("Answer:", response['text'])
    print(f"---⏱ {response['latency_s']}s | Tokens: {response['usage']['total_tokens']}")

'''
W projekcie końcowym w system_prompt wylądują instrukcje bezpieczeństwa 
(Guardrails) oraz definicja dostępnych narzędzi !!!
'''

# ============ Few-Shot Prompting (Uczenie na przykładach) ===============
question = """
Translate English → Polish:
Input: Good morning → Output: Dzień dobry
Input: Thank you → Output: Dziękuję
Input: See you later → Output:
"""

print("---\nQuestion:", question)
response = chat_once(f"{question}", temperature=0.3)
print("Answer:", response['text'])
print(f"---⏱ {response['latency_s']}s | Tokens: {response['usage']['total_tokens']}")


#============Chain-of-Thought (Myśl powoli – krok po kroku)==========
# dobry do zadan logicznych lub matematycznych
question = "If there are 3 red and 5 blue balls, and you take one randomly, what is the probability it’s red?"

print("---\nQuestion:", question)
response = chat_once(f"{question}", temperature=0.3)
print("---Answer (without CoT):", response['text'])
print(f"---⏱ {response['latency_s']}s | Tokens: {response['usage']['total_tokens']}")

response = chat_once(f"{question}\nThink step by step.", temperature=0.3)
print("---Answer (with CoT):", response['text'])
print(f"---⏱ {response['latency_s']}s | Tokens: {response['usage']['total_tokens']}")

'''
CoT generuje znacznie więcej tokenów. W produkcji używaj go tylko tam, gdzie logika jest kluczowa. 
W prostych zadaniach to strata pieniędzy i czasu (latency)'''


# ============ Walidacja i naprawa JSON =============

'''
Naiwna próba: JSON w promptcie
Często przy lepszych modelach wystarczające.
'''

import json

prompt = "Classify the sentiment of the text as positive, negative, or neutral. Return JSON {\"sentiment\": \"...\"}."
text = "I love how easy this app is to use!"
r = chat_once(f"{prompt}\n\nTEXT: {text}", temperature=0.0)
print(r["text"])
try:
    data = json.loads(r["text"])
    print("Parsowanie OK:", data)
except json.JSONDecodeError:
    print("Nie jest to czysty JSON. Potrzebujesz ostrzejszego formatowania + naprawy.")

'''
Ostrzejsze wymuszenie + „repair prompt”
To najprostszy wzorzec, który często realnie działa w aplikacjach: prosisz o JSON, parsujesz, 
jeśli fail prosisz o poprawę TYLKO JSON.
'''

import json

def get_json_or_repair(user_task: str, temperature: float = 0.0, max_repairs: int = 1):
    strict = (
        "Return ONLY valid JSON. "
        "No markdown. No code fences. No comments. No trailing text."
    )
    r = chat_once(user_task + "\n\n" + strict, temperature=temperature)
    txt = r["text"]
    for _ in range(max_repairs + 1):
        try:
            return json.loads(txt), r
        except json.JSONDecodeError as e:
            repair = (
                strict
                + f"\nThe previous output was not valid JSON. Fix it.\n"
                + f"JSON error: {str(e)}\n"
                + "Previous output:\n"
                + txt
            )
            r = chat_once(repair, temperature=0.0)
            txt = r["text"]
    raise ValueError("Failed to obtain valid JSON")

task = (
    "Classify the sentiment of the text as positive, negative, or neutral.\n"
    "Return JSON with exactly one key: sentiment.\n"
    "Text: I love how easy this app is to use!"
)
data, meta = get_json_or_repair(task, temperature=0.0, max_repairs=2)
print("JSON:", data)
print("Tokens:", (meta["usage"] or {}).get("total_tokens"))

'''
Walidacja wartości (mini-schemat)
JSON ma być nie tylko parsowalny, ale poprawny semantycznie. 
!!! W finalnym projekcie użyjemy biblioteki Pydantic, 
aby walidować nie tylko czy to jest JSON, 
ale czy ma odpowiednie pola (np. czy sentiment to faktycznie jedno ze słów: positive, negative, neutral).'''

from pydantic import BaseModel
from typing import Literal

class SentimentOut(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

task = """
Classify the sentiment of the text. Return ONLY JSON with:
{"sentiment": "positive"|"negative"|"neutral"}
Text: I love how easy this app is to use! 
"""
r = chat_once(task, temperature=0.0)
obj = SentimentOut.model_validate_json(json_data=r["text"])
print(obj.model_dump())

'''
Gdyby pojawiły się problemy z tym, że model zwraca dodatkowe frazy tak, jak ```json 
można wykorzystać wyrażenia regularne'''
import re
text = r["text"]
match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
if match:
    text = match.group(1).strip()
'''
r'' – Oznaczenie raw string; traktowac backslash (\) jako zwykły znak tekstowy (kluczowe w regexach).
```json – Dopasowuje dosłownie ciąg znaków: trzy grawisy i słowo "json".
\s – Oznacza dowolny biały znak (spacja, tabulator, nowa linia).
* – Oznacza "zero lub więcej razy" (dotyczy znaku stojącego przed nim).
( ... ) – Grupa przechwytująca; tylko to, co znajdzie się wewnątrz nawiasów, zostanie zwrócone jako wynik.
[ ... ] – Definicja zbioru znaków.
\S – Oznacza dowolny znak, który nie jest białym znakiem.
[\s\S] – Kombinacja powyższych: "znak biały LUB znak niebiały". To techniczny sposób na dopasowanie absolutnie każdego znaku, włącznie z enterami (czego zwykła kropka . nie robi).
*? – Kwantyfikator leniwy (non-greedy). Pobiera "zero lub więcej znaków", ale zatrzymuje się najszybciej jak to możliwe przed następnym dopasowaniem (czyli przed zamknięciem ```).
``` – Dopasowuje dosłownie trzy grawisy kończące blok.'''


# ================ Wymuszenie JSON przez API ===================
'''
client = genai.Client(api_key=GOOGLE_API_KEY)
config = genai.types.GenerateContentConfig(
    system_instruction=system,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=max_output_tokens,
    stop_sequences=["<END>"],
    response_mime_type="application/json",
    response_schema=Sentiment,
)
# lub
config = genai.types.GenerateContentConfig(
    system_instruction=system,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_output_tokens=max_output_tokens,
    stop_sequences=["<END>"],
    response_mime_type="application/json",
    response_json_schema={
        "type": "object",
        "properties": {
            "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                }
            },
        "required": ["sentiment"]
    }
)

resp = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt,
    config=config,
)
'''
