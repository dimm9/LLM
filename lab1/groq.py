import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE_URL = os.getenv("GROQ_BASE_URL")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

SYSTEM_PROMPT = "Odpowiadaj po polsku i zwięźle."
USER_PROMPT = "Podaj 3 krótkie pomysły na szybkie dania."

#=========Groq przez OpenAI-compatible (Responses API)=========

client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)

# temperature=0.0 + top_p=1 -> prawie deterministycznie
def groq_generate(prompt : str, system : str = "You are a helpful assistant", temperature : float = 0.0, top_p : float = 1.0, max_output_tokens : int = 256) -> dict:
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
    return {
        "text": response.output_text,
        "latency_s": round(dt, 3),
        "usage": usage_dict,
    }

out = groq_generate(USER_PROMPT, SYSTEM_PROMPT, temperature=0.0, top_p=0.1, max_output_tokens=180)

print(out["text"])
print(out["usage"])
print(out["latency_s"])

client.close()

#==========Groq przez Chat Completions (w nowych projektach warto trzymać się Responses)======
client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)

def groq_generate(prompt : str, system : str = "You are a helpful assistant", temperature : float = 0.0, top_p : float = 1.0, max_output_tokens : int = 256) -> dict:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
    )
    dt = time.perf_counter() - t0
    choice = r.choices[0]
    usage = getattr(r, "usage", None)
    usage_dict = None if usage is None else usage.model_dump()
    return {"text": choice.message.content, "latency_s": round(dt, 3), "usage": usage_dict}

out = groq_generate(USER_PROMPT, SYSTEM_PROMPT, temperature=0.7, top_p=0.8, max_output_tokens=180)

print(out["text"])
print(out["usage"])
print(out["latency_s"], "s")

client.close()
