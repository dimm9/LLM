import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

SYSTEM_PROMPT = "Odpowiadaj po polsku i zwięźle."
USER_PROMPT = "Podaj 3 krótkie pomysły na szybkie dania."

#=========Gemini przez Google GenAI SDK (google-genai)=========

gclient = genai.Client(api_key=GOOGLE_API_KEY)

def gemini_generate(prompt : str, system : str = "You are a helpful assistant", temperature : float = 0.0, top_p : float = 1.0, top_k : int = 40, max_output_tokens : int = 256) -> dict:
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
    #usage - informacja o zużyciu zasobów (tokenów) przez model podczas generowania odpowiedzi
    usage = getattr(response, "usage_metadata", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    } if usage is not None else None

    text = getattr(response, "text", None)
    return {
        "text": text if text is not None else str(response),
        "latency_s": round(dt, 3),
        "usage": usage_dict,
    }

out = gemini_generate(USER_PROMPT, SYSTEM_PROMPT, temperature=0.0, top_p=0.1, max_output_tokens=180)

print(out["text"])
print(out["usage"])
print(out["latency_s"], "s")
