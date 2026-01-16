import csv, os
from datetime import datetime
from typing import Any, Dict

TOP_P = 0.9
TOP_K = None
TEMPERATURE = 0.2

def log_row(path: str, row: Dict[str, Any]) -> None:
    # 1. Ustalanie kolejności kolumn w Excelu
    # Sortujemy klucze alfabetycznie, żeby kolumny zawsze były w tej samej kolejności
    fieldnames = sorted(row.keys())

    # 2. Sprawdzenie, czy plik już istnieje
    exists = os.path.exists(path)

    # 3. Otwieranie pliku w trybie "append" ('a')
    # 'a' oznacza: "nie kasuj treści, tylko dopisz na końcu".
    # newline="" jest ważne w Windows, żeby nie robiło pustych linii między wierszami.
    with open(path, "a", newline="", encoding="utf-8") as f:
        # Tworzymy pisarza, który rozumie słowniki
        w = csv.DictWriter(f, fieldnames=fieldnames)

        # Jeśli plik jest nowy (nie istniał), najpierw wpisujemy nazwy kolumn (nagłówek)
        if not exists:
            w.writeheader()

        # Wpisujemy właściwe dane z tego konkretnego wywołania
        w.writerow(row)

'''
# przykład: log lokalnego wywołania
out = local_generate(
    USER_PROMPT,
    system=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_output_tokens=256,
)
row = {
    "timestamp": datetime.utcnow().isoformat(),
    "mode": "local",
    "model": LOCAL_MODEL_NAME,
    "latency_s": out["latency_s"],
    "prompt_tokens": out["usage"]["prompt_tokens"],
    "completion_tokens": out["usage"]["completion_tokens"],
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
}
log_row("lab01_logs.csv", row)
print("OK: lab01_logs.csv")
'''
