'''
Embedding to proces zmiany tekstu (zdania, akapitu) na długi ciąg liczb (wektor), np. [0.12, -0.98, 0.55, ...].
w tej matematycznej przestrzeni, teksty o podobnym znaczeniu znajdują się blisko siebie.
    „Król” – „Mężczyzna” + „Kobieta” ~ „Królowa”
    „Pies” będzie blisko „Szczeniak”, nawet jeśli nie dzielą zbyt wiele liter.
Semantic Search; szukamy po znaczeniu, a nie po słowach.
'''


# ================ KROK 1: Dane i chunking
'''
LLM ma limit „pamięci” - okno kontekstowe(Context Window). Trzeba podzielić wiedzę na mniejsze fragmenty – chunki.
'''
import pandas as pd
import random

SEED = 42
random.seed(SEED)

TOPICS = {
    "ai": [
        "Large language models predict the next token using transformer architectures.",
        "Embeddings map text into dense vectors enabling semantic search.",
        "RAG combines retrieval with generation to ground responses."
    ],
    "sport": [
        "Marathon training plans balance long runs and recovery days.",
        "Strength training improves running economy and power.",
        "Interval sessions develop speed and lactate threshold."
    ],
    "cooking": [
        "Sourdough starter needs regular feeding to stay active.",
        "Sous vide cooking keeps precise temperatures for tenderness.",
        "Spices bloom in hot oil enhancing aroma and flavor."
    ],
    "geo": [
        "Rivers shape valleys through erosion and sediment transport.",
        "Plate tectonics explains earthquakes and mountain building.",
        "Deserts form where evaporation exceeds precipitation."
    ],
    "health": [
        "Sleep supports memory consolidation and hormonal balance.",
        "Aerobic exercise benefits cardiovascular health and VO2 max.",
        "Protein intake supports muscle repair and satiety."
    ]
}

def sync_docs(n_per_topic=40):
    """
    Metoda synth_docs generuje syntetyczny zbiór dokumentów do testowania algorytmów wyszukiwania semantycznego.
    Dla każdego tematu z listy TOPICS tworzy określoną liczbę dokumentów (n_per_topic).
    Każdy dokument zawiera tekst złożony z 2 losowo wybranych zdań z danego tematu oraz informację o temacie i unikalny identyfikator.
    Tak przygotowany korpus pozwala porównywać skuteczność różnych metod wyszukiwania na kontrolowanych danych, gdzie znana jest przynależność dokumentów do tematów.
    :param n_per_topic:
    :return:
    """
    docs = []
    for topic, seed in TOPICS.items():
        for i in range(n_per_topic):
            base = random.choice(seed)
            noise = random.choice(seed)
            txt = f"{base} {noise} ({topic} #{i})"
            docs.append({"docs_id": f"{topic}-{i}", "topic": topic, "text": txt})
    return docs

DOCS = sync_docs(40)

'''
funkcja do dzielenia tekstu.
parametr overlap (zakładka) zapobiega sytuacji, w której ważne słowo kluczowe, paragraf, akapit zostaje przecięty na granicy dwóch chunków.
'''
CHUNK_SIZE=280
OVERLAP = 40

def simple_chunk(text, chunk_chars=280, overlap=40):
    out = []
    i = 0
    while i < len(text):
        j = min(chunk_chars+i, len(text))
        out.append((i, j, text[i:j]))
        if j == len(text): break
        i = max(0, j-overlap)
    return out

def build_chunks(docs, chunk_chars=280, overlap=40):
    rows = []
    for d in docs:
        for chunk_id, (i, j, txt_chunk) in enumerate(simple_chunk(d['text'], chunk_chars, overlap)):
            rows.append({"docs_id": d["docs_id"], "topic": d["topic"], "chunk_id": chunk_id, "start": i, "end": j, "chunk": txt_chunk})
    return pd.DataFrame(rows)

df = build_chunks(DOCS, CHUNK_SIZE, OVERLAP)


# ============ KROK 2: Dense Retrieval z FAISS
'''zamiana chunkow na wektory i wrzucenie do indeksu FAISS (Facebook AI Similarity Search). '''
import time, faiss
from sentence_transformers import SentenceTransformer

# ladowanie modelu tlumaczacego tekst na liczby
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# zamiana chunkow na macierz(liste wektorow)
def embed_texts(texts, batch_size=64):
    return embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

chunks = df["chunk"].tolist()
embeddings = embed_texts(chunks, 64)

# budowanie indeksu (baza wektorowa w RAM)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# wyszukiwanie
# Retriever - zwróci 3 najbardziej pasujące semantycznie fragmenty tekstu.
query_vector = embedder.encode(["What improves running?"], normalize_embeddings=True)
scores, ids = index.search(query_vector, k=3)


# =================== KROK 3: BM25 - klasyczne wyszukiwanie słów kluczowych; wyszukiwanie nazw własnych, kodów produktów czy specyficznych fraz
# Hybrid Search: Wynik = 0.7 * VectorSearch + 0.3 * KeywordSearch.
import re
from rank_bm25 import BM25Okapi
# szukasz słowa "bieganie" => BM25 znajdzie dokumenty, gdzie to słowo występuje często

def tokenize(text : str): # Algorytm BM25 nie rozumie całych zdań. Musi dostać listę pojedynczych słów. Ta funkcja zamienia zdanie na listę słów.
    return re.findall(r"[a-z0-9]+", text.lower())   # "Cześć, jak się masz? 100%!" zrobi listę ['czesc', 'jak', 'sie', 'masz', '100']

bm25_corpus : list[list[str]] = [tokenize(c) for c in chunks]
bm25 = BM25Okapi(bm25_corpus)  # algorytm oblicza statystyki dla każdego słowa w dokumentach


# =================== KROK 4 Ewaluacja
import numpy as np

def retrieve_dense(query: str, k : int=5):
    q = embed_texts([query], batch_size=1) # Zamiana słów na liczby q - wektor (ciąg liczb), który reprezentuje "sens" tego pytania w przestrzeni matematycznej
    scores, idxs = index.search(q, k) # porównuje wektor pytania (q) ze wszystkimi wektorami w bazie (index).
    return [(float(scores[0][i]), df.iloc[idxs[0][i]].to_dict()) for i in range(k)] # Pętla for przechodzi przez k najlepszych wyników.
    # df.iloc[idxs[0][i]]: Wyciąga z ramki danych (df) konkretny wiersz odpowiadający znalezionemu indeksowi. Funkcja zwraca listę par: (wynik_liczbowy, treść_dokumentu)

def retrieve_bm25(query: str, k : int=5):
    toks = tokenize(query)  # Zamienia pytanie (np. "Jaki kod produktu?") na listę czystych słów (np. ['jaki', 'kod', 'produktu']
    scores = bm25.get_scores(toks) # Sprawdza każdy dokument w bazie i przydziela mu punkty. Jeśli dokument zawiera rzadkie i ważne słowa z zapytania (np. "kod"), dostaje wysoki wynik
    idxs = np.argsort(scores)[::-1][:k] # Sortuje wyniki rosnąco, Odwraca listę (aby najwyższe wyniki były na początku). Odcina listę, zostawiając tylko k najlepszych indeksów
    return [(float(scores[i]), df.iloc[i].to_dict()) for i in idxs] # Bierze numery zwycięskich wierszy (idxs), wyciąga dla nich tekst z bazy danych (df) i zwraca je w czytelnej formie wraz z ich punktacją.

print(retrieve_dense("What improves running economy?", 3)[0])
print(retrieve_bm25("What improves running economy?", 3)[0])

'''
Sprawdzamy 4 kluczowe metryki:
    Recall@k: Czy poprawna odpowiedź znalazła się w ogóle w top-5 wyników? (często najważniejsze dla RAG)
    Precision@k: Ile śmieci (niepoprawnych wyników) zwrócił system?
    MRR (Mean Reciprocal Rank): Jak wysoko na liście była poprawna odpowiedź? (lepiej być 1. niż 5.)
    nDCG: Bardziej zaawansowana metryka jakości rankingu.
'''
import math

GOLDEN_SET = [
    ("How do transformers predict tokens?", "ai"),
    ("What is an embedding used for?", "ai"),
    ("How does RAG work?", "ai"),
    ("How to train for a marathon?", "sport"),
    ("What improves running economy?", "sport"),
    ("What is a threshold workout?", "sport"),
    ("How to feed sourdough starter?", "cooking"),
    ("Why sous vide is precise?", "cooking"),
    ("How to bloom spices?", "cooking"),
    ("How do rivers shape valleys?", "geo"),
    ("What causes earthquakes?", "geo"),
    ("Why do deserts form?", "geo"),
    ("Why is sleep important?", "health"),
    ("Benefits of aerobic exercise?", "health"),
    ("Why eat protein?", "health"),
]
def dcg(rels):
    # Oblicza Discounted Cumulative Gain: suma ocen (0 lub 1) ważona pozycją (im niżej na liście, tym mniejsza waga).
    # "i+2" w logarytmie wynika z tego, że indeksujemy od 0, a logarytm liczymy dla pozycji 1, 2... (logarytm z 1 to 0).
    return sum((rel / math.log2(i+2) for i, rel in enumerate(rels)))

def ndcg_at_k(rels, k):
    # Ogranicza listę trafień tylko do pierwszych k wyników.
    rels_k = rels[:k]
    # Tworzy idealny ranking (posortowany malejąco) dla tych konkretnych wyników (tzw. IDCG).
    ideal = sorted(rels_k, reverse=True)
    # Oblicza DCG dla idealnego rankingu (zabezpieczenie 'or 1e-9' chroni przed dzieleniem przez zero).
    denom = dcg(ideal) or 1e-9
    # Zwraca znormalizowany wynik (NDCG): stosunek rzeczywistego DCG do idealnego (wynik od 0 do 1).
    return dcg(rels_k)/denom

# Funkcja oceniająca pojedyncze zapytanie
def eval_query(q, target_topic, retriever, k=5):
    # Uruchamia wybrany retriever (dense lub bm25) i pobiera k najlepszych wyników dla pytania q.
    hits = retriever(q, k=k)
    # Tworzy listę zer i jedynek: 1 jeśli temat wyniku zgadza się z oczekiwanym (target_topic), 0 jeśli nie.
    rels = [1 if h[1]["topic"]==target_topic else 0 for h in hits]
    # Recall (uproszczony): suma trafień (ile poprawnych dokumentów znalazło się w top-k).
    rec = sum(rels) * 1.0
    # Precision: jaki procent zwróconych wyników (top-k) jest poprawny.
    prec = sum(rels)/len(rels) if rels else 0.0
    # MRR (Mean Reciprocal Rank): sprawdza pozycję PIERWSZEGO poprawnego wyniku (1/pozycja).
    rr = 0.0
    for i, r in enumerate(rels, start=1):
        if r==1: rr = 1.0/i; break # Jeśli trafienie, zapisz odwrotność pozycji i przerwij pętlę.
    # Oblicza metrykę NDCG dla listy trafień.
    ndcg = ndcg_at_k(rels, k)
    # Zwraca słownik z wynikami wszystkich czterech metryk dla tego jednego pytania.
    return {"recall@k": rec, "precision@k": prec, "mrr": rr, "ndcg@k": ndcg}

# Funkcja oceniająca cały zbiór testowy
def evaluate(golden, retriever, k=5):
    # Lista, która przechowa wyniki dla każdego pytania ze zbioru testowego.
    rows = []
    # Iteruje przez pary (pytanie, poprawny_temat) ze zbioru "złotego standardu".
    for q,t in golden:
        # Wywołuje eval_query dla każdej pary i dodaje wynik do listy.
        rows.append({"query": q, "topic": t, **eval_query(q,t,retriever,k)}) # ** użyte wewnątrz klamerek {} oznaczają rozpakowanie słownika (dictionary unpacking)
    # Zamienia listę wyników na czytelną ramkę danych Pandas.
    return pd.DataFrame(rows)

K = 5
dense_df = evaluate(GOLDEN_SET, retrieve_dense, k=K)
bm25_df = evaluate(GOLDEN_SET, retrieve_bm25, k=K)

summary = pd.DataFrame({
    # Kolumna z nazwami metryk.
    "metric": ["recall@k","precision@k","mrr","ndcg@k"],
    # Kolumna ze średnimi wynikami dla metody Dense (średnia ze wszystkich pytań).
    "dense": [dense_df[m].mean() for m in ["recall@k","precision@k","mrr","ndcg@k"]],
    # Kolumna ze średnimi wynikami dla metody BM25.
    "bm25":  [bm25_df[m].mean()  for m in ["recall@k","precision@k","mrr","ndcg@k"]],
})
print(summary)

'''
Grid Search - badanie jak zmiana wielkości chunka i innych hiperparametrów wpływa na metryki.'''


def run_setting(chunk_size, overlap, kk):
    # 1. Budowanie bazy: potnij tekst na kawałki o zadanym rozmiarze i zakładce
    dff = build_chunks(DOCS, chunk_size, overlap)

    # 2. Embedding: zamień nowe kawałki tekstu na wektory (macierz float32)
    embs = embedder.encode(dff["chunk"].tolist(), batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True).astype("float32")

    # 3. Indeksowanie: stwórz nowy indeks FAISS dla tych konkretnych wektorów
    idx = faiss.IndexFlatIP(embs.shape[1]);
    idx.add(embs)

    # 4. Definicja Retrievera: lokalna funkcja szukająca, która "widzi" aktualny indeks 'idx' i ramkę 'dff'
    def retr(q, k):
        # Zamiana pytania na wektor
        qv = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        # Wyszukanie w indeksie FAISS
        scores, ids = idx.search(qv, k)
        # Zwrócenie wyników w formacie czytelnym dla funkcji ewaluacji
        return [(float(scores[0][i]), dff.iloc[ids[0][i]].to_dict()) for i in range(k)]

    # 5. Ewaluacja: uruchom testy na zbiorze GOLDEN używając tego konkretnego retrievera
    dfres = evaluate(GOLDEN_SET, retr, k=kk)
    # 6. Wynik: zwróć średnie wartości metryk (Recall, Precision, MRR, NDCG) jako słownik
    return dfres[["recall@k", "precision@k", "mrr", "ndcg@k"]].mean().to_dict()

# --- Pętla Grid Search (szukanie najlepszych parametrów) ---
grid = []
# Sprawdzamy różne długości chunków (od małych po duże)
for cs in [50, 200, 400, 800]:
    # Sprawdzamy z zakładką (overlap) i bez niej
    for ov in [0, 80]:
        # Sprawdzamy dla różnej liczby zwracanych wyników (Top-K)
        for kk in [1, 3, 5]:
            # Uruchamiamy test dla danej trójki parametrów
            met = run_setting(cs, ov, kk)
            # Dodajemy parametry i wyniki do listy (używając **met do rozpakowania słownika wyników)
            grid.append({"chunk_size": cs, "overlap": ov, "k": kk, **met})

# Tworzymy tabelę wyników i sortujemy:
# Najpierw po 'k' (rosnąco), potem po 'recall@k' (malejąco) - żeby widzieć najlepsze wyniki dla każdego k
grid_df = pd.DataFrame(grid).sort_values(["k","recall@k"], ascending=[True, False])
print(grid_df.head(10))

# --- Logowanie wyników do pliku (MLOps) ---
# Przydatne, żeby nie stracić wyników eksperymentów po restarcie notebooka

import os
from datetime import datetime

def log_row(path: str, row: dict):
    # Sprawdzamy, czy plik już istnieje (żeby wiedzieć, czy dopisać nagłówek)
    exists = os.path.exists(path)
    import csv
    # Otwieramy plik w trybie 'append' (dopisywanie na końcu), kodowanie utf-8
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        # Jeśli plik jest nowy, najpierw zapisz nazwy kolumn
        if not exists:
            w.writeheader()
        # Zapisz wiersz z danymi
        w.writerow(row)

# Przykład użycia funkcji logowania
log_row("lab04_logs.csv", {
    "timestamp": datetime.utcnow().isoformat(), # Znacznik czasu uruchomienia
    "model_name": "all-MiniLM-L6-v2",           # Nazwa modelu embeddingów
    "index_type": "faiss-flatip",               # Typ indeksu
    "k": K,                                     # Użyte K
    "chunk_size": CHUNK_SIZE,                   # Użyty rozmiar chunka
    "overlap": OVERLAP,                         # Użyty overlap
    # Pobranie konkretnych wartości liczbowych z wcześniejszej tabeli podsumowującej (summary)
    "dense_recall@k": float(summary.loc[summary.metric=='recall@k','dense'].values[0]),
    "bm25_recall@k": float(summary.loc[summary.metric=='recall@k','bm25'].values[0]),
})

'''
zbudowaliśmy wyszukiwarkę semantyczną. Potrafimy znaleźć fragmenty tekstu pasujące do zapytania
'''









