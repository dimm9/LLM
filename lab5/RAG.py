'''
Złożymy kompletny potok przetwarzania danych:
    Hybrid Retrieval: Połączymy precyzję słów kluczowych (BM25) ze zrozumieniem kontekstu (Embeddings).
    Reranking: Użyjemy modelu Cross-Encoder, aby posortować wyniki lepiej niż robi to zwykła baza wektorowa.
    Generator (LLM): Nauczymy model odpowiadać wyłącznie na podstawie dostarczonych źródeł i cytować je
'''
import faiss
from rank_bm25 import BM25Okapi

import lab1, lab2, lab3, lab4
import os, glob, pandas as pd
from pypdf import PdfReader

from lab2.chat_once import chat_once
from lab4.embeddings import simple_chunk, embed_texts, df, embedder


def load_pdf(path):
    rows = []; r = PdfReader(path)
    for i, p in enumerate(r.pages, start=1):
        try: txt = p.extract_text() or ""
        except Exception: txt = ""
        rows.append({"source": os.path.basename(path), "page": i, "text": txt})
    return rows

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return [{"source": os.path.basename(path), "page": 1, "text": txt}]

def load_md(path): return load_txt(path)
#wczytanie dokumentów tekstowych z określonego folderu
def load_corpus(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    rows = []
    for fp in glob.glob(os.path.join(data_dir, "*")):  # Szuka w folderze wskazanym przez zmienną (domyślnie "data" np data/*)
        l = fp.lower()
        if l.endswith(".pdf"): rows += load_pdf(fp)
        elif l.endswith(".txt"): rows += load_txt(fp)
        elif l.endswith(".md"): rows += load_md(fp)
    if not rows:
        rows = [{"source": "demo.md", "page": 1, "text": "RAG łączy retrieval z generacją."},
                {"source": "demo.md", "page": 2, "text": "Embeddingi to wektory semantyczne; podobieństwo kosinusowe."}]
    return pd.DataFrame(rows)

def make_chunks(df, chunk_chars=800, overlap=120):
    rows = []
    for _, r in df.iterrows(): # idx ignorujemy
        for k, (a, b, txt) in enumerate(simple_chunk(r["text"],chunk_chars,overlap)):
            if txt.strip():  # usuwa spacje
                rows.append({"source": r["source"], "page": r["page"], "chunk_id": k, "start": a, "end": b})
    return pd.DataFrame(rows)

docs_df = load_corpus('./data')
chunks_df =make_chunks(docs_df, 800, 120)
print("chunks: ", len(chunks_df))

# Zaktualizujmy również naszą bazę wektorową:
embs=embed_texts(df["chunk"].tolist())   # zamienia listę chunków na macierz liczb (wektory)
index=faiss.IndexFlatIP(embs.shape[1]) # tworzy pusty indeks w bibliotece FAISS (biblioteka od Facebooka/Meta do szybkiego wyszukiwania podobieństw
index.add(embs) # Wrzuca wyliczone wektory do indeksu

def retrieve_dense(query, k=5):  # Ta funkcja jest wywoływana za każdym razem, gdy zadajesz pytanie
    qv=embed_texts([query], batch_size=1)  # zamienia pytanie (tekst) na wektor (qv – query vector)
    scores, idxs=index.search(qv, k)  # Pyta indeks FAISS: "Znajdź mi k (domyślnie 5) wektorów najbardziej podobnych do mojego pytania
    # scores: Jak bardzo są podobne (im wyższa liczba, tym lepsze dopasowanie).
    # idxs: Numery wierszy (indeksy) znalezionych fragmentów w bazie
    return [(float(scores[0][i]), df.iloc[idxs[0][i]].to_dict()) for i in range(k)]   # zwraca listę krotek: (ocena_dopasowania, {treść_dokumentu})

'''
Architektura: lejek RAG
Nasze rozwiązanie przypomina lejek. Na każdym etapie zmniejszamy liczbę dokumentów, ale zwiększamy precyzję.
    Retriever (szybki, szeroki): Przeszukuje tysiące dokumentów w milisekundach. Używamy tu FAISS (wektory) i BM25 (słowa kluczowe). Pobieramy np. Top-50 wyników.
    Reranker (wolny, precyzyjny): Dokładnie czyta każdą parę Pytanie-Dokument i ocenia, czy faktycznie do siebie pasują. Zostawia nam Top-5 „perełek”.
    Generator (LLM): Dostaje te 5 fragmentów i pisze odpowiedź końcową.
'''

# ==================================== Krok 1: Hybryda (RRF)
'''
Samo wyszukiwanie wektorowe to za mało bo wektory czasem gubią szczegóły (np. numery seryjne, rzadkie nazwiska). 
Z kolei BM25 nie rozumie synonimów. Rozwiązaniem jest RRF (Reciprocal Rank Fusion). 
To algorytm, który bierze ranking z FAISS i ranking z BM25, a następnie skleja je w jeden, sprawiedliwy wynik.
'''
# Bierze słownik scores (gdzie kluczem jest ID dokumentu, a wartością zsumowane punkty RRF)
def sorted_by_score(scores):
    # Sortuje malejąco (reverse=True) według liczby punktów (item[1])
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

# Funkcja ignoruje surowe oceny punktowe i sumuje odwrotności miejsc zajętych w obu rankingach, co promuje dokumenty pojawiające się wysoko na obu listach jednocześnie.
def rrf_fuse(dense_result, sparse_result, k=60):
    scores = {}
    # Dla każdego wyniku dodajemy punkty odwrotnie proporcjonalne do jego pozycji
    for rank, (score, doc_id) in enumerate(dense_result):
        scores[doc_id] += 1.0 / (k + rank)

    for rank, (score, doc_id) in enumerate(sparse_result):
        scores[doc_id] += 1.0 / (k + rank)
    return sorted_by_score(scores)


# ==================================== Krok 2: Pakowanie kontekstu i cytowania
'''
„doklejamy” wiedzę do promptu. Musimy być sprytni:
    Limity: Nie możemy wrzucić 100 stron PDF-a, bo przekroczymy limit tokenów modelu.
    Różnorodność: Jeśli Top-5 wyników pochodzi z tej samej strony jednego dokumentu, nasza odpowiedź będzie płaska. Warto więc wprowadzić limit max_per_source.
    Cytowania: Model ma zakaz wymyślania. Mysi wykazywać źródła.
Takie podejście nazywa sie Grounding (ugruntowaniem) - Jeśli model nie znajdzie odpowiedzi w dostarczonych fragmentach, 
ma się przyznać do niewiedzy, zamiast halucynować.
'''

SYSTEM_RULES=(
    "You are a factual assistant. Answer ONLY using the provided context snippets. "
    "If missing, reply 'Nie wiem – brak informacji w źródłach.' Include citations like [1],[2].")
# Funkcja przygotowująca tekst dla AI (usuwa nadmiar i formatuje).
def pack_context(hits, max_per_source=2, max_chars=2000):
    per = {}; ordered = [] # Słownik do liczenia, ile razy użyliśmy danej strony (per) i lista na wybrane fragmenty (ordered).
    for _, rec in hits:  # # Pętla po znalezionych wynikach (hits). Ignorujemy ocenę (_), bierzemy dane rekordu
        key = (rec["source"], rec["page"])  # Tworzymy unikalny klucz: (nazwa pliku, numer strony)
        per.setdefault(key, 0) # Jeśli klucza nie ma w słowniku, ustawiamy licznik na 0
        if per[key] < max_per_source: # jeśli z tej strony wzięliśmy mniej niż 2 fragmenty, to bierzemy ten.
            ordered.append(rec); per[key] += 1 # Dodajemy rekord do listy wybranych i zwiększamy licznik użycia tej strony.
    cites = []; parts = []  # Listy pomocnicze: na metadane do cytatów (cites) i na tekst do promptu (parts).
    for i, rec in enumerate(ordered, start=1): # Numerujemy wybrane fragmenty od 1 (i=1, i=2...).
        # Zapisujemy, że numer [i] to konkretny plik i strona (potrzebne, by wyświetlić źródła użytkownikowi).
        cites.append({"n": i, "source": rec["source"], "page": rec["page"], "chunk_id": rec["chunk_id"]})
        # Doklejamy numer w nawiasie do treści fragmentu, np.: "[1] Treść fragmentu...".
        parts.append(f"[{i}] {rec['chunk']}")
    ctx="\n\n".join(parts)  # Łączymy wszystkie fragmenty w jeden długi napis, oddzielając je pustymi liniami.
    return (ctx[:max_chars], cites) #  Zwracamy kontekst przycięty do limitu znaków (np. 2000) oraz listę cytatów.

def answer_with_api(question, hits):
    ctx, cites = pack_context(hits)   # Wywołujemy funkcję wyżej, by sformatować kontekst i pobrać cytaty
    prompt = "Question: " + question + "\n\nContext: \n" + ctx + "\n\nAnswer in Polish with citations [n]." # # Budujemy ostateczny tekst (Prompt) dla modelu: Pytanie + Kontekst
    return chat_once(prompt, system=SYSTEM_RULES, max_output_tokens=256, temperature=0.0), cites # zwracamy wynik + źródła

print(answer_with_api('Jakie ery wyróżniamy?', retrieve_dense('RAG', k=4)))
print(answer_with_api('Co to jest RAG?', retrieve_dense('RAG', k=4)))


# ====================== Krok 3: Reranking
'''
Baza wektorowa liczy podobieństwo „matematyczne” (iloczyn skalarny). 
Model Cross-Encoder działa inaczej: widzi pełne pytanie i pełną odpowiedź jednocześnie. 
Jest znacznie mądrzejszy, ale też wolniejszy,  dlatego używamy go tylko do posortowania garstki kandydatów wybranych przez Retrievera.
'''
# Przykład Retrieval + Reranking
from typing import List, Tuple, Any, Dict
import re
import numpy as np
DOCS = [
    {'id':'d1','text':'Embeddings are vector representations of text used for semantic search.'},
    {'id':'d2','text':'BM25 is a bag-of-words retrieval algorithm based on term frequency.'},
    {'id':'d3','text':'RAG combines retrieval and generation to ground LLM outputs.'},
    {'id':'d4','text':'Cross-encoders score query+doc pairs with a deeper transformer for reranking.'},
    {'id':'d5','text':'Embedding models like all-MiniLM produce compact vectors.'},
]

# Funkcja czyszcząca tekst: wyciąga tylko litery i cyfry, zamieniając je na małe litery.
def tok(x): return re.findall(r"[a-ząćęłńóśźż0-9]+", x.lower())

# Tworzy listę samych tekstów z bazy dokumentów.
texts = [d['text'] for d in DOCS]
# Zamienia teksty na wektory (embeddingi), normalizuje je i konwertuje do formatu float32.
embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
# Tworzy indeks FAISS (wyszukiwarkę wektorową) i dodaje do niego wektory.
idx = faiss.IndexFlatIP(embs.shape[1]);
idx.add(embs)
# Przygotowuje teksty dla BM25, dzieląc je na listy słów.
bm25_corpus = [tok(t) for t in texts]
# Inicjalizuje algorytm BM25 na przygotowanym korpusie słów.
bm25 = BM25Okapi(bm25_corpus)

# Funkcja wyszukująca semantycznie k-najlepszych dokumentów.
def retrieve_dense(query: str, k: int = 5):
    # Zamienia zapytanie na wektor w taki sam sposób jak dokumenty.
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    # Przeszukuje indeks FAISS znajdując najbliższe wektory.
    scores, ids = idx.search(qv, k)
    # Zwraca listę wyników jako pary (ocena, dokument).
    return [(float(scores[0][i]), DOCS[ids[0][i]]) for i in range(min(k, len(ids[0])))]

# Zmienna sterująca użyciem Cross-Encodera (domyślnie wyłączony).
USE_CROSS_ENCODER = False
try:
    # Próbuje zaimportować klasę CrossEncoder.
    from sentence_transformers import CrossEncoder

    # Ładuje precyzyjny model do rerankingu (oceniania par).
    CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    # Jeśli sukces, włącza flagę użycia.
    USE_CROSS_ENCODER = True
except Exception:
    # W razie błędu importu ustawia brak modelu.
    CROSS_ENCODER = None
    USE_CROSS_ENCODER = False

# Funkcja rerankingu (ponownego sortowania) kandydatów.
def rerank(query: str, candidates: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
    # Jeśli mamy Cross-Encoder, używamy go (jest dokładniejszy).
    if USE_CROSS_ENCODER and CROSS_ENCODER is not None:
        # Tworzy pary (pytanie, tekst) dla każdego kandydata.
        pairs = [(query, c[1]['text']) for c in candidates]
        # Model ocenia każdą parę i zwraca jej wynik.
        scores = CROSS_ENCODER.predict(pairs)
        # Przypisuje nowe wyniki do dokumentów.
        scored = [(float(s), c[1]) for s, c in zip(scores, candidates)]
        # Sortuje dokumenty malejąco według nowej oceny.
        scored.sort(key=lambda x: x[0], reverse=True)
        # Zwraca posortowaną listę.
        return scored

    # Opcja zapasowa (gdy brak Cross-Encodera): ręczne obliczenie iloczynu skalarnego.
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')[0]
    out = []
    # Iteruje po liście kandydatów.
    for score, doc in candidates:
        # Znajduje indeks numeryczny dokumentu w bazie DOCS.
        idx_doc = next((i for i, d in enumerate(DOCS) if d['id'] == doc['id']), None)
        # Zabezpieczenie: jeśli nie znajdzie, zostawia starą ocenę.
        if idx_doc is None:
            s = score
        else:
            # Oblicza iloczyn skalarny (podobieństwo) między pytaniem a dokumentem.
            s = float(np.dot(qv, embs[idx_doc]))
        # Dodaje krotkę (wynik, dokument) do listy wyjściowej.
        out.append((s, doc))
    # Sortuje listę wyjściową malejąco.
    out.sort(key=lambda x: x[0], reverse=True)
    # Zwraca wynik opcji zapasowej.
    return out

query = "what are embeddings used for?"
print('=== BM25 top-3 ===')
# Oblicza wyniki BM25 dla zapytania.
bm = bm25.get_scores(tok(query))
# Znajduje indeksy 3 najlepszych wyników.
bm_idx = np.argsort(bm)[::-1][:3]
for i in bm_idx:
    print(texts[i])

print('\n=== Dense top-3 ===')
# Pobiera 3 najlepszych kandydatów z FAISS.
cand = retrieve_dense(query, k=3)
# Wypisuje ich ocenę i treść.
for s, d in cand:
    print(s, d['text'])

print('\n=== Reranked ===')
# Przesortowuje kandydatów używając rerankera.
rr = rerank(query, cand)
# Wypisuje ostateczne wyniki (ocena i treść).
for s, d in rr:
    print(s, d['text'])
