import numpy as np

def softmax(logits):
    x = np.asarray(logits, dtype=float)     # Konwersja wejścia na tablicę float
    x = x - np.max(x)                       # Stabilizacja numeryczna: odejmij max, by uniknąć overflow przy exp
    e = np.exp(x)                           # Oblicz funkcję wykładniczą dla każdego elementu
    return e / np.sum(e)                    # Normalizacja: podziel przez sumę, aby uzyskać prawdopodobieństwa (0-1)

def softmax_with_temperature(logits, temperature: float):
    if temperature <= 0:                    # Walidacja: temperatura musi być dodatnia
        raise ValueError("temperature must be > 0")
    x = np.asarray(logits, dtype=float) / float(temperature) # Skalowanie logitów (temp < 1 wyostrza, temp > 1 spłaszcza)
    return softmax(x)                       # Zastosuj softmax na przeskalowanych wartościach

def top_k_mask(probs, top_k: int | None):
    p = np.asarray(probs, dtype=float)      # Upewnij się, że pracujemy na tablicy numpy
    if top_k is None or top_k <= 0 or top_k >= len(p): # Jeśli brak limitu K lub limit niepoprawny...
        return np.ones_like(p, dtype=bool)  # ...zwróć maskę przepuszczającą wszystko (wszystkie True)
    
    idx = np.argpartition(-p, top_k - 1)[:top_k] # Znajdź indeksy K największych elementów (szybkie sortowanie częściowe)
    mask = np.zeros_like(p, dtype=bool)     # Inicjalizacja maski samymi wartościami False
    mask[idx] = True                        # Ustaw True tylko dla znalezionych indeksów Top-K
    return mask

def top_p_mask(probs, top_p: float):
    p = np.asarray(probs, dtype=float)      # Konwersja na tablicę numpy
    if top_p >= 1.0:                        # Jeśli próg P to 100%...
        return np.ones_like(p, dtype=bool)  # ...przepuść wszystko
    
    order = np.argsort(-p)                  # Pobierz indeksy sortując od największego prawdopodobieństwa
    csum = np.cumsum(p[order])              # Oblicz sumę skumulowaną na posortowanych wartościach
    keep = csum <= top_p                    # Zaznacz elementy mieszczące się w progu sumy (Nucleus)
    
    # zapewnij, że przynajmniej 1 token zostaje
    keep[0] = True                          # Zabezpieczenie: zawsze zachowaj najlepszy token, nawet jeśli przekracza próg
    mask = np.zeros_like(p, dtype=bool)     # Inicjalizacja pustej maski
    mask[order[keep]] = True                # Przenieś zaznaczenia 'keep' na oryginalne pozycje w tablicy
    return mask

def renorm(probs, mask):
    p = np.asarray(probs, dtype=float) * mask.astype(float) # Wyzeruj prawdopodobieństwa odrzuconych tokenów (False * x = 0)
    s = p.sum()                             # Oblicz nową sumę pozostałych prawdopodobieństw
    return p / s if s > 0 else p            # Podziel przez nową sumę, aby całość znów wynosiła 1.0

def sample_next_token(
    logits,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()    # Użyj podanego generatora losowego lub utwórz domyślny
    base = softmax(logits)                  # Oblicz bazowy softmax (tylko do podglądu)
    scaled = softmax_with_temperature(logits, temperature=temperature) # Zastosuj temperaturę do logitów
    
    mask = top_k_mask(scaled, top_k) & top_p_mask(scaled, top_p) # Połącz filtry: token musi spełniać OBA warunki (AND)
    
    filtered = renorm(scaled, mask)         # Odrzuć niechciane tokeny i zrenormalizuj resztę
    idx = rng.choice(len(filtered), p=filtered) # Wylosuj indeks tokena zgodnie z nowym rozkładem prawdopodobieństwa
    return idx, base, scaled, filtered, mask # Zwróć wylosowany indeks i dane diagnostyczne

def fmt(a): return " ".join(f"{v:0.3f}" for v in a) # Formatowanie pomocnicze: wyświetl tablicę z 3 miejscami po przecinku

# --- CZĘŚĆ WYKONAWCZA ---
logits = [2.0, 1.0, 0.0, -0.5, -1.0]        # Przykładowe wejście (punkty przypisane słowom)
idx, base, scaled, filtered, mask = sample_next_token(logits, temperature=0.3, top_p=0.95, top_k=None) # Uruchom symulację
print("logits:   ", logits)
print("softmax:  ", fmt(base))              # Prawdopodobieństwa bez temperatury
print("temp:     ", fmt(scaled))            # Prawdopodobieństwa po wyostrzeniu (temp=0.3)
print("mask:     ", mask)                   # Które tokeny przeszły przez filtry Top-P/Top-K
print("filtered: ", fmt(filtered))          # Finalne prawdopodobieństwa użyte do losowania
print("sampled idx:", idx)                  # Wynik losowania