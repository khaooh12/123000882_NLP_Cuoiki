import os
import pickle
import sys

print("Python executable:", sys.executable)
print("Checking model files...")

paths = [
    "models/lda_model/lda_model.model",
    "models/lda_model/dictionary.dict",
    "models/lda_model/corpus.pkl",
    "models/bertopic_model/fallback/model.pkl",
    "models/metrics.json"
]

for p in paths:
    exists = os.path.exists(p)
    print(f"  {p}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        print(f"    Size: {os.path.getsize(p)} bytes")

# Try to inspect LDA dictionary/corpus size if exists
if os.path.exists("models/lda_model/dictionary.dict"):
    try:
        from gensim.corpora import Dictionary
        dct = Dictionary.load("models/lda_model/dictionary.dict")
        print("LDA Dictionary size:", len(dct))
    except Exception as e:
        print("Error loading dictionary:", e)

if os.path.exists("models/lda_model/corpus.pkl"):
    try:
        with open("models/lda_model/corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
        print("LDA Corpus size:", len(corpus))
    except Exception as e:
        print("Error loading corpus:", e)
