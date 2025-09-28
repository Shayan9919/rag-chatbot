# ingest_annoy.py
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "data/docs.md"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "index.ann"
CHUNKS_PATH = "chunks.npy"
N_TREES = 10  #we could make a better recall for a slower build

def load_corpus(path=CORPUS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("Corpus is empty. Add lines to data/docs.md")
    return lines

def l2_normalize(X):
    X = np.asarray(X, dtype="float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

if __name__ == "__main__":
    corpus = load_corpus()
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(corpus, convert_to_numpy=True)
    embs = l2_normalize(embs)  # angular metric ≈ cosine distance when normalized

    dim = embs.shape[1]
    index = AnnoyIndex(dim, metric="angular")
    for i, v in enumerate(embs):
        index.add_item(i, v.tolist())
    index.build(N_TREES)
    index.save(INDEX_PATH)

    np.save(CHUNKS_PATH, np.array(corpus, dtype=object))
    print(f"Annoy index built with {len(corpus)} items (dim={dim}) → {INDEX_PATH}, {CHUNKS_PATH}")
