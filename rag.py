# rag_annoy.py
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "index.ann"
CHUNKS_PATH = "chunks.npy"
GEN_MODEL = "distilgpt2"

def l2_normalize(x):
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def load_resources():
    chunks = np.load(CHUNKS_PATH, allow_pickle=True)
    #We need index dim to load Annoy
    embedder = SentenceTransformer(EMBED_MODEL)
    #dummy vector to infer dim
    dim = embedder.encode(["dim"], convert_to_numpy=True).shape[1]
    index = AnnoyIndex(dim, metric="angular")
    index.load(INDEX_PATH)
    gen = pipeline("text-generation", model=GEN_MODEL)
    return index, chunks, embedder, gen

def retrieve(index, embedder, chunks, query, k=3):
    q = embedder.encode([query], convert_to_numpy=True)
    q = l2_normalize(q)[0]
    ids, dists = index.get_nns_by_vector(q.tolist(), k, include_distances=True)
    hits = []
    for i, d in zip(ids, dists):
        if 0 <= i < len(chunks):
            hits.append((chunks[i], float(d)))
    return hits

def answer(index, embedder, chunks, gen, query, k=3, max_new_tokens=80):
    hits = retrieve(index, embedder, chunks, query, k=k)
    context = " ".join([c for c, _ in hits])
    prompt = f"Context: {context}\nQ: {query}\nA:"
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True)[0]["generated_text"]
    return out.split("A:")[-1].strip(), hits

if __name__ == "__main__":
    index, chunks, embedder, gen = load_resources()
    print(f"Annoy index loaded. #chunks={len(chunks)} | embed={EMBED_MODEL}")
    print("RAG Chatbot (Annoy) ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Bye!"); break
        try:
            ans, hits = answer(index, embedder, chunks, gen, q, k=3)
            print("\nRetrieved (angular distance):")
            for c, d in hits:
                print(f"  • {c}  (dist={d:.4f})")
            print("\nBot:", ans, "\n")
        except Exception as e:
            print("Error:", e, "\n")
