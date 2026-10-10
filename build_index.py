import os
import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
DATA_DIR = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)

def load_docs(doc_dir=DOCS_DIR):
    paths = []
    for ext in ("*.md", "*.txt"):
        paths.extend(glob.glob(os.path.join(doc_dir, ext)))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            docs.append({"path": p, "text": text})
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return [c for c in chunks if c]

def build_and_save():
    docs = load_docs()
    if not docs:
        print("docs/ 目录下没有文档，请先放入 .md/.txt 文档。")
        return

    print(f"加载 {len(docs)} 个文档，开始切分并计算 embedding...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    metadatas = []
    embeddings = []

    for d in docs:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            metadatas.append({"source": os.path.basename(d["path"]), "chunk_id": i, "text": c})
            emb = model.encode(c, show_progress_bar=False)
            embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"向量维度 {dim}, 段落数 {len(embeddings)}")

    try:
        import faiss
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(DATA_DIR, "index.faiss"))
        print("已保存 FAISS 索引到 data/index.faiss")
    except Exception as e:
        print("faiss 写入失败，保存为 embeddings.npy：", e)
        np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print("已保存 metadata.json")

if __name__ == "__main__":
    build_and_save()
