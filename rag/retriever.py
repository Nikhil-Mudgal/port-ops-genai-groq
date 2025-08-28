# --- SQLite patch for Streamlit Cloud / Python 3.12+ ---
try:
    import pysqlite3  # type: ignore
    import sys as _sys
    _sys.modules["sqlite3"] = pysqlite3
    _sys.modules["sqlite"] = pysqlite3
except Exception:
    pass
# rag/retriever.py
import os, yaml, chromadb
from typing import List, Dict, Any
from chromadb.utils import embedding_functions

def _load_settings():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def _collection():
    settings = _load_settings()
    client = chromadb.PersistentClient(path=settings['paths']['vectorstore'])
    embed_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model_name)
    return client.get_or_create_collection(name="portops", embedding_function=ef), settings

def retrieve(query: str, k: int = None) -> List[Dict[str, Any]]:
    coll, settings = _collection()
    top_k = int(k or settings['retrieval']['top_k'])

    # IMPORTANT: Do NOT include "ids" here; Chroma will still return them in res["ids"]
    res = coll.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # ← valid keys only
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)
    ids   = res.get("ids", [[]])[0] if res.get("ids") else [None] * len(docs)  # ids usually present by default

    out: List[Dict[str, Any]] = []
    for i, txt in enumerate(docs):
        out.append({
            "id": ids[i] if i < len(ids) else None,
            "text": txt,
            "meta": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return out
