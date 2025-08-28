import os, glob, uuid, yaml
from typing import List
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document as DocxDocument

from chunkers import chunk_text   # ✅ absolute import

def _load_settings():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])

def _read_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _load_files(raw_dir: str) -> List[str]:
    files = []
    for pat in ["*.pdf", "*.docx", "*.txt", "*.md"]:
        files += glob.glob(os.path.join(raw_dir, pat))
    return files

def main():
    settings = _load_settings()
    raw_dir = settings['paths']['raw_docs']
    vec_dir = settings['paths']['vectorstore']
    os.makedirs(vec_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=vec_dir)
    embed_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model_name)
    coll = client.get_or_create_collection(name='portops', embedding_function=ef)

    files = _load_files(raw_dir)
    if not files:
        print(f"No documents found in: {raw_dir}")
        return

    for fpath in tqdm(files, desc="Ingesting"):
        ext = os.path.splitext(fpath)[1].lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(fpath)
            elif ext == ".docx":
                text = _read_docx(fpath)
            else:
                text = _read_txt(fpath)
        except Exception as e:
            print(f"Failed to read {fpath}: {e}")
            continue

        if not text.strip():
            print(f"Empty file (skipped): {fpath}")
            continue

        chunks = chunk_text(
            text,
            settings['retrieval']['chunk_size'],
            settings['retrieval']['chunk_overlap']
        )
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": os.path.basename(fpath), "path": fpath} for _ in chunks]
        coll.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    print("✅ Ingest complete. Vector store at:", vec_dir)

if __name__ == "__main__":
    main()
