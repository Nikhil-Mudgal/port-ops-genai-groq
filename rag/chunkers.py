from typing import List

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """
    Simple, dependency‑free splitter by characters with overlap.
    Keeps things lightweight for Streamlit Cloud.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - chunk_overlap if end - chunk_overlap > start else end
    return chunks
