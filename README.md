# PortOps GenAI — Groq Only (Streamlit + RAG)

This is a Groq-only version of the PortOps assistant. Users supply their **own Groq API key**.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/chat_groq_app.py
```

On Streamlit Cloud, just deploy the repo and each user pastes their own Groq API key in the sidebar.

## Repo Layout
- `app/chat_groq_app.py` — Groq chat UI
- `rag/` — RAG pipeline (ingest SOPs → Chroma)
- `prompts/` — SOP-style system prompts
- `config/settings.yaml` — retrieval config


