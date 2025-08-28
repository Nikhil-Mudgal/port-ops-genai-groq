import yaml
import streamlit as st
from openai import OpenAI
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rag.retriever import retrieve


st.set_page_config(page_title="PortOps Chat", page_icon="⚓", layout="wide")

MODEL = "llama3-8b-8192"
TEMPERATURE = 0.7
MAX_TOKENS = 1000
SYSTEM_PROMPT = """You are PortOps, an assistant for port operations.
Answer using the retrieved SOP context below. If the context doesn’t contain the answer, say what’s missing.
Respond with:
**Summary**
**Steps**
**Exceptions/Notes**
Cite sources in parentheses like (source: <filename>)."""


# ----------------------
# Helpers
# ----------------------
def mask_email(e: str) -> str:
    if "@" not in e:
        return e
    user, domain = e.split("@", 1)
    if len(user) <= 2:
        masked_user = "*" * len(user)
    else:
        masked_user = user[0] + "*" * max(1, len(user) - 2) + user[-1]
    return f"{masked_user}@{domain}"

def looks_like_email(e: str) -> bool:
    e = e.strip()
    return "@" in e and "." in e.split("@")[-1] and " " not in e

# ----------------------
# Session State
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Port Ops is here to help you with all your Queries."}
    ]
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "api_key_submitted" not in st.session_state:
    st.session_state.api_key_submitted = False

# ----------------------
# Gate: Email + API Key Form
# ----------------------
if not st.session_state.api_key_submitted:
    st.markdown(
    "<h1 style='text-align: center;'>PortOps Chat</h1>",
    unsafe_allow_html=True
)
    st.subheader("Enter your details to start")

    with st.form("api_key_form", clear_on_submit=False):
        email = st.text_input("Email ID", value=st.session_state.user_email, placeholder="you@company.com")
        api_key = st.text_input("Groq API Key (starts with gsk_…)", type="password", value=st.session_state.api_key)
        submit = st.form_submit_button("Submit")

    if submit:
        errors = []
        if not email.strip():
            errors.append("Email is required.")
        elif not looks_like_email(email):
            errors.append("Please enter a valid email address (e.g., you@company.com).")
        if not api_key.strip():
            errors.append("Groq API key is required.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            st.session_state.user_email = email.strip()
            st.session_state.api_key = api_key.strip()
            st.session_state.api_key_submitted = True
            st.success(f"✅ Details saved successfully for **{mask_email(st.session_state.user_email)}**. You can start chatting now.")
            st.rerun()

    st.stop()

# ----------------------
# Main Chat UI (after submit)
# ----------------------

st.markdown(
    "<h1 style='text-align:center; margin-top: 0;'>PortOps Chat</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center; opacity:0.85;'>Signed in as: "
    f"<strong>{mask_email(st.session_state.user_email)}</strong></p>",
    unsafe_allow_html=True,
)

# ---------- Sidebar with bottom-pinned Clear button ----------
# CSS: make the sidebar a full-height flex column and push last child to the bottom
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stVerticalBlock {
        height: 100vh !important;
        display: flex;
        flex-direction: column;
    }
    /* The last direct child in the sidebar gets pushed to the bottom */
    [data-testid="stSidebar"] .stVerticalBlock > div:last-child {
        margin-top: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# RAG controls
_,_,_,col_rag = st.columns([1,1,1,1.35])
with col_rag:
    use_rag = st.toggle("🔎 RAG (use JNPT Specific Data)", value=True, help="When on, answers are grounded in ingested SOPs.")

with st.sidebar:
    # Top content
    st.caption("Tip: Refresh the page to re-enter a different email or key.")

    # Bottom content (last child) → pinned by CSS above
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Input ----
user_input = st.chat_input("Ask your port operations question…")

if user_input:
    # 1) Save and render the user's question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Build messages for Groq (optionally with RAG context)
    msgs = []
    if use_rag:
        # Retrieve SOP snippets
        hits = retrieve(user_input)
        if hits:
            ctx_lines = []
            for i, h in enumerate(hits, start=1):
                src = h["meta"].get("source", "unknown")
                ctx_lines.append(f"[CTX {i}] {h['text']}\n(source: {src})")
            context_block = "\n\n".join(ctx_lines)

            # System + user with context appended
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
            msgs += st.session_state.messages[:-1]  # prior history (without the last user msg we already have)
            msgs.append({
                "role": "user",
                "content": f"{user_input}\n\n---\nRETRIEVED SOP CONTEXT:\n{context_block}"
            })
        else:
            # No context found → still set system prompt that asks to be honest about missing info
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
            msgs += st.session_state.messages
    else:
        # No RAG — pure chat
        msgs = st.session_state.messages

    # 3) Call Groq and display assistant answer
    client = OpenAI(
        api_key=st.session_state.api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            answer = resp.choices[0].message.content
            placeholder.markdown(answer)

            # Show simple source list when RAG is on
            if use_rag:
                with st.expander("Sources"):
                    for h in hits or []:
                        st.write("•", h["meta"].get("source", "unknown"))
        except Exception as e:
            placeholder.error(f"Error: {e}")
            answer = ""

    # 4) Save assistant reply
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
