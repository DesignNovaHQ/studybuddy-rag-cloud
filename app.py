"""
StudyBuddy RAG - CLOUD version (for Streamlit Community Cloud).

Differences from the offline version:
- LLM: Groq API (free tier) instead of a local Ollama model
- Embeddings: fastembed (ONNX, small) instead of sentence-transformers + torch
- Storage: in memory per visitor session (nothing is saved on the server)

Run locally:  streamlit run app.py   (needs GROQ_API_KEY, see README)
"""

import io
import os

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# ----------------------------- CONFIG ---------------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_CHOICES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_PAGES_PER_INDEX = 150  # keeps memory low on free hosting
BATCH = 32

st.set_page_config(page_title="StudyBuddy RAG", page_icon="📚", layout="wide")


# ----------------------------- CORE -----------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        out.extend(list(load_embedder().embed(texts[i : i + BATCH])))
    vecs = np.array(out, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def get_store():
    if "store" not in st.session_state:
        st.session_state.store = {"docs": [], "metas": [], "emb": None}
    return st.session_state.store


def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or st.session_state.get("user_key", "")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("۔", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 30]


def ingest_pdf(uploaded, start_page=1, end_page=None):
    reader = PdfReader(io.BytesIO(uploaded.getvalue()))
    total = len(reader.pages)
    first = max(start_page, 1)
    last = min(end_page or total, total)
    if last - first + 1 > MAX_PAGES_PER_INDEX:
        last = first + MAX_PAGES_PER_INDEX - 1

    docs, metas = [], []
    for p in range(first, last + 1):
        text = (reader.pages[p - 1].extract_text() or "").strip()
        for chunk in chunk_text(text):
            docs.append(chunk)
            metas.append({"source": uploaded.name, "page": p})
    if not docs:
        return 0, last

    store = get_store()
    # replace old chunks of the same file and pages (safe to re-index)
    keep = [
        i
        for i, m in enumerate(store["metas"])
        if not (m["source"] == uploaded.name and first <= m["page"] <= last)
    ]
    store["docs"] = [store["docs"][i] for i in keep]
    store["metas"] = [store["metas"][i] for i in keep]
    if store["emb"] is not None:
        store["emb"] = store["emb"][keep]

    new = embed(docs)
    if store["emb"] is None or len(store["emb"]) == 0:
        store["emb"] = new
    else:
        store["emb"] = np.vstack([store["emb"], new])
    store["docs"] += docs
    store["metas"] += metas
    return len(docs), last


def retrieve(query, k, source=None, p1=None, p2=None):
    store = get_store()
    if store["emb"] is None or not store["docs"]:
        return []
    idx = [
        i
        for i, m in enumerate(store["metas"])
        if (source is None or m["source"] == source)
        and (p1 is None or p1 <= m["page"] <= p2)
    ]
    if not idx:
        return []
    q = embed([query])[0]
    sims = store["emb"][idx] @ q
    order = np.argsort(-sims)[:k]
    return [
        {
            "text": store["docs"][idx[o]],
            "source": store["metas"][idx[o]]["source"],
            "page": store["metas"][idx[o]]["page"],
            "score": round(float(sims[o]), 3),
        }
        for o in order
    ]


def list_sources():
    return sorted({m["source"] for m in get_store()["metas"]})


def page_bounds(source):
    pages = [m["page"] for m in get_store()["metas"] if m["source"] == source]
    return (min(pages), max(pages)) if pages else (1, 1)


def quiz_chunks(source, p1, p2, topic, n):
    if topic:
        return retrieve(topic, n, source, p1, p2)
    store = get_store()
    idx = [
        i
        for i, m in enumerate(store["metas"])
        if m["source"] == source and p1 <= m["page"] <= p2
    ]
    idx.sort(key=lambda i: store["metas"][i]["page"])
    if len(idx) > n:
        step = len(idx) / n
        idx = [idx[int(j * step)] for j in range(n)]
    return [
        {
            "text": store["docs"][i],
            "source": store["metas"][i]["source"],
            "page": store["metas"][i]["page"],
            "score": 1.0,
        }
        for i in idx
    ]


def format_context(hits):
    return "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)


def stream_llm(messages, model, max_tokens=600):
    client = Groq(api_key=get_api_key())
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


SYSTEM_QA = (
    "You are StudyBuddy, a careful study assistant. "
    "Answer ONLY using the provided context. "
    "Reply in the SAME language and script as the user's question "
    "(English, Urdu, or Roman Urdu). "
    "After each fact, cite the source like [file.pdf p.3]. "
    "If the context does not contain the answer, say you could not find it "
    "in the uploaded documents. Never invent information."
)


def quiz_prompt(n):
    return (
        "You are an exam question writer. Using ONLY the provided context, write "
        f"exactly {n} multiple-choice questions, numbered 1 to {n}, each with "
        "options A, B, C, D and only one correct answer. Make wrong options "
        "plausible. Cover different parts of the context, not just one idea. "
        "Do not write questions about page numbers or the book itself. "
        "After all questions, add an 'Answer Key' section listing the correct "
        "letter and a one-line explanation, citing the source like [file.pdf p.3]. "
        "Do not use outside knowledge."
    )


def show_llm_error(e):
    st.error(
        f"LLM error: {e}\n\nCheck that GROQ_API_KEY is set correctly. "
        "If it says rate limit, wait a minute or switch to llama-3.1-8b-instant."
    )


# ----------------------------- UI -------------------------------------
st.title("📚 StudyBuddy RAG")
st.caption("Upload your own PDF • English / Urdu / Roman Urdu • Answers with page citations")

with st.sidebar:
    st.header("⚙️ Settings")
    if not get_api_key():
        st.session_state.user_key = st.text_input(
            "Groq API key (free at console.groq.com)", type="password"
        )
    llm_model = st.selectbox("LLM", LLM_CHOICES)
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 3)
    threshold = st.slider(
        "Min. relevance (below this = 'not found')", 0.0, 0.8, 0.25, 0.05
    )
    mode = st.radio("Mode", ["💬 Ask questions", "📝 Generate quiz"])

    st.header("📄 Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    c1, c2 = st.columns(2)
    start_pg = c1.number_input("From page", min_value=1, value=1, step=1)
    end_pg = c2.number_input("To page (0 = up to 150 pages)", min_value=0, value=0, step=1)
    st.caption(f"Up to {MAX_PAGES_PER_INDEX} pages per indexing run. Index one chapter at a time.")
    if st.button("Index uploaded PDFs", use_container_width=True) and files:
        with st.spinner("Indexing..."):
            for f in files:
                n, last = ingest_pdf(f, int(start_pg), int(end_pg) or None)
                if n:
                    st.success(f"{f.name}: {n} chunks stored (up to page {last})")
                else:
                    st.info(f"{f.name}: no text found in those pages")

    st.write(f"Chunks indexed: **{len(get_store()['docs'])}**")
    if st.button("🗑️ Clear all", use_container_width=True):
        st.session_state.store = {"docs": [], "metas": [], "emb": None}
        st.session_state.messages = []
        st.session_state.pop("quiz_text", None)
        st.rerun()
    st.caption(
        "Privacy: matching text snippets are sent to the Groq API to write answers. "
        "Do not upload confidential documents."
    )

if not get_api_key():
    st.info("Add your Groq API key in the sidebar to start (it is free).")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------- QUIZ MODE ------------------------------
if mode.startswith("📝"):
    st.subheader("📝 Quiz generator")
    sources = list_sources()
    if not sources:
        st.info("Index a PDF first (sidebar).")
        st.stop()

    src = st.selectbox("Book / file", sources)
    lo, hi = page_bounds(src)
    st.caption(f"Indexed pages for this file: {lo} to {hi}")
    c1, c2, c3 = st.columns(3)
    p1 = c1.number_input("From page", min_value=lo, max_value=hi, value=lo, step=1)
    p2 = c2.number_input("To page", min_value=lo, max_value=hi, value=hi, step=1)
    n_q = c3.slider("Questions", 5, 15, 10)
    topic = st.text_input("Optional: focus on a topic (e.g. 'deadlock', 'paging')")

    generated = False
    if st.button("⚡ Generate quiz", type="primary"):
        hits = quiz_chunks(src, int(p1), int(p2), topic.strip(), min(max(8, n_q), 12))
        if not hits:
            st.warning("No content found in that page range. Index those pages first.")
        else:
            msgs = [
                {"role": "system", "content": quiz_prompt(n_q)},
                {
                    "role": "user",
                    "content": f"Context:\n{format_context(hits)}\n\nWrite {n_q} questions.",
                },
            ]
            try:
                st.session_state.quiz_text = st.write_stream(
                    stream_llm(msgs, llm_model, max_tokens=300 + 120 * n_q)
                )
                generated = True
            except Exception as e:
                show_llm_error(e)

    if st.session_state.get("quiz_text"):
        if not generated:
            st.markdown(st.session_state.quiz_text)
        st.download_button(
            "⬇️ Download quiz (.txt)", st.session_state.quiz_text, file_name="quiz.txt"
        )
    st.stop()

# ----------------------------- CHAT MODE ------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📎 Sources"):
                for s in m["sources"]:
                    st.markdown(
                        f"**{s['source']} - page {s['page']}** "
                        f"(relevance {s['score']})\n\n> {s['text'][:350]}..."
                    )

question = st.chat_input("Ask about your documents (English / اردو / Roman Urdu)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = [h for h in retrieve(question, top_k) if h["score"] >= threshold]

        if not hits:
            answer = (
                "I could not find this in your uploaded documents. "
                "Index the right pages first, rephrase, or lower the relevance slider."
            )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-5:-1]
            ]
            msgs = (
                [{"role": "system", "content": SYSTEM_QA}]
                + history
                + [
                    {
                        "role": "user",
                        "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}",
                    }
                ]
            )
            try:
                answer = st.write_stream(stream_llm(msgs, llm_model))
            except Exception as e:
                answer = f"LLM error: {e}"
                show_llm_error(e)

            with st.expander("📎 Sources"):
                for s in hits:
                    st.markdown(
                        f"**{s['source']} - page {s['page']}** "
                        f"(relevance {s['score']})\n\n> {s['text'][:350]}..."
                    )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits}
            )
