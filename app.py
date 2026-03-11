"""
app.py  –  Streamlit chatbot UI for the dual-embedding RAG framework
Run:  streamlit run app.py
"""

import os
import time
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="EKG IAM Assistant",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* ── main header ── */
.main-header {
    background: linear-gradient(135deg, #0f1923 0%, #0d1f35 50%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #3b82f6);
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}
.badge {
    display: inline-block;
    background: #0f2744;
    border: 1px solid #1d4ed8;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 10px;
}

/* ── chat messages ── */
.user-bubble {
    background: #1a2744;
    border: 1px solid #1e3a6e;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 16px 0 8px auto;
    max-width: 75%;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
}
.user-label {
    text-align: right;
    color: #475569;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
}

/* ── intent badge ── */
.intent-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 18px 0 14px 0;
}
.intent-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 12px;
    border-radius: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.intent-sod   { background:#2d1515; color:#f87171; border:1px solid #7f1d1d; }
.intent-role  { background:#1c2a14; color:#86efac; border:1px solid #14532d; }
.intent-user  { background:#1a2744; color:#93c5fd; border:1px solid #1e3a6e; }
.intent-general { background:#1e1a2e; color:#c4b5fd; border:1px solid #4c1d95; }

/* ── embedding answer cards ── */
.emb-card {
    background: #111318;
    border-radius: 10px;
    padding: 20px 22px;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.emb-card-a { border: 1px solid #1e3a6e; }
.emb-card-b { border: 1px solid #1a3028; }
.emb-card-a::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.emb-card-b::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #34d399);
}
.emb-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.emb-title-a { color: #60a5fa; }
.emb-title-b { color: #34d399; }
.emb-subtitle {
    font-size: 0.72rem;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2330;
}
.emb-body {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* ── source pills ── */
.source-pill {
    display: inline-block;
    background: #0d1520;
    border: 1px solid #1e2d45;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 9px;
    border-radius: 4px;
    margin: 3px 3px 3px 0;
}

/* ── winner card ── */
.winner-card {
    background: #0f1a0f;
    border: 1px solid #14532d;
    border-radius: 10px;
    padding: 20px 22px;
    margin-top: 20px;
    position: relative;
    overflow: hidden;
}
.winner-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b);
}
.winner-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #f59e0b;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.winner-body {
    color: #d1fae5;
    font-size: 0.92rem;
    line-height: 1.75;
    white-space: pre-wrap;
}
.winner-verdict {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #1a3a28;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6ee7b7;
}

/* ── divider ── */
.chat-divider {
    border: none;
    border-top: 1px solid #1e2330;
    margin: 24px 0;
}

/* ── input area ── */
.stTextInput > div > div > input {
    background: #111318 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
}

/* ── expander ── */
details { background: #0d0f14 !important; }
summary { color: #475569 !important; font-size: 0.8rem !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }

/* ── hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 EKG IAM Assistant")
    st.markdown("---")

    st.markdown("### Configuration")
    google_api_key = 'AIzaSyB2x0bRfm9f26Az9dPJZSwOPGLrOBfSq4w'
    gemini_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
        help="gemini-1.5-flash recommended for free tier (1500 req/day)"
    )
    st.caption("Free tier tip: use gemini-1.5-flash for best quota")

    vs_dir = st.text_input(
        "Vector Store Path",
        value="./vectorstores",
        help="Path to the folder containing embedding_A and embedding_B",
    )

    top_k = st.slider("Retrieved chunks (top-k)", min_value=3, max_value=12, value=6)

    st.markdown("---")
    st.markdown("### Embedding Setup")
    st.markdown("""
<div style='font-size:0.78rem; color:#64748b; line-height:1.8;'>
<span style='color:#60a5fa;'>● Embedding A</span><br>
&nbsp;&nbsp;Intent_based_Scenarios.xlsx<br>
&nbsp;&nbsp;+ 4 shared files + V3.docx<br><br>
<span style='color:#34d399;'>● Embedding B</span><br>
&nbsp;&nbsp;Chatbot_Intents.xlsx<br>
&nbsp;&nbsp;+ 4 shared files + V3.docx
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Intent Types")
    st.markdown("""
<div style='font-size:0.78rem; line-height:2;'>
<span style='color:#f87171;'>🔴 sod_check</span> — SoD violations<br>
<span style='color:#86efac;'>🟢 role_query</span> — Roles & permissions<br>
<span style='color:#93c5fd;'>🔵 user_query</span> — User attributes<br>
<span style='color:#c4b5fd;'>🟣 general</span> — Org model & policies
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.graph = None
        st.rerun()


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔐 Enterprise Knowledge Graph · IAM Chatbot</h1>
  <p>Dual-embedding RAG  ·  LangGraph Agentic Pipeline  ·  Gemini LLM</p>
  <div>
    <span class="badge">SAP S/4HANA</span>
    <span class="badge">Entra ID</span>
    <span class="badge">Neo4j EKG</span>
    <span class="badge">SoD Rules</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None


# ── Load graph ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_graph(vs_dir, gemini_model, google_api_key, top_k):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    print(f"[app] Loading graph: model={gemini_model}, vs_dir={vs_dir}, top_k={top_k}")
    from graph import build_graph
    g = build_graph(vs_dir=vs_dir, gemini_model=gemini_model, top_k=top_k)
    print("[app] Graph loaded successfully")
    return g


# ── Render past messages ──────────────────────────────────────
INTENT_CHIP = {
    "sod_check":  ("🔴 SoD Check",    "intent-sod"),
    "role_query": ("🟢 Role Query",   "intent-role"),
    "user_query": ("🔵 User Query",   "intent-user"),
    "general":    ("🟣 General",      "intent-general"),
}

def render_message(msg):
    if msg["role"] == "user":
        st.markdown(f"""
<div class="user-label">YOU</div>
<div class="user-bubble">{msg["content"]}</div>
""", unsafe_allow_html=True)
    else:
        r = msg["result"]
        intent = r.get("intent", "general")
        chip_label, chip_class = INTENT_CHIP.get(intent, ("🟣 General", "intent-general"))

        st.markdown(f"""
<div class="intent-row">
  <span class="intent-chip {chip_class}">{chip_label}</span>
  <span style="color:#334155; font-size:0.72rem; font-family:'IBM Plex Mono',monospace;">
    intent detected
  </span>
</div>
""", unsafe_allow_html=True)

        # Side-by-side embedding answers
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            sources_html = "".join(
                f'<span class="source-pill">📄 {s["file"]} › {s["sheet"]}</span>'
                for s in r.get("sources_a", [])
            )
            st.markdown(f"""
<div class="emb-card emb-card-a">
  <div class="emb-title emb-title-a">⬡ Embedding A</div>
  <div class="emb-subtitle">Intent_based_Scenarios.xlsx + shared files</div>
  <div class="emb-body">{r.get("answer_a", "—")}</div>
  <div style="margin-top:14px;">{sources_html}</div>
</div>
""", unsafe_allow_html=True)

        with col_b:
            sources_html = "".join(
                f'<span class="source-pill">📄 {s["file"]} › {s["sheet"]}</span>'
                for s in r.get("sources_b", [])
            )
            st.markdown(f"""
<div class="emb-card emb-card-b">
  <div class="emb-title emb-title-b">⬡ Embedding B</div>
  <div class="emb-subtitle">Chatbot_Intents.xlsx + shared files</div>
  <div class="emb-body">{r.get("answer_b", "—")}</div>
  <div style="margin-top:14px;">{sources_html}</div>
</div>
""", unsafe_allow_html=True)

        # Winner / reconciled answer
        winner = r.get("winner", "Both")
        reason = r.get("winner_reason", "")
        winner_color = {"A": "#60a5fa", "B": "#34d399", "Both": "#f59e0b"}.get(winner, "#f59e0b")

        st.markdown(f"""
<div class="winner-card">
  <div class="winner-label">⚡ Reconciled Best Answer</div>
  <div class="winner-body">{r.get("answer", "—")}</div>
  <div class="winner-verdict">
    🏆 Better embedding: <span style="color:{winner_color}; font-weight:600;">Embedding {winner}</span>
    &nbsp;·&nbsp; {reason}
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)


for msg in st.session_state.messages:
    render_message(msg)


# ── Suggested questions ───────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style="color:#334155; font-size:0.78rem; font-family:'IBM Plex Mono',monospace;
            margin-bottom:12px; letter-spacing:0.5px;">
  SUGGESTED QUERIES
</div>
""", unsafe_allow_html=True)

    suggestions = [
        "Does user jdevlin have an SoD violation?",
        "Which users are both PO Creator and PO Releaser?",
        "What roles are assigned to Finance department users?",
        "List all L4 users in the United States",
        "What are the SoD conflict scenarios defined?",
        "What happens when a user joins the procurement team in UK?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(q, key=f"sug_{i}"):
                st.session_state.pending_question = q
                st.rerun()


# ── Input area ────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Ask a question",
        label_visibility="collapsed",
        placeholder="e.g. Does user jdevlin have an SoD conflict?",
        key="user_input",
    )

with col_btn:
    send_clicked = st.button("Send →", use_container_width=True)


# ── Handle pending suggestion click ──────────────────────────
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
    send_clicked = True


# ── Process query ─────────────────────────────────────────────
if send_clicked and user_input.strip():
    if not google_api_key:
        st.error("⚠️ Please enter your Google API Key in the sidebar.")
        st.stop()

    if not os.path.isdir(vs_dir) or (
        not os.path.isdir(os.path.join(vs_dir, "embedding_A")) or
        not os.path.isdir(os.path.join(vs_dir, "embedding_B"))
    ):
        st.error(f"⚠️ Vector stores not found at `{vs_dir}`. Run `python ingest.py` first.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    status_box = st.empty()
    progress_bar = st.empty()

    try:
        rag_app = get_graph(vs_dir, gemini_model, google_api_key, top_k)
        print(f"[app] Invoking graph for: {user_input}")

        steps = [
            (0.15, "Step 1/4 — Classifying intent..."),
            (0.40, "Step 2/4 — Retrieving documents from both embeddings..."),
            (0.65, "Step 3/4 — Generating answers (A & B) with Gemini..."),
            (0.90, "Step 4/4 — Comparing results & selecting best answer..."),
        ]

        import threading
        result_holder = {}

        def run_graph():
            try:
                result_holder["result"] = rag_app.invoke({
                    "question": user_input,
                    "intent": "",
                    "docs_a": [],
                    "docs_b": [],
                    "sources_a": [],
                    "sources_b": [],
                    "answer_a": "",
                    "answer_b": "",
                    "answer": "",
                    "winner": "",
                    "winner_reason": "",
                })
            except Exception as ex:
                result_holder["error"] = ex

        t = threading.Thread(target=run_graph)
        t.start()

        step_idx = 0
        while t.is_alive():
            if step_idx < len(steps):
                pct, label = steps[step_idx]
                status_box.markdown(f"""
<div style="background:#111318;border:1px solid #1e2d45;border-radius:8px;
            padding:14px 18px;font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#60a5fa;">
  ⏳ {label}
  <span style="color:#334155;font-size:0.72rem;float:right;">
    Free-tier: auto-retrying on quota limits
  </span>
</div>""", unsafe_allow_html=True)
                progress_bar.progress(pct)
                step_idx += 1
            time.sleep(3)

        t.join()
        status_box.empty()
        progress_bar.empty()

        if "error" in result_holder:
            raise result_holder["error"]

        result = result_holder["result"]
        print(f"[app] Graph complete. Answer length: {len(result.get('answer',''))}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "result": result,
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[app] ERROR:\n{tb}")
        status_box.empty()
        progress_bar.empty()
        err_str = str(e)
        if "QUOTA_EXHAUSTED" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            st.warning(
                "**Gemini Free-Tier Quota Exceeded**\n\n"
                "The app already auto-retried with the delay Gemini requested.\n\n"
                "**To fix:**\n"
                "- Switch model to `gemini-1.5-flash` in the sidebar (1,500 req/day free)\n"
                "- Or wait 1-2 minutes for the per-minute quota to reset\n"
                "- Or add billing at https://aistudio.google.com"
            )
        else:
            st.error(f"Error: {e}")
            st.code(tb, language="text")

    st.rerun()