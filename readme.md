# EKG IAM Chatbot v2 — Gemini + Streamlit

Dual-embedding RAG agentic chatbot with **Google Gemini** LLM and a polished **Streamlit** UI.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Place your data files in `./data/`

```
data/
├── Intent_based_Scenarios_.xlsx   ← Embedding A (unique)
├── Chatbot_Intents.xlsx           ← Embedding B (unique)
├── ekg-entra-dataload.xlsx        ← shared
├── User_Mapping.xlsx              ← shared
├── Functional_Role_Matrix.xlsx    ← shared
├── Business_Role_Template.xlsx    ← shared
└── V3.docx                        ← shared
```

### 3. Build vector stores (run once)

```bash
python ingest.py --data_dir ./data --output_dir ./vectorstores
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 and enter your **Google API Key** in the sidebar.
Get a free key at https://aistudio.google.com/

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐   Gemini classifies intent:
│ route_query │   sod_check | role_query | user_query | general
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ retrieve_both│   Queries BOTH FAISS stores simultaneously
└──┬───────┬───┘
   │       │
   ▼       ▼
┌──────┐ ┌──────┐
│ Gen A│ │ Gen B│   Gemini generates independent answers
└──┬───┘ └───┬──┘
   └────┬────┘
        ▼
┌──────────────────┐
│compare_and_final │   Gemini picks winner + writes best answer
└────────┬─────────┘
         ▼
   Streamlit UI
   ┌─────────┬─────────┐
   │Embed A  │ Embed B │  ← side-by-side answers
   └─────────┴─────────┘
   ┌─────────────────────┐
   │ ⚡ Best Answer       │  ← reconciled + winner badge
   └─────────────────────┘
```

---

## UI Features

* **Side-by-side panels** showing Embedding A vs B answers independently
* **Source pills** showing which files were retrieved per embedding
* **Intent badge** (SoD / Role / User / General) shown per query
* **Winner card** with Gemini's reconciled best answer + which embedding won
* **Suggested query buttons** on first load
* **Dark terminal aesthetic** — IBM Plex Mono + clean card layout

---

## Gemini Models

| Model                | Speed        | Quality         |
| -------------------- | ------------ | --------------- |
| `gemini-1.5-flash` | ⚡ Fast      | Good — default |
| `gemini-2.0-flash` | ⚡⚡ Fastest | Great           |
| `gemini-1.5-pro`   | 🐢 Slower    | Best quality    |
# ekg
