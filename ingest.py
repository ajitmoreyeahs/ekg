"""
ingest.py  –  Build two FAISS vector stores from your Excel + docx files.

  Embedding A: Intent_based_Scenarios_.xlsx + 4 shared files + V3.docx
  Embedding B: Chatbot_Intents.xlsx         + 4 shared files + V3.docx

Usage:
    python ingest.py --data_dir ./data --output_dir ./vectorstores
"""

import argparse
import os
import pandas as pd
from docx import Document
from langchain_core.documents import Document as LCDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ──────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────

def load_excel(path: str, label: str) -> list:
    docs = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet).fillna("")
        for idx, row in df.iterrows():
            row_text = "\n".join(
                f"{col}: {val}" for col, val in row.items() if str(val).strip()
            )
            if not row_text.strip():
                continue
            docs.append(LCDoc(
                page_content=row_text,
                metadata={"source": os.path.basename(path), "sheet": sheet, "row": idx, "label": label},
            ))
    print(f"  {os.path.basename(path)}: {len(docs)} rows loaded")
    return docs


def load_docx(path: str, label: str) -> list:
    doc = Document(path)
    docs = []
    current_heading = "General"
    buffer = []

    def flush(heading, paras):
        text = "\n".join(paras).strip()
        if text:
            docs.append(LCDoc(
                page_content=f"[Section: {heading}]\n{text}",
                metadata={"source": os.path.basename(path), "section": heading, "label": label},
            ))

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            flush(current_heading, buffer)
            current_heading = text
            buffer = []
        else:
            buffer.append(text)

    flush(current_heading, buffer)
    print(f"  {os.path.basename(path)}: {len(docs)} sections loaded")
    return docs


# ──────────────────────────────────────────────────────────────
# BUILD
# ──────────────────────────────────────────────────────────────

def build_vectorstore(docs, store_path, embeddings, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks from {len(docs)} documents")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(store_path)
    print(f"  ✓ Saved → {store_path}\n")


def main(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    SHARED = [
        "ekg-entra-dataload.xlsx",
        "User_Mapping.xlsx",
        "Functional_Role_Matrix.xlsx",
        "Business_Role_Template.xlsx",
    ]

    print("\n📂 Loading shared files …")
    shared_docs = []
    for fname in SHARED:
        shared_docs.extend(load_excel(os.path.join(data_dir, fname), fname))
    shared_docs.extend(load_docx(os.path.join(data_dir, "V3.docx"), "V3.docx"))

    print("\n🤖 Loading embedding model (sentence-transformers/all-MiniLM-L6-v2) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print("\n⬡  Building Embedding A  (Intent_based_Scenarios_) …")
    docs_a = shared_docs + load_excel(
        os.path.join(data_dir, "Intent_based_Scenarios_.xlsx"), "Intent_based_Scenarios_.xlsx"
    )
    build_vectorstore(docs_a, os.path.join(output_dir, "embedding_A"), embeddings)

    print("⬡  Building Embedding B  (Chatbot_Intents) …")
    docs_b = shared_docs + load_excel(
        os.path.join(data_dir, "Chatbot_Intents.xlsx"), "Chatbot_Intents.xlsx"
    )
    build_vectorstore(docs_b, os.path.join(output_dir, "embedding_B"), embeddings)

    print("✅ Both vector stores are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./vectorstores")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)