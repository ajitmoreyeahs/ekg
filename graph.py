"""
graph.py - LangGraph RAG pipeline with free-tier Gemini quota management.

Key free-tier optimizations:
  - Parses retryDelay from Gemini 429 error and waits exactly that long
  - Trims context to ~1500 chars per doc to minimize token usage
  - Combines classify + RAG into fewer LLM calls (3 total instead of 4)
  - Exponential backoff with jitter on retries
"""

import os
import re
import time
import random
import json
import boto3
from typing import TypedDict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, END


# =============================================================
# CUSTOM BEDROCK LLM USING MESSAGES API (CONVERSE)
# =============================================================

class BedrockMessagesLLM(LLM):
    """Custom LLM wrapper for Bedrock using the Messages API (converse)."""
    
    model_id: str
    region: str = "eu-west-2"
    max_tokens: int = 512
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "bedrock-messages"
    
    def _call(
        self,
        prompt: str,
        stop=None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> str:
        """Call the Bedrock Messages API (converse)."""
        client = boto3.client(
            'bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        response = client.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        
        return response['output']['message']['content'][0]['text']


# =============================================================
# FREE-TIER QUOTA MANAGER
# =============================================================

def parse_retry_delay(err_str: str) -> float:
    """Extract the retryDelay seconds from Gemini 429 error message."""
    # Matches patterns like: 'retryDelay': '13s'  or  Please retry in 51.8s
    match = re.search(r"retry(?:Delay|ing)[\s\S]{0,20}?(\d+(?:\.\d+)?)\s*s", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # add 2s buffer
    return 20.0  # safe default


def llm_invoke_with_retry(chain, inputs: dict, max_retries: int = 5, step_label: str = ""):
    """
    Invoke chain with smart retry on Gemini 429 errors.
    - Reads the actual retryDelay from the error response
    - Adds small jitter to avoid thundering herd
    - Raises a clear RuntimeError after max_retries
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if not is_quota:
                raise  # Not a quota error, re-raise immediately

            if attempt >= max_retries - 1:
                raise RuntimeError(
                    "QUOTA_EXHAUSTED: Gemini free-tier quota still exceeded after "
                    f"{max_retries} retries.\n\n"
                    "Options:\n"
                    "  1. Wait 1-2 minutes and try again\n"
                    "  2. Switch to 'gemini-1.5-flash' in the sidebar (most generous free tier)\n"
                    "  3. Add billing at https://aistudio.google.com to remove limits"
                ) from e

            wait = parse_retry_delay(err_str) + random.uniform(0.5, 3.0)
            label = f" [{step_label}]" if step_label else ""
            print(f"[graph]{label} Quota hit (attempt {attempt+1}/{max_retries}). "
                  f"Waiting {wait:.1f}s as requested by Gemini...")
            time.sleep(wait)

    raise RuntimeError("Unreachable")


# =============================================================
# TOKEN SAVER - trim docs to reduce input token usage
# =============================================================

def format_docs_trimmed(docs: list, max_chars_per_doc: int = 800) -> str:
    """
    Format retrieved docs with character limit per doc.
    Keeps token usage low for free-tier accounts.
    Increase max_chars_per_doc if you have billing enabled.
    """
    parts = []
    for d in docs:
        source = d.metadata.get("source", "?")
        sheet  = d.metadata.get("sheet", d.metadata.get("section", "?"))
        body   = d.page_content[:max_chars_per_doc]
        if len(d.page_content) > max_chars_per_doc:
            body += "... [trimmed]"
        parts.append(f"[{source} / {sheet}]\n{body}")
    return "\n\n---\n\n".join(parts)


def extract_source_meta(docs: list) -> list:
    seen = set()
    sources = []
    for d in docs:
        key = (
            d.metadata.get("source", "?"),
            d.metadata.get("sheet", d.metadata.get("section", "?")),
        )
        if key not in seen:
            seen.add(key)
            sources.append({"file": key[0], "sheet": key[1]})
    return sources


# =============================================================
# AGENT STATE
# =============================================================

class AgentState(TypedDict):
    question: str
    intent: str
    docs_a: list
    docs_b: list
    sources_a: list
    sources_b: list
    answer_a: str
    answer_b: str
    answer: str
    winner: str
    winner_reason: str


# =============================================================
# GRAPH BUILDER
# =============================================================

def build_graph(
    vs_dir: str = "./vectorstores",
    bedrock_model_id: str = "eu.anthropic.claude-opus-4-5-20251101-v1:0s",
    top_k: int = 4,   # reduced from 6 -> saves tokens on free tier
):
    print(f"[graph] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"[graph] Loading vector stores from: {vs_dir}")
    vs_a = FAISS.load_local(
        os.path.join(vs_dir, "embedding_A"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    vs_b = FAISS.load_local(
        os.path.join(vs_dir, "embedding_B"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"[graph] Initializing Bedrock Claude Opus: {bedrock_model_id}")
    llm = BedrockMessagesLLM(
        model_id=bedrock_model_id,
        region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")),
        max_tokens=512,
        temperature=0,
    )

    # =========================================================
    # PROMPTS  (kept short to save input tokens)
    # =========================================================

    # LLM call 1: classify intent
    CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Classify this IAM question into ONE label only:\n"
         "sod_check | role_query | user_query | general\n"
         "Reply with the label only, no other text."),
        ("human", "{question}"),
    ])

    # LLM call 2 & 3: answer from each embedding (same prompt, called twice)
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are an IAM assistant for SAP/Entra ID.\n"
         "Answer using ONLY the context below. Be concise and precise.\n"
         "Include user names, role IDs, or SoD rule IDs when available.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # LLM call 4: compare and pick winner
    COMPARE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You have two answers from different IAM knowledge bases.\n"
         "Embedding A used Intent_based_Scenarios data.\n"
         "Embedding B used Chatbot_Intents data.\n\n"
         "Answer A: {answer_a}\n\n"
         "Answer B: {answer_b}\n\n"
         "Write the single best answer combining A and B.\n"
         "Then on a new line: WINNER: A  or  WINNER: B  or  WINNER: Both\n"
         "Then on a new line: REASON: one sentence."),
        ("human", "Question: {question}"),
    ])

    # =========================================================
    # NODES
    # =========================================================

    def route_query(state: AgentState) -> AgentState:
        print(f"[graph] Step 1/4 - Classifying intent...")
        chain = CLASSIFY_PROMPT | llm
        result = llm_invoke_with_retry(chain, {"question": state["question"]}, step_label="classify")
        intent = result.strip().lower().split()[0]  # take first word only
        if intent not in ("sod_check", "role_query", "user_query", "general"):
            intent = "general"
        print(f"[graph] Intent: {intent}")
        return {**state, "intent": intent}

    def retrieve_both(state: AgentState) -> AgentState:
        print(f"[graph] Step 2/4 - Retrieving docs (top-{top_k} per store, no LLM call)...")
        q = state["question"]
        docs_a = vs_a.similarity_search(q, k=top_k)
        docs_b = vs_b.similarity_search(q, k=top_k)
        print(f"[graph] Got {len(docs_a)} docs from A, {len(docs_b)} from B")
        return {
            **state,
            "docs_a": docs_a,
            "docs_b": docs_b,
            "sources_a": extract_source_meta(docs_a),
            "sources_b": extract_source_meta(docs_b),
        }

    def generate_both_answers(state: AgentState) -> AgentState:
        chain = RAG_PROMPT | llm

        print("[graph] Step 3/4 - Generating answer from Embedding A...")
        res_a = llm_invoke_with_retry(chain, {
            "context": format_docs_trimmed(state["docs_a"]),
            "question": state["question"],
        }, step_label="answer_A")
        answer_a = res_a
        print(f"[graph] Answer A ready ({len(answer_a)} chars)")

        # Small pause between calls to respect per-minute rate limits
        time.sleep(2)

        print("[graph] Step 3b/4 - Generating answer from Embedding B...")
        res_b = llm_invoke_with_retry(chain, {
            "context": format_docs_trimmed(state["docs_b"]),
            "question": state["question"],
        }, step_label="answer_B")
        answer_b = res_b
        print(f"[graph] Answer B ready ({len(answer_b)} chars)")

        return {**state, "answer_a": answer_a, "answer_b": answer_b}

    def compare_and_finalize(state: AgentState) -> AgentState:
        print("[graph] Step 4/4 - Comparing and finalizing...")
        time.sleep(2)  # brief pause before 4th call
        chain = COMPARE_PROMPT | llm
        result = llm_invoke_with_retry(chain, {
            "answer_a": state["answer_a"],
            "answer_b": state["answer_b"],
            "question": state["question"],
        }, step_label="compare")
        raw = result

        winner = "Both"
        reason = ""
        final_lines = []
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("WINNER:"):
                w = s.replace("WINNER:", "").strip()
                winner = "A" if w == "A" else ("B" if w == "B" else "Both")
            elif s.startswith("REASON:"):
                reason = s.replace("REASON:", "").strip()
            else:
                final_lines.append(line)

        final_answer = "\n".join(final_lines).strip()
        print(f"[graph] Done. Winner: {winner}")
        return {**state, "answer": final_answer, "winner": winner, "winner_reason": reason}

    # =========================================================
    # WIRE GRAPH
    # =========================================================

    graph = StateGraph(AgentState)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_both", retrieve_both)
    graph.add_node("generate_both_answers", generate_both_answers)
    graph.add_node("compare_and_finalize", compare_and_finalize)

    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "retrieve_both")
    graph.add_edge("retrieve_both", "generate_both_answers")
    graph.add_edge("generate_both_answers", "compare_and_finalize")
    graph.add_edge("compare_and_finalize", END)

    return graph.compile()