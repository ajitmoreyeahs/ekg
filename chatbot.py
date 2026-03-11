"""
chatbot.py  –  Interactive CLI chatbot powered by the LangGraph RAG pipeline.

Usage:
    python chatbot.py
    python chatbot.py --vs_dir ./vectorstores --model gpt-4o-mini
"""

import argparse
import sys
from graph import build_graph
from dotenv import load_dotenv
load_dotenv()
BANNER = """
╔══════════════════════════════════════════════════════════╗
║   EKG Identity & Access Management Chatbot (POC)        ║
║   Dual-Embedding RAG   |   Powered by LangGraph         ║
╚══════════════════════════════════════════════════════════╝

Type your question and press Enter.
Commands:  /exit  → quit  |  /clear → new session  |  /debug → show sources
"""

EXAMPLE_QUESTIONS = [
    "Does user jdevlin have an SoD violation?",
    "Which users are both PO Creator and PO Releaser?",
    "What roles are assigned to users in the Finance department?",
    "What are the SoD conflict scenarios defined in the system?",
    "Who is the manager of ftrombley?",
    "List all users in the UK with an L4 level.",
    "What is the intent for a user who just joined the procurement team?",
]


def run_chatbot(vs_dir: str, model: str, show_sources: bool = False):
    print(BANNER)
    print("📌 Example questions:")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()

    print("⏳ Loading vector stores and models …")
    try:
        app = build_graph(vs_dir=vs_dir, model_name=model)
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        sys.exit(1)

    print("✅ Ready!\n")

    debug_mode = show_sources


    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Batch mode: process all example questions
        if user_input.lower() == "/batch":
            for q in EXAMPLE_QUESTIONS:
                print(f"\nYou: {q}\n🤔 Thinking …\n")
                try:
                    result = app.invoke({
                        "question": q,
                        "intent": "",
                        "docs_a": [],
                        "docs_b": [],
                        "answer_a": "",
                        "answer_b": "",
                        "answer": "",
                        "embedding_chosen": "",
                    })
                    print("💬 Answer:\n")
                    print(result.get("answer", "No answer generated."))
                    print("\n" + "═" * 60 + "\n")
                except Exception as e:
                    print(f"❌ Error: {e}\n")
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            print("\n" + "─" * 60 + "\n")
            continue

        if user_input.lower() == "/debug":
            debug_mode = not debug_mode
            print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
            continue

        print("\n🤔 Thinking …\n")

        try:
            result = app.invoke({
                "question": user_input,
                "intent": "",
                "docs_a": [],
                "docs_b": [],
                "answer_a": "",
                "answer_b": "",
                "answer": "",
                "embedding_chosen": "",
            })
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue

        # ── Print intent ──
        intent_labels = {
            "sod_check": "🔴 SoD Check",
            "role_query": "🟡 Role Query",
            "user_query": "🟢 User Query",
            "general": "🔵 General",
        }
        intent_display = intent_labels.get(result.get("intent", ""), "❓ Unknown")
        print(f"Intent detected: {intent_display}")
        print("─" * 60)

        # ── Debug: show retrieved sources ──
        if debug_mode:
            print("\n📄 Sources from Embedding A:")
            for d in result.get("docs_a", []):
                meta = d.metadata
                print(f"  [{meta.get('source')} / {meta.get('sheet', meta.get('section',''))}]")
                print(f"  {d.page_content[:200]}\n")

            print("\n📄 Sources from Embedding B:")
            for d in result.get("docs_b", []):
                meta = d.metadata
                print(f"  [{meta.get('source')} / {meta.get('sheet', meta.get('section',''))}]")
                print(f"  {d.page_content[:200]}\n")

            print("\n🅰️ Answer from Embedding A:")
            print(result.get("answer_a", "—"))
            print("\n🅱️ Answer from Embedding B:")
            print(result.get("answer_b", "—"))
            print("\n" + "─" * 60)

        # ── Final answer ──
        print("\n💬 Answer:\n")
        print(result.get("answer", "No answer generated."))
        print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKG IAM Chatbot")
    parser.add_argument("--vs_dir", default="./vectorstores", help="Path to FAISS vector stores")
    parser.add_argument("--model", default="gemini-2.5-flash", help="OpenAI model name")
    parser.add_argument("--debug", action="store_true", help="Show retrieved sources by default")
    args = parser.parse_args()

    run_chatbot(vs_dir=args.vs_dir, model=args.model, show_sources=args.debug)
