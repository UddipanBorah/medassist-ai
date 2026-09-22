#!/usr/bin/env python3
"""
MedAssist AI - CLI entry point.

Usage:
    python app.py                # interactive chat loop
    python app.py --demo         # run a scripted set of demo queries and exit

By default the agent auto-selects an LLM backend:
    - OPENAI_API_KEY set     -> uses OpenAI (gpt-4o-mini)
    - ANTHROPIC_API_KEY set  -> uses Anthropic (claude-3-5-haiku)
    - neither set            -> uses the offline MockBackend (no API needed)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import MedAssistAgent  # noqa: E402

KB_PATH = Path(__file__).parent / "data" / "medical_kb.json"

DEMO_QUERIES = [
    "I have a runny nose and sore throat for two days",
    "I've been feeling really thirsty and tired lately and losing weight",
    "sick",
    "I have sudden chest pain and pain going down my left arm",
    "My child has hives all over and their lips are swelling after eating peanuts",
    "I've had trouble sleeping and I'm anxious about work all the time",
]


def print_banner(agent: MedAssistAgent) -> None:
    print("=" * 70)
    print(" MedAssist AI - Agentic RAG Healthcare Q&A Assistant (DEMO PROJECT)")
    print(f" LLM backend in use: {agent.backend.name}")
    print(" Type 'quit' to exit.")
    print(" NOTE: For educational/demo purposes only. Not a substitute for")
    print("       professional medical advice. In a real emergency, call your")
    print("       local emergency number immediately.")
    print("=" * 70)


def run_interactive(agent: MedAssistAgent) -> None:
    print_banner(agent)
    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_message:
            continue
        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye! Stay healthy.")
            break

        turn = agent.handle(user_message)
        print(f"\nMedAssist [{turn.turn_type}]: {turn.response}")
        if turn.retrieved_conditions:
            print(f"  (retrieved: {', '.join(turn.retrieved_conditions)})")


def run_demo(agent: MedAssistAgent) -> None:
    print_banner(agent)
    for query in DEMO_QUERIES:
        turn = agent.handle(query)
        print("\n" + "-" * 70)
        print(f"You: {query}")
        print(f"MedAssist [{turn.turn_type}]: {turn.response}")
        if turn.retrieved_conditions:
            print(f"  (retrieved: {', '.join(turn.retrieved_conditions)})")
        print(f"  (tools used: {', '.join(turn.tools_used)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedAssist AI CLI")
    parser.add_argument("--demo", action="store_true", help="run scripted demo queries")
    args = parser.parse_args()

    agent = MedAssistAgent(kb_path=str(KB_PATH))

    if args.demo:
        run_demo(agent)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
