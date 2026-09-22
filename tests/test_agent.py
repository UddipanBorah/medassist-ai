"""
Basic pipeline tests for the MedAssist agent.

Run with:  python -m pytest tests/ -v
(or simply: python tests/test_agent.py)

These use the offline MockBackend so they run without any API key or
network access, which keeps the project easy to grade/CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import MedAssistAgent
from src.llm_backend import MockBackend

KB_PATH = Path(__file__).parent.parent / "data" / "medical_kb.json"


def make_agent() -> MedAssistAgent:
    return MedAssistAgent(kb_path=str(KB_PATH), backend=MockBackend())


def test_red_flag_emergency_detected():
    agent = make_agent()
    turn = agent.handle("I have sudden chest pain and pain radiating to my arm")
    assert turn.turn_type == "emergency"
    assert "EMERGENCY" in turn.response


def test_vague_message_triggers_clarification():
    agent = make_agent()
    turn = agent.handle("sick")
    assert turn.turn_type == "clarification"


def test_normal_query_returns_rag_answer_with_disclaimer():
    agent = make_agent()
    turn = agent.handle("I have a runny nose and sore throat for two days")
    assert turn.turn_type == "answer"
    assert "Common Cold" in turn.retrieved_conditions
    assert "not a medical diagnosis" in turn.response.lower() or "consult a licensed" in turn.response.lower()


def test_memory_accumulates_turns():
    agent = make_agent()
    agent.handle("I have a headache")
    agent.handle("I also feel nauseous and light bothers my eyes")
    assert len(agent.memory.turns) == 2


def test_retriever_returns_relevant_condition():
    agent = make_agent()
    results = agent.retriever.search("frequent urination and excessive thirst", top_k=3)
    conditions = [r.condition for r in results]
    assert any("Diabetes" in c for c in conditions)


if __name__ == "__main__":
    # allow running as a plain script too (no pytest dependency required)
    tests = [
        test_red_flag_emergency_detected,
        test_vague_message_triggers_clarification,
        test_normal_query_returns_rag_answer_with_disclaimer,
        test_memory_accumulates_turns,
        test_retriever_returns_relevant_condition,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
