"""
MedAssist agentic orchestrator.

This is the heart of the "agentic AI" part of the project. Rather than
sending every user message straight to an LLM (a plain chatbot), the agent
follows an explicit perceive -> decide -> act loop, choosing at each turn
which tool(s) to invoke based on the current input and conversation state:

    1. PERCEIVE  - read the user's message and the running conversation memory
    2. DECIDE    - route the turn:
                     a. Is this a medical emergency (red flag)?  -> escalate now
                     b. Is the message too vague to act on?      -> ask a
                        clarifying follow-up question
                     c. Otherwise                                -> retrieve
                        relevant knowledge (RAG) and generate an answer
    3. ACT       - call the corresponding tool(s) and, where relevant, the
                   LLM backend to produce a response
    4. REMEMBER  - append the turn to conversation memory so future turns
                   have context (e.g. a follow-up answer to a clarifying
                   question)

This ReAct-style "router + tools" pattern is the same idea used by
frameworks such as LangChain/LangGraph agents; it is implemented here
directly in Python so the underlying mechanics are fully transparent for
academic/demonstration purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from .llm_backend import LLMBackend, get_default_backend
from .retriever import KBEntry, MedicalKBRetriever
from .tools import get_disclaimer, needs_clarification, red_flag_check, retrieve_medical_info

SYSTEM_PROMPT = (
    "You are MedAssist, a cautious healthcare information assistant. "
    "You are NOT a doctor and must never provide a definitive diagnosis or "
    "prescribe medication. Answer using ONLY the provided context. Be clear, "
    "empathetic, and concise. Always recommend professional consultation for "
    "anything serious or uncertain."
)

TurnType = Literal["emergency", "clarification", "answer"]


@dataclass
class AgentTurn:
    """A structured record of one agent decision, useful for the report/demo."""
    turn_type: TurnType
    user_message: str
    tools_used: List[str]
    retrieved_conditions: List[str]
    response: str


@dataclass
class ConversationMemory:
    turns: List[AgentTurn] = field(default_factory=list)

    def add(self, turn: AgentTurn) -> None:
        self.turns.append(turn)

    def as_transcript(self) -> str:
        lines = []
        for t in self.turns:
            lines.append(f"User: {t.user_message}")
            lines.append(f"MedAssist [{t.turn_type}]: {t.response}")
        return "\n".join(lines)


class MedAssistAgent:
    def __init__(self, kb_path: str, backend: LLMBackend | None = None):
        self.retriever = MedicalKBRetriever(kb_path)
        self.backend = backend or get_default_backend()
        self.memory = ConversationMemory()

    def handle(self, user_message: str) -> AgentTurn:
        """Run one full perceive -> decide -> act -> remember cycle."""

        # --- DECIDE step 1: safety-critical red-flag routing -------------
        flag_result = red_flag_check(user_message, self.retriever)
        if flag_result.triggered:
            if flag_result.matched_entries:
                advice = " ".join(e.advice for e in flag_result.matched_entries)
                conditions = [e.condition for e in flag_result.matched_entries]
            else:
                advice = (
                    "This sounds like it could be a medical emergency. Please call "
                    "your local emergency number or go to the nearest emergency "
                    "room right away."
                )
                conditions = []
            response = f"\U0001F6A8 EMERGENCY WARNING: {advice}"
            turn = AgentTurn(
                turn_type="emergency",
                user_message=user_message,
                tools_used=["red_flag_check"],
                retrieved_conditions=conditions,
                response=response,
            )
            self.memory.add(turn)
            return turn

        # --- DECIDE step 2: is the message too vague to act on? ----------
        if needs_clarification(user_message):
            response = (
                "Could you tell me a bit more? For example: what symptoms are "
                "you experiencing, how long they've lasted, and how severe they "
                "feel? This helps me give you more relevant information."
            )
            turn = AgentTurn(
                turn_type="clarification",
                user_message=user_message,
                tools_used=["needs_clarification"],
                retrieved_conditions=[],
                response=response,
            )
            self.memory.add(turn)
            return turn

        # --- DECIDE step 3: standard RAG answer ---------------------------
        retrieved: List[KBEntry] = retrieve_medical_info(user_message, self.retriever, top_k=3)
        context = "\n\n".join(e.as_context() for e in retrieved)

        user_prompt = (
            f"CONTEXT:\n{context if context else '(no matching entries found)'}\n\n"
            f"USER QUESTION:\n{user_message}"
        )
        generated = self.backend.generate(SYSTEM_PROMPT, user_prompt)
        response = f"{generated}\n\n{get_disclaimer()}"

        turn = AgentTurn(
            turn_type="answer",
            user_message=user_message,
            tools_used=["retrieve_medical_info", f"llm:{self.backend.name}", "get_disclaimer"],
            retrieved_conditions=[e.condition for e in retrieved],
            response=response,
        )
        self.memory.add(turn)
        return turn
