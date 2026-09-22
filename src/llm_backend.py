"""
Pluggable LLM backend.

The agent's control logic (src/agent.py) only depends on the LLMBackend
interface's `generate()` method, so the underlying model can be swapped
freely:

  - OpenAIBackend    -> uses the `openai` SDK (needs OPENAI_API_KEY)
  - AnthropicBackend -> uses the `anthropic` SDK (needs ANTHROPIC_API_KEY)
  - MockBackend      -> deterministic, template-based generation that needs
                        NO API key or network access. This makes the whole
                        pipeline runnable end-to-end offline for grading,
                        CI, or demoing without incurring API costs, while
                        still exercising the full agent/tool/RAG pipeline.

get_default_backend() auto-selects the best available backend based on
which API keys are present in the environment.
"""

from __future__ import annotations

import os
import textwrap
from abc import ABC, abstractmethod
from typing import List


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # imported lazily so this is optional
        self.client = OpenAI()
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        import anthropic  # imported lazily so this is optional
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()


class MockBackend(LLMBackend):
    """
    A deterministic, offline stand-in for a real LLM call.

    It does not "understand" language the way a real LLM does; instead it
    composes a readable answer from the retrieved context using a fixed
    template. This lets every other part of the system (retrieval, tool
    routing, memory, red-flag safety checks) be developed, tested, and
    demonstrated without any API key — swap in OpenAIBackend/AnthropicBackend
    for real natural-language generation.
    """

    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Very small "extractive" synthesis: pull the context block back out
        # of the prompt and present it in a structured, conversational way.
        context = ""
        if "CONTEXT:" in user_prompt:
            context = user_prompt.split("CONTEXT:", 1)[1].split("USER QUESTION:")[0].strip()
        question = user_prompt.split("USER QUESTION:")[-1].strip()

        if not context:
            return (
                "I don't have specific information about that in my knowledge base. "
                "Could you describe your main symptoms in a bit more detail, or "
                "consider asking a licensed healthcare professional directly?"
            )

        summary = textwrap.shorten(context.replace("\n", " "), width=600, placeholder=" ...")
        return (
            f"Based on the information I have, here's what may be relevant to "
            f"\"{question}\":\n\n{summary}\n\n"
            "If your symptoms are severe, worsening, or you're unsure, please "
            "consult a doctor for a proper evaluation."
        )


def get_default_backend() -> LLMBackend:
    """Pick the best available backend based on environment configuration."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception:
            pass
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            pass
    return MockBackend()
