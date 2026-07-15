"""
LLM client with Groq backend when GROQ_API_KEY is set,
and a data-driven mock mode when no key is available.

Mock mode is NOT a generic placeholder — it reads the actual
statistics passed in the prompt and produces dataset-specific text.
"""

import os
import json
import re


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.mock_mode = not bool(self.api_key)
        self._client = None

        if not self.mock_mode:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                print("  [LLM] Groq API connected — using llama-3.3-70b-versatile")
            except ImportError:
                print("  [LLM] groq package not installed; falling back to mock mode")
                self.mock_mode = True

        if self.mock_mode:
            print("  [LLM] Mock mode active (set GROQ_API_KEY to use real LLM)")

    @property
    def available(self) -> bool:
        return not self.mock_mode

    def chat(self, prompt: str, system: str = "You are an expert ML analyst.") -> str:
        if self.mock_mode:
            raise RuntimeError(
                "LLMClient.chat() called in mock mode — "
                "nodes should use their own mock builders instead."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        return resp.choices[0].message.content.strip()


# Module-level singleton used by all nodes
llm = LLMClient()
