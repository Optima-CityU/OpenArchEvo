"""Interfaces for the two language-model roles in the search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Proposal:
    """One LLM proposal: a named program with its design rationale."""

    name: str
    rationale: str
    code: str


class ProgramGenerator(Protocol):
    """Generates a new program from a system and a user prompt."""

    def propose(self, system_prompt: str, user_prompt: str) -> Proposal: ...


class TextEmbedder(Protocol):
    """Embeds a design rationale into a fixed-size vector."""

    def embed(self, text: str) -> list[float]: ...
