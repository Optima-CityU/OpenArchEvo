"""Language-model roles: program generation (with rationale) and rationale embedding."""

from openarchevo.llm.base import ProgramGenerator, Proposal, TextEmbedder
from openarchevo.llm.fake import FakeLLM, HashEmbedder
from openarchevo.llm.openai_compat import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from openarchevo.llm.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "FakeLLM", "HashEmbedder", "OpenAICompatibleEmbedder", "OpenAICompatibleLLM",
    "ProgramGenerator", "Proposal", "SYSTEM_PROMPT", "TextEmbedder", "build_user_prompt",
]
