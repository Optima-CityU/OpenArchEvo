"""Prompt construction for program evolution.

The system and user templates (``templates/*.md``) are the ones listed in the
paper's appendix. The user template has one slot, ``{context}``, filled with the
sampled programs as numbered versions in increasing order of predicted fitness::

    [Version 1]
    ### Motivation
    <rationale>
    ### Program
    ```python ... ```

    We find that the below version outperforms [Version 1].
    [Version 2]
    ...

so the last version is the strongest program and the "outperforms" statements
are true. This order is the only performance information the LLM gets: no
predicted or measured metric values are included, even though the template
mentions results. Edit the templates to change the search space or task.
"""

from __future__ import annotations

from importlib import resources
from typing import Sequence

from openarchevo.candidate import Candidate


def _template(name: str) -> str:
    return resources.files("openarchevo.llm").joinpath("templates", name).read_text(encoding="utf-8")


SYSTEM_PROMPT = _template("system.md")
USER_TEMPLATE = _template("user.md")


def _version_block(index: int, candidate: Candidate, include_rationale: bool) -> str:
    parts = [f"[Version {index}]"]
    if include_rationale:
        parts += ["### Motivation", candidate.rationale.strip() or "(none)", "### Program"]
    parts.append(f"```python\n{candidate.code.strip()}\n```")
    return "\n".join(parts)


def build_user_prompt(parents: Sequence[Candidate], include_rationale: bool = True) -> str:
    """Fill the user template with parent programs, weakest (lowest ``yhat``) first."""
    if not parents:
        raise ValueError("At least one parent program is required.")
    ordered = sorted(parents, key=lambda c: c.yhat)
    blocks = [_version_block(1, ordered[0], include_rationale)]
    for i, cand in enumerate(ordered[1:], start=2):
        blocks.append(f"We find that the below version outperforms [Version {i - 1}].\n"
                      + _version_block(i, cand, include_rationale))
    return USER_TEMPLATE.replace("{context}", "\n\n".join(blocks))
