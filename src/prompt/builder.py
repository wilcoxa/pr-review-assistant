"""Prompt construction with token budgeting and truncation."""

import logging
from dataclasses import dataclass

from ..config import Config
from ..llm.base import LLMProvider
from .templates import (
    FOCUS_EMPHASIS,
    PERSONAS,
    SCORING_PROMPT,
    STANDARDIZED_CHECKLIST,
    detect_language,
)

ALL_FOCUS_AREAS = {"security", "quality", "performance", "education"}

logger = logging.getLogger(__name__)

# Token budget allocation
SYSTEM_MSG_BUDGET = 500
CONTEXT_BUDGET = 2000
TOOL_FINDINGS_BUDGET = 1500
MIN_CODE_BUDGET = 1000


@dataclass
class PromptParts:
    system_message: str
    user_message: str
    total_tokens: int


def build_system_message(config: Config) -> str:
    """Build the system message based on the configured persona."""
    persona = PERSONAS.get(config.review_persona, PERSONAS["normal"])
    return persona


def build_user_message(
    filename: str,
    contents: str,
    pr_description: str,
    comments: list[str],
    readme: str,
    tool_findings: str,
    config: Config,
    llm: LLMProvider,
) -> str:
    """Build the user message with token-aware truncation."""
    language = detect_language(filename) or "code"
    max_context = llm.max_context_tokens(config.openai_model)
    output_budget = config.openai_max_tokens
    available_tokens = max_context - output_budget - SYSTEM_MSG_BUDGET

    parts = []

    # File header
    parts.append(f"## File Under Review: `{filename}`")
    parts.append(f"**Language:** {language}\n")

    # PR Context (truncated to budget)
    context_text = _build_context(pr_description, comments, readme)
    context_text = _truncate_to_budget(context_text, CONTEXT_BUDGET, llm)
    parts.append(context_text)
    available_tokens -= llm.count_tokens(context_text)

    # Tool findings (truncated to budget)
    if tool_findings:
        findings_text = f"## Static Analysis Findings\n{tool_findings}"
        findings_text = _truncate_to_budget(findings_text, TOOL_FINDINGS_BUDGET, llm)
        parts.append(findings_text)
        available_tokens -= llm.count_tokens(findings_text)

        parts.append(
            "For each tool finding above:\n"
            "1. If valid and important, explain WHY it matters and HOW to fix it\n"
            "2. If it appears to be a false positive, explain why\n"
            "3. Also look for issues the tools MISSED (design, logic, architecture)\n"
        )

    # Code content (use remaining budget)
    code_budget = max(available_tokens - 500, MIN_CODE_BUDGET)  # reserve 500 for instructions
    code_text = f"## Code\n```{language.lower().split()[0] if language else ''}\n{contents}\n```"
    code_text = _truncate_to_budget(code_text, code_budget, llm)
    parts.append(code_text)

    # Review instructions
    instructions = _build_instructions(config)
    parts.append(instructions)

    # Scoring rubric
    if config.enable_scoring:
        parts.append(SCORING_PROMPT)

    return "\n\n".join(parts)


def build_prompt(
    filename: str,
    contents: str,
    pr_description: str,
    comments: list[str],
    readme: str,
    tool_findings: str,
    config: Config,
    llm: LLMProvider,
) -> PromptParts:
    """Build complete prompt with system and user messages."""
    system_msg = build_system_message(config)
    user_msg = build_user_message(
        filename, contents, pr_description, comments, readme,
        tool_findings, config, llm,
    )
    total = llm.count_tokens(system_msg) + llm.count_tokens(user_msg)
    return PromptParts(system_message=system_msg, user_message=user_msg, total_tokens=total)


def _build_context(pr_description: str, comments: list[str], readme: str) -> str:
    """Build the PR context section."""
    parts = ["## PR Context"]
    parts.append(f"**Description:** {pr_description}")

    if comments:
        recent = comments[-10:]  # Keep most recent 10 comments
        parts.append("**Key Comments:**")
        for c in recent:
            parts.append(f"- {c[:200]}")  # Truncate individual comments

    if readme and readme != "No README file found.":
        # Keep first 100 lines of README
        readme_lines = readme.split("\n")[:100]
        parts.append(f"**README (excerpt):**\n{''.join(line + chr(10) for line in readme_lines)}")

    return "\n".join(parts)


def _build_instructions(config: Config) -> str:
    """Build review instructions: standardized checklist + optional focus emphasis + custom notes."""
    parts = [
        "## Review Instructions",
        "Focus your review strictly on the specific file shown above. "
        "Do not include a PR-level overview or general summary — "
        "address only what is present in this file.",
        STANDARDIZED_CHECKLIST,
    ]

    selected = set(config.focus_areas)
    if selected and selected != ALL_FOCUS_AREAS:
        emphasis_lines = [FOCUS_EMPHASIS[a] for a in config.focus_areas if a in FOCUS_EMPHASIS]
        if emphasis_lines:
            parts.append("### Focus Emphasis\n" + "\n".join(f"- {line}" for line in emphasis_lines))

    if config.severity_threshold.lower() not in ("low", ""):
        parts.append(
            f"### Severity Filter\n"
            f"Only report issues at **{config.severity_threshold}** severity or above. "
            f"Do not include low-severity or informational observations in your feedback."
        )

    if config.custom_instructions:
        parts.append(f"### Additional Instructions\n{config.custom_instructions}")

    return "\n\n".join(parts)


def _truncate_to_budget(text: str, max_tokens: int, llm: LLMProvider) -> str:
    """Truncate text to fit within a token budget."""
    token_count = llm.count_tokens(text)
    if token_count <= max_tokens:
        return text

    # Binary search for the right truncation point
    lines = text.split("\n")
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = "\n".join(lines[:mid])
        if llm.count_tokens(candidate) <= max_tokens - 20:  # Reserve space for truncation notice
            low = mid
        else:
            high = mid - 1

    truncated = "\n".join(lines[:low])
    omitted = len(lines) - low
    if omitted > 0:
        truncated += f"\n\n*[{omitted} lines truncated to fit token budget]*"

    return truncated
