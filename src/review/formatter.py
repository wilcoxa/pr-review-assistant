"""Format review output for GitHub PR comments."""

import logging

logger = logging.getLogger(__name__)


def format_review_comment(llm_review: str) -> str:
    """Format a review comment for a single file - file-specific content only."""
    return llm_review


def format_review_body(
    file_count: int,
    tools_used: list[str],
    total_findings: int,
    persona: str,
    summary: str | None = None,
    quality_observations: list[str] | None = None,
    test_observations: list[str] | None = None,
    hygiene_observations: list[str] | None = None,
) -> str:
    """Format the top-level review body including overall summary and PR-level observations."""
    parts = ["**Automated Code Review**\n"]
    parts.append(f"Reviewed **{file_count}** file(s)")
    if tools_used:
        tools_str = ", ".join(tools_used)
        parts.append(f" | Tools: {tools_str}")
        parts.append(f" | {total_findings} static analysis finding(s)")
    parts.append(f" | Mode: {persona}")
    body = "".join(parts)

    if summary:
        body += f"\n\n{summary}"

    if quality_observations:
        body += "\n\n#### PR Quality Notes\n"
        body += "\n".join(f"- {obs}" for obs in quality_observations)

    if test_observations:
        body += "\n\n#### Test Coverage\n"
        body += "\n".join(f"- {obs}" for obs in test_observations)

    if hygiene_observations:
        body += "\n\n#### Git Hygiene\n"
        body += "\n".join(f"- {obs}" for obs in hygiene_observations)

    return body
