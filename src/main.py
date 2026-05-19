"""Main orchestrator — ties together all components."""

import logging
import os
import sys
from urllib.parse import quote

from .config import Config, load_config
from .github_client import (
    get_repo_and_pull,
    files_for_review,
    fetch_contextual_info,
    get_file_content,
    build_summary_review_body,
)
from .llm.base import LLMConfig
from .prompt.builder import build_prompt, build_summary_prompt
from .review.formatter import format_review_comment, format_review_body
from .tools.base import format_findings_for_prompt
from .tools.registry import get_tools_for_config
from .tools.runner import run_tools
from .tools.stack_detector import detect_stack
from .checks.pr_quality import check_pr_quality
from .checks.test_coverage import analyze_test_coverage
from .checks.git_hygiene import check_git_hygiene

logger = logging.getLogger(__name__)


def create_llm_provider(config: Config):
    """Create the appropriate LLM provider based on configuration."""
    if config.llm_provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=config.anthropic_api_key)
    else:
        from .llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.api_base_url or None,
        )


def main():
    config = load_config()

    # Setup logging
    logging.basicConfig(
        encoding="utf-8",
        level=getattr(logging, config.logging_level.upper(), logging.WARNING),
        format="%(levelname)s: %(name)s: %(message)s",
    )

    if not config.openai_api_key and config.llm_provider == "openai":
        logger.error("OpenAI API key is required")
        sys.exit(1)
    if not config.github_token:
        logger.error("GitHub token is required")
        sys.exit(1)
    if not config.github_pr_id:
        logger.error("GitHub PR ID is required")
        sys.exit(1)

    # Initialize LLM provider
    llm = create_llm_provider(config)
    llm_config = LLMConfig(
        model=config.openai_model,
        temperature=config.openai_temperature,
        max_tokens=config.openai_max_tokens,
    )

    # Initialize GitHub
    repo_name = os.getenv("GITHUB_REPOSITORY", "")
    repo, pull = get_repo_and_pull(config.github_token, repo_name, config.github_pr_id)

    # Collect files for review
    files = files_for_review(pull, config.file_patterns)
    n_files = len(files)
    if n_files == 0:
        logger.info("No files to review")
        return
    if n_files > config.max_files:
        logger.error(
            f"Too many files to review ({n_files}), limit is {config.max_files}. "
            "Use the 'files' input to target specific files."
        )
        sys.exit(1)

    logger.info(f"Reviewing {n_files} file(s)")

    # Fetch PR context
    pr_description, pr_comments, readme = fetch_contextual_info(pull, repo)

    # Run static analysis tools
    workspace = os.environ.get("GITHUB_WORKSPACE", ".")
    changed_filenames = list(files.keys())
    tool_findings_map: dict[str, str] = {}  # filename -> formatted findings
    tools_used: list[str] = []
    total_findings = 0

    if config.tools != "none":
        # Detect tech stack
        detected_languages = detect_stack(changed_filenames, workspace)
        logger.info(f"Detected languages: {detected_languages}")

        # Get tools to run
        selected_tools = get_tools_for_config(
            detected_languages, config.tools, config.tools_list or None,
        )
        tools_used = [t.name for t in selected_tools]

        if selected_tools:
            # Validate workspace has repo checkout
            if not os.path.exists(os.path.join(workspace, ".git")):
                logger.warning(
                    "Repository not checked out in workspace. "
                    "Static analysis tools require 'actions/checkout' before this action. "
                    "Skipping tool analysis."
                )
            else:
                # Run all tools in parallel
                all_findings = run_tools(
                    selected_tools, changed_filenames, workspace,
                    config.tool_configs, config.severity_threshold,
                )
                total_findings = len(all_findings)
                logger.info(f"Total findings from tools: {total_findings}")

                # Group findings by file
                for finding in all_findings:
                    if finding.file not in tool_findings_map:
                        tool_findings_map[finding.file] = []
                    tool_findings_map[finding.file].append(finding)

                # Format findings per file
                tool_findings_map = {
                    filename: format_findings_for_prompt(findings_list)
                    for filename, findings_list in tool_findings_map.items()
                }

    # Run quality checks
    quality_observations = check_pr_quality(pull, n_files) if "education" in config.focus_areas else []
    test_observations = analyze_test_coverage(changed_filenames, workspace)
    hygiene_observations = check_git_hygiene(pull, files, workspace)

    # Review each file
    comments = []

    for filename, commit_info in files.items():
        commit_sha = commit_info["sha"]
        content = get_file_content(repo, filename, commit_sha)
        if not content:
            logger.info(f"Skipping {filename}: empty or unreadable")
            continue

        # Build prompt with tool findings for this file
        file_findings = tool_findings_map.get(filename, "")
        prompt = build_prompt(
            filename, content, pr_description, pr_comments, readme,
            file_findings, config, llm,
        )

        logger.info(f"Reviewing {filename} ({prompt.total_tokens} tokens)")

        # Call LLM
        try:
            llm_review = llm.complete(
                prompt.system_message, prompt.user_message, llm_config,
            )
        except Exception as e:
            logger.error(f"LLM review failed for {filename}: {e}")
            continue

        comments.append({
            "path": quote(filename, safe="/"),
            "position": 1,
            "body": format_review_comment(llm_review),
        })

    if comments:
        from urllib.parse import unquote
        summary = None
        file_reviews = [(unquote(c["path"]), c["body"]) for c in comments]
        summary_prompt = build_summary_prompt(pr_description, file_reviews, config, llm)
        try:
            summary = llm.complete(summary_prompt.system_message, summary_prompt.user_message, llm_config)
            logger.info("Generated overall PR summary")
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")

        review_body = format_review_body(
            len(comments), tools_used, total_findings, config.review_persona,
            summary=summary,
            quality_observations=quality_observations,
            test_observations=test_observations,
            hygiene_observations=hygiene_observations,
        )
        body = build_summary_review_body(review_body, comments)
        pull.create_review(body=body, event="COMMENT")
        logger.info(f"Posted consolidated review for {len(comments)} file(s)")
    else:
        logger.info("No review comments to post")


if __name__ == "__main__":
    main()
