"""
ASTRA Rich Logging — Structured console output for pipeline progress.
Replaces unreadable JSON loguru file logs with human-readable, coloured output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Rich is already a dependency (used in main.py)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box as rich_box

    _console = Console(stderr=True, highlight=False)
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    _console = None  # type: ignore


# ─── Log sink ─────────────────────────────────────────────────────────────────

def _rich_sink(message) -> None:
    """Custom loguru sink that formats messages with Rich markup."""
    if not _RICH_AVAILABLE or _console is None:
        print(str(message), end="", file=sys.stderr)
        return

    record = message.record
    level = record["level"].name
    msg = record["message"]
    time_str = record["time"].strftime("%H:%M:%S")

    # Node boundary — make prominently visible
    if msg.startswith("[Node "):
        _console.print(
            f"\n[bold blue]━━━ {time_str}[/bold blue] "
            f"[bold white on blue] {msg} [/bold white on blue]"
        )
    elif msg.startswith("[Layer "):
        _console.print(f"  [dim]{time_str}[/dim]  [cyan]{msg}[/cyan]")
    elif msg.startswith("[ASTRA]"):
        _console.print(f"[bold magenta]{time_str}[/bold magenta] [magenta]{msg}[/magenta]")
    elif msg.startswith("[Graph]"):
        _console.print(f"[dim]{time_str}[/dim] [dim cyan]{msg}[/dim cyan]")
    elif level == "WARNING":
        _console.print(f"  [dim]{time_str}[/dim] [yellow]⚠  {msg}[/yellow]")
    elif level == "ERROR":
        _console.print(f"  [dim]{time_str}[/dim] [bold red]✗  {msg}[/bold red]")
    elif level == "SUCCESS":
        _console.print(f"  [dim]{time_str}[/dim] [bold green]✓  {msg}[/bold green]")
    elif level == "DEBUG":
        _console.print(f"  [dim]{time_str}[/dim] [dim]{msg}[/dim]")
    else:
        _console.print(f"  [dim]{time_str}[/dim]  {msg}")

    # Flush so output appears immediately
    try:
        sys.stderr.flush()
    except Exception:
        pass


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_rich_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    session_log_file: Optional[str] = None,
) -> None:
    """
    Configure loguru with Rich console + human-readable file logs.

    Args:
        log_level: Console log level.
        log_file: Global log file path (human-readable text, not JSON).
        session_log_file: Per-session log file inside the session dir.
    """
    logger.remove()

    # Rich console sink
    logger.add(
        _rich_sink,
        format="{message}",
        level=log_level,
        colorize=False,
    )

    # Global log file (human-readable)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
            level="DEBUG",
            rotation="50 MB",
            retention="7 days",
            serialize=False,   # Human-readable, not JSON
        )

    # Per-session log file
    if session_log_file:
        Path(session_log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            session_log_file,
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            serialize=False,
        )


def add_session_log(session_log_path: str) -> None:
    """Add a per-session log file sink (call after session dir is created)."""
    Path(session_log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        session_log_path,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        serialize=False,
    )


# ─── Pipeline progress helpers ────────────────────────────────────────────────

def print_pipeline_start(query: str, session_dir: Optional[str] = None) -> None:
    """Print the ASTRA pipeline start banner."""
    if not _RICH_AVAILABLE or _console is None:
        print(f"\n{'='*60}\nASTRA Research Session\nQuery: {query}\n{'='*60}\n",
              file=sys.stderr)
        return

    lines = [
        f"[bold cyan]ASTRA Research Session[/bold cyan]",
        f"Query: [yellow]{query[:120]}[/yellow]",
    ]
    if session_dir:
        lines.append(f"Session: [dim]{session_dir}[/dim]")

    _console.print()
    _console.print(Panel("\n".join(lines), border_style="cyan", padding=(0, 2)))
    _console.print()


def print_evaluation_scores(evaluation_results: dict) -> None:
    """Print evaluation scores as a Rich table."""
    if not _RICH_AVAILABLE or _console is None or not evaluation_results:
        return

    table = Table(
        title="Section Quality Scores",
        box=rich_box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Section", style="cyan", no_wrap=False, max_width=32)
    table.add_column("Overall", justify="center", style="bold")
    table.add_column("Factual", justify="center")
    table.add_column("Complete", justify="center")
    table.add_column("Coherence", justify="center")
    table.add_column("Needs Fix", justify="center")

    for title, result in evaluation_results.items():
        score = result.get("overall_score", 0)
        color = "green" if score >= 0.75 else ("yellow" if score >= 0.6 else "red")
        needs = result.get("needs_refinement", False)

        table.add_row(
            title[:32],
            f"[{color}]{score:.2f}[/{color}]",
            f"{result.get('factual_accuracy', 0):.2f}",
            f"{result.get('completeness', 0):.2f}",
            f"{result.get('coherence', 0):.2f}",
            "[red]YES[/red]" if needs else "[green]ok[/green]",
        )

    _console.print()
    _console.print(table)
    _console.print()


def print_session_summary(final_state: dict) -> None:
    """Print final session summary panel."""
    if not _RICH_AVAILABLE or _console is None:
        return

    lines = [
        f"[bold green]Research Complete![/bold green]",
        f"Words:    [cyan]{final_state.get('final_word_count', 0):,}[/cyan]",
        f"Sections: [cyan]{len(final_state.get('draft_sections', {})):,}[/cyan]",
        f"Sources:  [cyan]{len(final_state.get('collected_sources', [])):,}[/cyan]",
        f"Iterations: [cyan]{final_state.get('iteration', 0)}[/cyan]",
    ]

    output_lines = []
    if d := final_state.get("session_output_dir"):
        output_lines.append(f"[bold]Session dir:[/bold] {d}")
    if p := final_state.get("final_html_path"):
        output_lines.append(f"[bold]HTML:[/bold]        {p}")
    if p := final_state.get("final_pdf_path"):
        output_lines.append(f"[bold]PDF:[/bold]         {p}")
    if p := final_state.get("final_md_path"):
        output_lines.append(f"[bold]Markdown:[/bold]    {p}")

    if output_lines:
        lines.append("")
        lines.extend(output_lines)

    _console.print()
    _console.print(Panel("\n".join(lines), border_style="green", padding=(0, 2)))
    _console.print()
