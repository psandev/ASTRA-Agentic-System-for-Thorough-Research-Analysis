"""
ASTRA — Entry Point

Usage:
  # Run Gradio UI (default)
  python main.py

  # Run a single research session from CLI
  python main.py --query "Your research topic here"

  # Test a specific layer
  python main.py --test-layer 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from loguru import logger


def check_vllm_servers() -> None:
    """
    Check vLLM server health and auto-start servers if they are not running.
    Called at startup before LLM imports so models are available for the pipeline.
    Reads VLLM_BASE_URL / ASTRA_VISION_BASE_URL to determine health endpoints.
    Reads ASTRA_MAIN_MODEL / ASTRA_VISION_MODEL for the actual model paths to serve.
    """
    import os
    import subprocess
    import time

    import httpx
    from rich.console import Console

    console = Console()

    main_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    vision_base_url = os.environ.get("ASTRA_VISION_BASE_URL", "http://localhost:8001/v1")
    main_model = os.environ.get("ASTRA_MAIN_MODEL", "")
    vision_model = os.environ.get("ASTRA_VISION_MODEL", "")

    def _health_url(base_url: str) -> str:
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url + "/health"

    main_health = _health_url(main_base_url)
    vision_health = _health_url(vision_base_url)

    def _is_up(url: str) -> bool:
        try:
            r = httpx.get(url, timeout=3.0)
            return r.status_code < 500
        except Exception:
            return False

    def _wait_for_server(health_url: str, label: str, timeout: int = 180) -> bool:
        console.print(f"[cyan]⏳ Waiting for {label} to be ready (up to {timeout}s)...[/cyan]")
        for _ in range(timeout // 5):
            time.sleep(5)
            if _is_up(health_url):
                return True
        return False

    # ── Main LLM ─────────────────────────────────────────────────────────────
    if _is_up(main_health):
        console.print("[green]✓ Main vLLM server is up (port 8000)[/green]")
    elif main_model:
        console.print(
            f"[yellow]⚠ Main vLLM (port 8000) is down — auto-starting {main_model}...[/yellow]"
        )
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 vllm serve {main_model} "
            "--gpu-memory-utilization 0.50 --tensor-parallel-size 1 "
            "--port 8000 --served-model-name astra-main "
            "--reasoning-parser qwen3 --enforce-eager "
            "--trust-remote-code --max-model-len 8192"
        )
        subprocess.Popen(cmd, shell=True)
        if _wait_for_server(main_health, "main vLLM"):
            console.print("[green]✓ Main vLLM server is up.[/green]")
        else:
            console.print(
                "[red]✗ Main vLLM server did not start in time. "
                "Layers 1, 5, 6 may be limited.[/red]"
            )
    else:
        console.print(
            "[yellow]⚠ Main vLLM (port 8000) is down. "
            "Set ASTRA_MAIN_MODEL env var to enable auto-start.[/yellow]"
        )

    # ── Vision LLM ───────────────────────────────────────────────────────────
    if _is_up(vision_health):
        console.print("[green]✓ Vision vLLM server is up (port 8001)[/green]")
    elif vision_model:
        console.print(
            f"[yellow]⚠ Vision vLLM (port 8001) is down — auto-starting {vision_model}...[/yellow]"
        )
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 vllm serve {vision_model} "
            "--gpu-memory-utilization 0.46 --max-model-len 4096 "
            "--enforce-eager --port 8001 --served-model-name astra-vision "
            '--limit-mm-per-prompt \'{"image": 1}\''
        )
        subprocess.Popen(cmd, shell=True)
        if _wait_for_server(vision_health, "vision vLLM"):
            console.print("[green]✓ Vision vLLM server is up.[/green]")
        else:
            console.print(
                "[red]✗ Vision vLLM did not start in time. "
                "Vision features will be disabled.[/red]"
            )
    else:
        console.print(
            "[yellow]⚠ Vision vLLM (port 8001) is down. "
            "Set ASTRA_VISION_MODEL env var to enable auto-start.[/yellow]"
        )


def setup_logging() -> None:
    from astra.config import get_config
    from astra.utils.rich_log import setup_rich_logging

    cfg = get_config()
    setup_rich_logging(
        log_level=cfg.astra_log_level,
        log_file=cfg.astra_log_file,
    )


def run_cli_research(query: str) -> None:
    from rich.console import Console
    from rich.table import Table

    from astra.graph import run_query_analysis, run_research
    from astra.utils.rich_log import print_pipeline_start, print_session_summary

    console = Console()

    # ── Phase 1: Analyze query ────────────────────────────────────────────────
    console.print(f"\n[bold cyan]🔍 Analyzing query...[/bold cyan]")
    analysis = run_query_analysis(query)

    enriched_query = analysis.get("enriched_query", query)
    section_outline = analysis.get("section_outline", [])
    clarifying_questions = analysis.get("clarifying_questions") or []
    implicit_needs = analysis.get("implicit_needs", [])

    console.print(f"\n[bold]Enriched query:[/bold] {enriched_query}")
    console.print(
        f"[dim]Expertise: {analysis.get('expertise_level')} | "
        f"Purpose: {analysis.get('purpose')}[/dim]"
    )

    if implicit_needs:
        console.print(f"[dim]Implicit needs: {', '.join(implicit_needs)}[/dim]")

    # Show section outline as a Rich table
    if section_outline:
        tbl = Table(title="Proposed Section Outline", show_lines=True)
        tbl.add_column("#", style="dim", width=4)
        tbl.add_column("Section Title", style="bold")
        for i, title in enumerate(section_outline, 1):
            tbl.add_row(str(i), title)
        console.print(tbl)

    # Ask clarifying questions
    clarif_dict: dict[str, str] = {}
    for cq in clarifying_questions:
        try:
            answer = input(f"\n[ASTRA] {cq}\nYour answer (Enter to skip): ").strip()
            if answer:
                clarif_dict[cq] = answer
        except (EOFError, KeyboardInterrupt):
            break

    # Confirm the enriched query or let user override
    try:
        user_query_override = input(
            f"\nResearch query [Enter to use enriched]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        user_query_override = ""

    final_query = user_query_override if user_query_override else enriched_query

    # ── Phase 2: Run full pipeline ────────────────────────────────────────────
    print_pipeline_start(final_query)
    result = run_research(final_query, clarifications=clarif_dict if clarif_dict else None)
    print_session_summary(result)


def run_layer_test(layer: int) -> None:
    """Run isolated test for a specific layer."""
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold cyan]Testing Layer {layer}...[/bold cyan]\n")

    if layer == 1:
        from tests.test_layer1 import run_tests
    elif layer == 2:
        from tests.test_layer2 import run_tests
    elif layer == 3:
        from tests.test_layer3 import run_tests
    elif layer == 4:
        from tests.test_layer4 import run_tests
    elif layer == 5:
        from tests.test_layer5 import run_tests
    elif layer == 6:
        from tests.test_layer6 import run_tests
    elif layer == 7:
        from tests.test_layer7 import run_tests
    else:
        console.print(f"[red]Unknown layer: {layer}[/red]")
        sys.exit(1)

    run_tests()


def main() -> None:
    check_vllm_servers()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="ASTRA — Agentic System for Thorough Research & Analysis"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Research query for CLI mode (skips Gradio UI)",
    )
    parser.add_argument(
        "--test-layer",
        type=int,
        metavar="N",
        help="Run isolated test for layer N (1-7)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Skip Gradio UI and run in CLI mode",
    )

    args = parser.parse_args()

    if args.test_layer:
        run_layer_test(args.test_layer)
        return

    if args.query:
        run_cli_research(args.query)
        return

    # Default: launch Gradio UI
    logger.info("Launching ASTRA Gradio UI...")
    import gradio as gr
    from app import build_ui, _UI_CSS
    from astra.config import get_config

    cfg = get_config()
    demo = build_ui()
    demo.queue(max_size=5)
    demo.launch(
        server_name=cfg.astra_gradio_host,
        server_port=cfg.astra_gradio_port,
        share=cfg.astra_gradio_share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_UI_CSS,
        allowed_paths=[cfg.astra_sessions_dir, cfg.astra_output_dir],
    )


if __name__ == "__main__":
    main()
