"""
ASTRA LangSmith Observability — tracing setup and decorators.

Every LangGraph node is traced with:
  - session_id, layer, paradigm, iteration
  - run name: "{layer_name}:{operation}:{session_id[:8]}"
"""
from __future__ import annotations

import os
import functools
from typing import Any, Callable, Optional

from loguru import logger


def setup_langsmith(
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Configure LangSmith tracing environment variables.
    Returns True if LangSmith is properly configured.
    """
    from astra.config import get_config

    cfg = get_config()
    api_key = api_key or cfg.langchain_api_key
    project = project or cfg.langchain_project

    if not api_key:
        logger.warning("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = str(cfg.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_HIDE_INPUTS"] = "false"
    os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "false"

    logger.info(f"LangSmith tracing enabled → project: {project}")
    return True


def make_run_name(layer_name: str, operation: str, session_id: str) -> str:
    """Format: '{layer_name}:{operation}:{session_id[:8]}'"""
    return f"{layer_name}:{operation}:{session_id[:8]}"


def traced_node(
    layer: int,
    layer_name: str,
    paradigm: str = "hybrid",
) -> Callable:
    """
    Decorator for LangGraph nodes that adds LangSmith metadata.

    Usage:
        @traced_node(layer=2, layer_name="crawler", paradigm="hybrid")
        def my_node(state: AstraState) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> Any:
            session_id = state.get("session_id", "unknown")
            run_name = make_run_name(layer_name, fn.__name__, session_id)
            iteration = state.get("iteration", 0)

            # Add metadata to LangSmith run via context
            try:
                from langsmith import traceable
                @traceable(
                    name=run_name,
                    metadata={
                        "session_id": session_id,
                        "layer": layer,
                        "paradigm": paradigm,
                        "iteration": iteration,
                    },
                )
                def _inner(*a: Any, **kw: Any) -> Any:
                    return fn(*a, **kw)
                return _inner(state, *args, **kwargs)
            except ImportError:
                return fn(state, *args, **kwargs)

        return wrapper
    return decorator


def get_tracer(session_id: str, layer: int, paradigm: str = "json"):
    """
    Returns a LangSmith tracer context manager for manual tracing.
    Falls back to a no-op context if LangSmith is unavailable.
    """
    try:
        from langsmith import trace

        class _Ctx:
            def __init__(self) -> None:
                self._trace = trace(
                    name=f"layer{layer}:{session_id[:8]}",
                    metadata={"session_id": session_id, "layer": layer, "paradigm": paradigm},
                )

            def __enter__(self):
                return self._trace.__enter__()

            def __exit__(self, *exc):
                return self._trace.__exit__(*exc)

        return _Ctx()
    except Exception:
        class _NoOp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return _NoOp()
