"""
ASTRA Code Execution Sandbox — runs Python code blocks safely.

In production: Docker + gVisor (runsc) with network disabled.
In development: restricted exec() with timeout (no subprocess/os/socket).

Safety rules (mirrors system prompt Section 3):
  - Never imports: os, subprocess, socket, ctypes, importlib, sys.modules
  - Never uses: eval(), exec() with external strings, __import__()
  - Only writes to ./data/temp and ./output
  - No network inside sandbox
  - 120s timeout, 3 retries
"""
from __future__ import annotations

import ast
import io
import sys
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from loguru import logger

BLOCKED_IMPORTS = frozenset(
    ["os", "subprocess", "socket", "ctypes", "importlib", "sys", "builtins"]
)

BLOCKED_BUILTINS = frozenset(["eval", "exec", "__import__", "compile", "open"])


def _check_ast_safety(code: str) -> list[str]:
    """Static AST analysis for blocked patterns. Returns list of violations."""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0]] if node.module else []
            )
            for name in names:
                if name in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {name}")
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in BLOCKED_BUILTINS:
                violations.append(f"Blocked call: {func_name}()")
    return violations


class SandboxResult:
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        success: bool = True,
        error: str = "",
        local_vars: dict[str, Any] | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.error = error
        self.local_vars = local_vars or {}


def execute_code(
    code: str,
    globals_dict: dict[str, Any] | None = None,
    timeout_seconds: int = 120,
    max_retries: int = 3,
) -> SandboxResult:
    """
    Execute a Python code block in the restricted sandbox.

    Args:
        code: Python source to execute.
        globals_dict: Pre-populated globals (e.g., loaded data).
        timeout_seconds: Max execution time.
        max_retries: Retry count on error before giving up.

    Returns:
        SandboxResult with stdout, stderr, success flag, and local_vars.
    """
    code = textwrap.dedent(code)

    # Static safety check
    violations = _check_ast_safety(code)
    if violations:
        msg = f"Safety violations: {'; '.join(violations)}"
        logger.warning(f"Sandbox blocked code — {msg}")
        return SandboxResult(success=False, error=msg)

    gdict: dict[str, Any] = {
        "__builtins__": {
            k: v
            for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k not in BLOCKED_BUILTINS
        }
        if isinstance(__builtins__, dict)
        else {},
    }
    if globals_dict:
        gdict.update(globals_dict)

    local_vars: dict[str, Any] = {}
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def _run() -> None:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            try:
                exec(compile(code, "<astra_sandbox>", "exec"), gdict, local_vars)  # noqa: S102
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                future.result(timeout=timeout_seconds)

            return SandboxResult(
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                success=True,
                local_vars=local_vars,
            )

        except FuturesTimeout:
            error = f"Sandbox timeout after {timeout_seconds}s (attempt {attempt}/{max_retries})"
            logger.warning(error)
            if attempt >= max_retries:
                return SandboxResult(success=False, error=error)

        except Exception as exc:
            tb_lines = traceback.format_exc().splitlines()
            # Summarize to 3 lines max (per system prompt rules)
            short_tb = "\n".join(tb_lines[-3:])
            error = f"Execution error (attempt {attempt}/{max_retries}): {short_tb}"
            logger.warning(error)
            if attempt >= max_retries:
                return SandboxResult(
                    stdout=stdout_buf.getvalue(),
                    stderr=stderr_buf.getvalue() + "\n" + error,
                    success=False,
                    error=error,
                    local_vars=local_vars,
                )

    return SandboxResult(success=False, error="Max retries exceeded")
