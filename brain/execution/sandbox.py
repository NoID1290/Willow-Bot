"""Module 3: AST Security Validator, Collectible Namespace Scope & LRU-64 ScriptSandbox."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
import inspect
import logging
import re
import time
from typing import Any, Callable

from hardware_opt.llm_client import CancellationToken, NONE_TOKEN, OperationCanceledError
from hardware_opt.memory_pool import CompiledSkillCache

from .bot_actions import ActionResult, CombatTactic, WillowBotActions
from rl_rewards.telemetry_snapshot import Vector3

LOG = logging.getLogger("localai.sandbox")

FORBIDDEN_IDENTIFIERS = frozenset(
    {
        # File system / OS / Process
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "io",
        "open",
        "fs",
        "child_process",
        "System",
        "Environment",
        # Network sockets
        "socket",
        "http",
        "urllib",
        "requests",
        "aiohttp",
        "httpx",
        "Net",
        # Reflection / Dynamic Code / Exit
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "exit",
        "quit",
        "Reflection",
        # Blocking thread sleep
        "Thread",
        "time",
    }
)

FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "sleep",
        "Sleep",
        "Exit",
        "exit",
        "popen",
        "system",
        "spawn",
        "fork",
        "open_connection",
        "create_subprocess_exec",
        "create_subprocess_shell",
    }
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class SandboxSecurityError(ValueError):
    """Raised when static AST inspection rejects unsafe, blocking, or unbounded LLM code."""


@dataclass(frozen=True, slots=True)
class SandboxExecutionOutcome:
    action_result: ActionResult
    compilation_failed: bool
    security_violation: str
    runtime_exception: str
    cache_hit: bool
    duration_ms: int


def _loop_contains_cooperative_yield(loop_node: ast.For | ast.While | ast.AsyncFor) -> bool:
    """Checks that a loop directly contains an `await` or `ct.ThrowIfCancellationRequested()` check."""
    for child in ast.walk(loop_node):
        if child is loop_node:
            continue
        if isinstance(child, ast.Await):
            return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr in ("ThrowIfCancellationRequested", "throw_if_cancellation_requested"):
                return True
    return False


def _sanitize_minor_syntax(source: str) -> str:
    """Strips markdown fences so 7B coder models' fenced code blocks compile cleanly."""
    text = (source or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate_skill_ast(source_code: str) -> tuple[ast.Module, str]:
    """Parses source into AST, rejects unsafe constructs, and returns (wrapped_ast, entry_fn_name)."""
    cleaned = _sanitize_minor_syntax(source_code)
    if not cleaned:
        raise SandboxSecurityError("Empty skill code.")
    if len(cleaned.encode("utf-8")) > 16_384:
        raise SandboxSecurityError("Skill source exceeds 16KB safety cap.")

    # Quick lexical guard for C#/JS unbounded `for(;;)` or `System.IO` before Python AST parse
    if re.search(r"\bfor\s*\(\s*;\s*;", cleaned):
        raise SandboxSecurityError("Unbounded synchronous for(;;) loop is forbidden.")

    try:
        tree = ast.parse(cleaned, filename="<voyager_skill>", mode="exec")
    except SyntaxError as exc:
        # Check if body-only snippet containing `await` was emitted without `async def` header
        indented = "\n".join("    " + line for line in cleaned.splitlines())
        wrapped_src = f"async def __synthesized_skill__(bot, ct):\n{indented}\n"
        try:
            tree = ast.parse(wrapped_src, filename="<voyager_skill>", mode="exec")
        except SyntaxError:
            raise SandboxSecurityError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc

    entry_fn_name = ""
    top_async_defs = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]
    top_sync_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]

    if not top_async_defs and not top_sync_defs:
        indented = "\n".join("    " + line for line in cleaned.splitlines())
        wrapped_src = f"async def __synthesized_skill__(bot, ct):\n{indented}\n"
        tree = ast.parse(wrapped_src, filename="<voyager_skill>", mode="exec")
        top_async_defs = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]

    if top_async_defs:
        entry_fn_name = top_async_defs[0].name
    elif top_sync_defs:
        entry_fn_name = top_sync_defs[0].name

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxSecurityError("Imports (os, sys, subprocess, network, etc.) are strictly forbidden.")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_IDENTIFIERS or node.id.startswith("__"):
                raise SandboxSecurityError(f"Forbidden identifier '{node.id}' in sandbox.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise SandboxSecurityError(f"Forbidden attribute access '.{node.attr}' in sandbox.")
            if node.attr in ("IO", "Net", "Reflection"):
                raise SandboxSecurityError(f"Forbidden namespace '.{node.attr}' in sandbox.")
        elif isinstance(node, (ast.While, ast.For)):
            if not _loop_contains_cooperative_yield(node):
                raise SandboxSecurityError(
                    "Unbounded synchronous loop without `await` or `ct.ThrowIfCancellationRequested()` is forbidden."
                )
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise SandboxSecurityError(f"Unsupported statement '{type(node).__name__}' in skill sandbox.")

    return tree, entry_fn_name


class ScriptSandbox:
    """Executes AST-verified async skills with SHA256 LRU-64 delegate caching and collectible scopes."""

    def __init__(
        self,
        cache: CompiledSkillCache | None = None,
        *,
        default_timeout_ms: int = 45_000,
    ) -> None:
        self.cache = cache or CompiledSkillCache()
        self.default_timeout_ms = max(500, int(default_timeout_ms))
        self._global_skills: dict[str, Callable[..., Any]] = {}
        self._lock_skills = asyncio.Lock()

    def register_composed_skill(self, skill_name: str, fn: Callable[..., Any]) -> None:
        """Exposes verified skills into sandbox global scope for Voyager skill composition."""
        if skill_name and skill_name.isidentifier() and skill_name not in FORBIDDEN_IDENTIFIERS:
            self._global_skills[skill_name] = fn

    def unregister_composed_skill(self, skill_name: str) -> None:
        self._global_skills.pop(skill_name, None)

    def compile_skill(self, source_code: str, skill_name_hint: str = "") -> tuple[Callable[..., Any], bool, str]:
        """Returns (callable, cache_hit, sha256_hash) or raises SandboxSecurityError."""
        code_hash = CompiledSkillCache.compute_hash(source_code)
        cached = self.cache.get(code_hash)
        if cached is not None:
            return cached.fn, True, code_hash

        tree, entry_fn_name = validate_skill_ast(source_code)
        compiled_code = compile(tree, filename=f"<skill:{skill_name_hint or entry_fn_name}>", mode="exec")

        isolated_scope: dict[str, Any] = {
            "__builtins__": dict(SAFE_BUILTINS),
            "ActionResult": ActionResult,
            "Vector3": Vector3,
            "CombatTactic": CombatTactic,
        }
        isolated_scope.update(self._global_skills)

        exec(compiled_code, isolated_scope, isolated_scope)
        fn = isolated_scope.get(entry_fn_name)
        if not callable(fn):
            isolated_scope.clear()
            raise SandboxSecurityError(f"Entry function '{entry_fn_name}' was not defined.")

        self.cache.put(code_hash, skill_name_hint or entry_fn_name, fn, isolated_scope)
        return fn, False, code_hash

    async def execute_async(
        self,
        source_code: str,
        bot: WillowBotActions,
        *,
        ct: CancellationToken | None = None,
        timeout_ms: int | None = None,
        skill_name: str = "",
    ) -> SandboxExecutionOutcome:
        token = ct or NONE_TOKEN
        bot.bind_cancellation_token(token)
        started_ns = time.perf_counter_ns()
        code_hash = CompiledSkillCache.compute_hash(source_code)

        try:
            fn, cache_hit, code_hash = self.compile_skill(source_code, skill_name_hint=skill_name)
        except SandboxSecurityError as exc:
            bot.telemetry_recorder.compilation_failed = True
            self.cache.unload(code_hash)
            dur_ms = int((time.perf_counter_ns() - started_ns) // 1_000_000)
            LOG.warning("[CRITIC] Sandbox AST/Syntax rejection (-15.0 penalty): %s", exc)
            return SandboxExecutionOutcome(
                action_result=ActionResult.fail("MISSING_ITEM", str(exc)),
                compilation_failed=True,
                security_violation=str(exc),
                runtime_exception="",
                cache_hit=False,
                duration_ms=dur_ms,
            )

        eff_timeout_sec = (timeout_ms if timeout_ms is not None else self.default_timeout_ms) / 1000.0
        try:
            token.throw_if_cancellation_requested()
            sig = inspect.signature(fn)
            arity = len(sig.parameters)
            if arity == 0:
                call_res = fn()
            elif arity == 1:
                call_res = fn(bot)
            else:
                call_res = fn(bot, token)

            if inspect.isawaitable(call_res):
                raw_out = await asyncio.wait_for(call_res, timeout=eff_timeout_sec)
            else:
                raw_out = call_res

            dur_ms = int((time.perf_counter_ns() - started_ns) // 1_000_000)
            if isinstance(raw_out, ActionResult):
                final_res = raw_out
            elif isinstance(raw_out, bool):
                final_res = ActionResult.ok("Completed") if raw_out else ActionResult.fail("MISSING_ITEM", "Returned False")
            else:
                final_res = ActionResult.ok("Skill coroutine completed")

            return SandboxExecutionOutcome(
                action_result=final_res,
                compilation_failed=False,
                security_violation="",
                runtime_exception="",
                cache_hit=cache_hit,
                duration_ms=dur_ms,
            )
        except OperationCanceledError:
            raise
        except asyncio.TimeoutError:
            bot.telemetry_recorder.pathfinding_timeouts += 1
            self.cache.unload(code_hash)
            dur_ms = int((time.perf_counter_ns() - started_ns) // 1_000_000)
            return SandboxExecutionOutcome(
                action_result=ActionResult.fail("TIMEOUT", f"Skill exceeded {eff_timeout_sec:.1f}s timeout"),
                compilation_failed=False,
                security_violation="",
                runtime_exception="",
                cache_hit=cache_hit,
                duration_ms=dur_ms,
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            bot.telemetry_recorder.record_exception(err_msg)
            # Immediately unload broken script scope from RAM so failed scripts never leak
            self.cache.unload(code_hash)
            dur_ms = int((time.perf_counter_ns() - started_ns) // 1_000_000)
            return SandboxExecutionOutcome(
                action_result=ActionResult.fail("MISSING_ITEM", err_msg),
                compilation_failed=False,
                security_violation="",
                runtime_exception=err_msg,
                cache_hit=cache_hit,
                duration_ms=dur_ms,
            )

    def execute_sync(
        self,
        source_code: str,
        bot: WillowBotActions,
        *,
        ct: CancellationToken | None = None,
        timeout_ms: int | None = None,
        skill_name: str = "",
    ) -> SandboxExecutionOutcome:
        return asyncio.run(
            self.execute_async(
                source_code,
                bot,
                ct=ct,
                timeout_ms=timeout_ms,
                skill_name=skill_name,
            )
        )
