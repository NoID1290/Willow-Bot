"""Module 0: 12GB VRAM Local LLM Client, Priority Single-Flight Gate & Cancellation Primitives."""

from __future__ import annotations

import asyncio
from enum import IntEnum
import heapq
import json
import logging
import os
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

LOG = logging.getLogger("localai.vram_opt")

DEFAULT_RESIDENT_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
MAX_AI_VRAM_BUDGET_MIB = 8192
RECOMMENDED_NUM_CTX = 4096
RECOMMENDED_NUM_PREDICT = 768
RECOMMENDED_NUM_BATCH = 256
MAX_DYNAMIC_SUFFIX_TOKENS = 1200


class OperationCanceledError(RuntimeError):
    """Raised when a CancellationToken is signaled by the 20Hz Reflex Loop or shutdown."""


class CancellationToken:
    """Zero-overhead thread-safe cancellation token compatible with sync and async execution."""

    __slots__ = ("_event", "_callbacks", "_lock")

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event if event is not None else threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def is_cancellation_requested(self) -> bool:
        return self._event.is_set()

    def throw_if_cancellation_requested(self) -> None:
        if self._event.is_set():
            raise OperationCanceledError("Operation preempted by cancellation token.")

    # Alias matching C# signature referenced in synthesized skill contracts
    def ThrowIfCancellationRequested(self) -> None:
        self.throw_if_cancellation_requested()

    def register(self, callback: Callable[[], None]) -> None:
        invoke_now = False
        with self._lock:
            if self._event.is_set():
                invoke_now = True
            else:
                self._callbacks.append(callback)
        if invoke_now:
            callback()

    def _fire_callbacks(self) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                LOG.debug("[VRAM-OPT] Cancellation callback error: %s", exc)


NONE_TOKEN = CancellationToken()


class CancellationTokenSource:
    """Master preemption source held by the Fast Reactive Reflex Loop."""

    __slots__ = ("_event", "_token", "_children", "_lock")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._token = CancellationToken(self._event)
        self._children: list[CancellationTokenSource] = []
        self._lock = threading.Lock()

    @property
    def token(self) -> CancellationToken:
        return self._token

    @property
    def is_cancellation_requested(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        if not self._event.is_set():
            self._event.set()
            self._token._fire_callbacks()
        with self._lock:
            children = list(self._children)
            self._children.clear()
        for child in children:
            child.cancel()

    def create_linked_source(self) -> CancellationTokenSource:
        child = CancellationTokenSource()
        with self._lock:
            if self._event.is_set():
                child.cancel()
            else:
                self._children.append(child)
        return child


class InferencePriority(IntEnum):
    """Priority ordering for single-flight LLM semaphore: SkillMutation > NewSkillSynthesis > TaskDecomposition."""

    SKILL_MUTATION = 0
    NEW_SKILL_SYNTHESIS = 1
    TASK_DECOMPOSITION = 2
    FALLBACK_CRITIC = 3


def compute_safe_num_threads(processor_count: int | None = None) -> int:
    """Enforce num_thread = max(2, processor_count // 2 - 1) so LLM prompt eval never starves Minecraft."""
    logical = processor_count if processor_count is not None else (os.cpu_count() or 4)
    return max(2, (int(logical) // 2) - 1)


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~3.6 UTF-8 bytes per token for code + compact telemetry)."""
    if not text:
        return 0
    return max(1, int(len(text.encode("utf-8")) / 3.6 + 0.999))


class LocalInferenceOptimizer:
    """Enforces <= 8.0GB AI VRAM footprint on 12GB GPUs and single-resident model policy."""

    __slots__ = (
        "resident_model",
        "ollama_url",
        "keep_alive",
        "num_ctx",
        "num_predict",
        "temperature",
        "top_p",
        "num_batch",
        "num_thread",
        "_lock",
        "_loaded_model",
    )

    def __init__(
        self,
        resident_model: str = DEFAULT_RESIDENT_MODEL,
        ollama_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        processor_count: int | None = None,
    ) -> None:
        self.resident_model = resident_model or DEFAULT_RESIDENT_MODEL
        self.ollama_url = ollama_url.rstrip("/")
        self.keep_alive = keep_alive
        self.num_ctx = RECOMMENDED_NUM_CTX
        self.num_predict = RECOMMENDED_NUM_PREDICT
        self.temperature = 0.15
        self.top_p = 0.9
        self.num_batch = RECOMMENDED_NUM_BATCH
        self.num_thread = compute_safe_num_threads(processor_count)
        self._lock = threading.Lock()
        self._loaded_model: str | None = None

    def vram_budget_report(self) -> dict[str, Any]:
        weights_gb = 4.6
        kv_cache_gb = 0.6
        compute_overhead_gb = 0.8
        total_ai_gb = round(weights_gb + kv_cache_gb + compute_overhead_gb, 2)
        return {
            "resident_model": self.resident_model,
            "weights_gb": weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "compute_overhead_gb": compute_overhead_gb,
            "total_ai_vram_gb": total_ai_gb,
            "max_ai_budget_gb": 8.0,
            "minecraft_os_headroom_gb": round(12.0 - total_ai_gb, 2),
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "num_batch": self.num_batch,
            "num_thread": self.num_thread,
        }

    def build_options(
        self,
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        capped_predict = min(
            self.num_predict,
            max(32, int(num_predict if num_predict is not None else self.num_predict)),
        )
        return {
            "num_ctx": self.num_ctx,
            "num_predict": capped_predict,
            "temperature": round(float(self.temperature if temperature is None else temperature), 3),
            "top_p": self.top_p,
            "num_batch": self.num_batch,
            "num_thread": self.num_thread,
            "use_mmap": True,
            "use_mlock": False,
            "flash_attn": True,
        }

    def build_request_payload(
        self,
        static_prefix: str,
        dynamic_suffix: str,
        *,
        stream: bool = True,
        num_predict: int | None = None,
        temperature: float | None = None,
        format_schema: dict[str, Any] | str | None = "json",
    ) -> dict[str, Any]:
        """Constructs KV-cache aligned Ollama /api/chat payload with 100% static system prefix."""
        trimmed_suffix = self.clamp_dynamic_suffix(dynamic_suffix)
        messages = [
            {"role": "system", "content": static_prefix},
            {"role": "user", "content": trimmed_suffix},
        ]
        payload: dict[str, Any] = {
            "model": self.resident_model,
            "keep_alive": self.keep_alive,
            "stream": bool(stream),
            "messages": messages,
            "options": self.build_options(num_predict=num_predict, temperature=temperature),
        }
        if format_schema is not None:
            payload["format"] = format_schema
        return payload

    @staticmethod
    def clamp_dynamic_suffix(dynamic_suffix: str, max_tokens: int = MAX_DYNAMIC_SUFFIX_TOKENS) -> str:
        if estimate_tokens(dynamic_suffix) <= max_tokens:
            return dynamic_suffix
        max_bytes = int(max_tokens * 3.5)
        encoded = dynamic_suffix.encode("utf-8")
        if len(encoded) <= max_bytes:
            return dynamic_suffix
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def ensure_single_resident_model(
        self,
        target_model: str | None = None,
        transport: Callable[[str, bytes, float], bytes] | None = None,
    ) -> str:
        """Unloads any non-resident model before switching so multiple LLMs never coexist in VRAM."""
        desired = target_model or self.resident_model
        with self._lock:
            previous = self._loaded_model
            if previous == desired:
                return desired
            self.resident_model = desired
            self._loaded_model = desired

        if previous and previous != desired:
            self._unload_model(previous, transport=transport)
        return desired

    def _unload_model(
        self,
        model_name: str,
        transport: Callable[[str, bytes, float], bytes] | None = None,
    ) -> None:
        body = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        url = f"{self.ollama_url}/api/generate"
        try:
            if transport is not None:
                transport(url, body, 3.0)
            else:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    resp.read(1024)
            LOG.info("[VRAM-OPT] Unloaded previous model '%s' (keep_alive=0) to preserve single-resident VRAM policy.", model_name)
        except Exception as exc:
            LOG.debug("[VRAM-OPT] Model unload best-effort warning for %s: %s", model_name, exc)


class _PriorityGate:
    """Strict single-flight (capacity=1) priority semaphore across threads and coroutines."""

    __slots__ = ("_cond", "_busy", "_counter", "_waiters")

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._counter = 0
        self._waiters: list[tuple[int, int, threading.Event]] = []

    def acquire(self, priority: InferencePriority, ct: CancellationToken | None = None, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + max(0.1, timeout)
        with self._cond:
            if ct is not None:
                ct.throw_if_cancellation_requested()
            if not self._busy and not self._waiters:
                self._busy = True
                return
            self._counter += 1
            my_evt = threading.Event()
            entry = (int(priority), self._counter, my_evt)
            heapq.heappush(self._waiters, entry)

        try:
            while True:
                if ct is not None:
                    ct.throw_if_cancellation_requested()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for single-flight LLM semaphore.")
                if my_evt.wait(timeout=min(0.05, remaining)):
                    return
        except BaseException:
            with self._cond:
                if my_evt.is_set():
                    self._release_locked()
                else:
                    try:
                        self._waiters.remove(entry)
                        heapq.heapify(self._waiters)
                    except ValueError:
                        pass
            raise

    def release(self) -> None:
        with self._cond:
            self._release_locked()

    def _release_locked(self) -> None:
        while self._waiters:
            _, _, next_evt = heapq.heappop(self._waiters)
            next_evt.set()
            return
        self._busy = False

    @property
    def in_flight(self) -> bool:
        with self._cond:
            return self._busy

    @property
    def queued_count(self) -> int:
        with self._cond:
            return len(self._waiters)


class LocalLlmClient:
    """Hardware-bounded local LLM client enforcing single-flight priority queuing and combat pause."""

    def __init__(
        self,
        optimizer: LocalInferenceOptimizer | None = None,
        *,
        transport: Callable[[dict[str, Any], CancellationToken], str] | None = None,
    ) -> None:
        self.optimizer = optimizer or LocalInferenceOptimizer()
        self._gate = _PriorityGate()
        self._combat_clear = threading.Event()
        self._combat_clear.set()
        self._transport = transport
        self._stats_lock = threading.Lock()
        self._total_calls = 0
        self._last_ttft_ms = 0.0
        self._last_static_prefix_hash: int | None = None
        self._prefix_cache_hits = 0

    @property
    def is_paused_for_reflex(self) -> bool:
        return not self._combat_clear.is_set()

    @property
    def in_flight(self) -> bool:
        return self._gate.in_flight

    @property
    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "total_calls": self._total_calls,
                "last_ttft_ms": round(self._last_ttft_ms, 2),
                "prefix_cache_hits": self._prefix_cache_hits,
                "queued_requests": self._gate.queued_count,
                "paused_for_reflex": self.is_paused_for_reflex,
                "vram_budget": self.optimizer.vram_budget_report(),
            }

    def pause_for_reflex(self) -> None:
        """Called by 20Hz Reflex Loop while actively fighting or fleeing to freeze new LLM dispatches."""
        if self._combat_clear.is_set():
            self._combat_clear.clear()
            LOG.info("[VRAM-OPT] Paused LLM inference queue during active survival reflex.")

    def resume_after_reflex(self) -> None:
        """Restores LLM inference dispatches once immediate survival hazard clears."""
        if not self._combat_clear.is_set():
            self._combat_clear.set()
            LOG.info("[VRAM-OPT] Resumed LLM inference queue after survival reflex cleared.")

    def _wait_for_reflex_clear(self, ct: CancellationToken, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self._combat_clear.is_set():
            ct.throw_if_cancellation_requested()
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError("LLM inference paused by active combat reflex beyond timeout.")
            self._combat_clear.wait(timeout=min(0.05, rem))

    def generate_sync(
        self,
        static_prefix: str,
        dynamic_suffix: str,
        *,
        priority: InferencePriority = InferencePriority.NEW_SKILL_SYNTHESIS,
        num_predict: int | None = None,
        temperature: float | None = None,
        format_schema: dict[str, Any] | str | None = "json",
        ct: CancellationToken | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        token = ct or NONE_TOKEN
        token.throw_if_cancellation_requested()
        self._wait_for_reflex_clear(token, timeout=timeout)
        self._gate.acquire(priority=priority, ct=token, timeout=timeout)
        try:
            # Re-verify reflex state right after acquiring single-flight lock
            self._wait_for_reflex_clear(token, timeout=timeout)
            self.optimizer.ensure_single_resident_model()
            payload = self.optimizer.build_request_payload(
                static_prefix=static_prefix,
                dynamic_suffix=dynamic_suffix,
                stream=True,
                num_predict=num_predict,
                temperature=temperature,
                format_schema=format_schema,
            )
            prefix_hash = hash((self.optimizer.resident_model, static_prefix))
            started = time.perf_counter()
            raw_text, ttft_ms = self._execute_streaming_request(payload, token, timeout)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            with self._stats_lock:
                self._total_calls += 1
                self._last_ttft_ms = ttft_ms
                if self._last_static_prefix_hash == prefix_hash:
                    self._prefix_cache_hits += 1
                    cache_hit = True
                else:
                    self._last_static_prefix_hash = prefix_hash
                    cache_hit = False

            LOG.info(
                "[VRAM-OPT] model=%s prio=%s ctx=%d predict=%d threads=%d kv_prefix_hit=%s ttft=%.1fms total=%.1fms",
                self.optimizer.resident_model,
                priority.name,
                payload["options"]["num_ctx"],
                payload["options"]["num_predict"],
                payload["options"]["num_thread"],
                cache_hit,
                ttft_ms,
                elapsed_ms,
            )
            parsed = self._parse_json_object(raw_text)
            return parsed
        finally:
            self._gate.release()

    async def generate_async(
        self,
        static_prefix: str,
        dynamic_suffix: str,
        *,
        priority: InferencePriority = InferencePriority.NEW_SKILL_SYNTHESIS,
        num_predict: int | None = None,
        temperature: float | None = None,
        format_schema: dict[str, Any] | str | None = "json",
        ct: CancellationToken | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.generate_sync,
            static_prefix,
            dynamic_suffix,
            priority=priority,
            num_predict=num_predict,
            temperature=temperature,
            format_schema=format_schema,
            ct=ct,
            timeout=timeout,
        )

    def _execute_streaming_request(
        self,
        payload: dict[str, Any],
        ct: CancellationToken,
        timeout: float,
    ) -> tuple[str, float]:
        if self._transport is not None:
            started = time.perf_counter()
            out = self._transport(payload, ct)
            return out, (time.perf_counter() - started) * 1000.0

        url = f"{self.optimizer.ollama_url}/api/chat"
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        started = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                while True:
                    ct.throw_if_cancellation_requested()
                    line = resp.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    frame = json.loads(line_str)
                    if "error" in frame:
                        raise OSError(f"Ollama streaming error: {frame['error']}")
                    msg = frame.get("message")
                    piece = ""
                    if isinstance(msg, dict):
                        piece = str(msg.get("content") or "")
                    elif "response" in frame:
                        piece = str(frame.get("response") or "")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000.0
                        chunks.append(piece)
                    if frame.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OSError(f"Local LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OSError(f"Local LLM connection error: {exc.reason}") from exc

        total_text = "".join(chunks)
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - started) * 1000.0
        return total_text, ttft_ms

    @staticmethod
    def _parse_json_object(raw_text: str) -> dict[str, Any]:
        cleaned = (raw_text or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Extract first balanced JSON object if reasoning prefix leaked outside JSON
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError(f"LLM output did not contain a JSON object: {cleaned[:160]}")
            parsed = json.loads(cleaned[start_idx : end_idx + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response root must be an object.")
        return parsed
