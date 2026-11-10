"""Module 1: Bounded Ring-Buffer Verbal Reinforcement Memory (ReflexionMemoryLog)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

from .reward_calculator import RewardBreakdown


@dataclass(frozen=True, slots=True)
class EpisodeReflectionRecord:
    task_id: str
    failure_index: int
    r_total: float
    p_vital: float
    p_exec: float
    cause: str
    fix: str
    timestamp_ms: int

    def format_compact(self) -> str:
        return (
            f"[FAIL #{self.failure_index} | R={self.r_total:+.1f} "
            f"(Vital:-{self.p_vital:.0f}, Exec:-{self.p_exec:.0f})] "
            f"Cause: {self.cause} Fix: {self.fix}"
        )

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "ts": self.timestamp_ms,
                "task_id": self.task_id,
                "fail_num": self.failure_index,
                "r_total": round(self.r_total, 2),
                "p_vital": round(self.p_vital, 2),
                "p_exec": round(self.p_exec, 2),
                "cause": self.cause,
                "fix": self.fix,
            },
            separators=(",", ":"),
        )


class ReflexionMemoryLog:
    """Caps RAM reflection history to 5 entries per task, injects max 3 into prompts, flushes to .jsonl."""

    MAX_RAM_ENTRIES_PER_TASK = 5
    MAX_PROMPT_INJECTION_LINES = 3

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self._lock = threading.Lock()
        self._buffers: dict[str, deque[EpisodeReflectionRecord]] = {}
        self._failure_counts: dict[str, int] = {}
        self._flush_queue: queue.Queue[str | None] | None = None
        self._writer_thread: threading.Thread | None = None

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._flush_queue = queue.Queue(maxsize=256)
            self._writer_thread = threading.Thread(
                target=self._background_writer,
                name="ReflexionJsonlFlusher",
                daemon=True,
            )
            self._writer_thread.start()

    def record_if_needed(
        self,
        task_id: str,
        reward: RewardBreakdown,
        critic_success: bool,
        cause: str,
        fix: str,
    ) -> EpisodeReflectionRecord | None:
        """Stores a compact reflection whenever R_total < 0 or critic.success == False."""
        if reward.r_total >= 0.0 and critic_success:
            return None

        evicted: EpisodeReflectionRecord | None = None
        with self._lock:
            fail_num = self._failure_counts.get(task_id, 0) + 1
            self._failure_counts[task_id] = fail_num
            buf = self._buffers.get(task_id)
            if buf is None:
                buf = deque(maxlen=self.MAX_RAM_ENTRIES_PER_TASK)
                self._buffers[task_id] = buf
            elif len(buf) == self.MAX_RAM_ENTRIES_PER_TASK:
                evicted = buf[0]

            rec = EpisodeReflectionRecord(
                task_id=task_id,
                failure_index=fail_num,
                r_total=reward.r_total,
                p_vital=reward.p_vital,
                p_exec=reward.p_exec,
                cause=(cause or "Postcondition not met")[:120],
                fix=(fix or "Verify prerequisites and tool tiers before execution")[:120],
                timestamp_ms=int(time.time() * 1000),
            )
            buf.append(rec)

        # Flush both new analytics and evicted ring-buffer entries asynchronously to disk
        if self._flush_queue is not None:
            line = rec.to_json_line()
            try:
                self._flush_queue.put_nowait(line)
            except queue.Full:
                pass
            _ = evicted
        return rec

    def get_prompt_lines(self, task_id: str, max_lines: int = MAX_PROMPT_INJECTION_LINES) -> list[str]:
        cap = max(1, min(self.MAX_PROMPT_INJECTION_LINES, int(max_lines)))
        with self._lock:
            buf = self._buffers.get(task_id)
            if not buf:
                return []
            recent = list(buf)[-cap:]
        return [r.format_compact() for r in recent]

    def format_for_prompt(self, task_id: str, max_lines: int = MAX_PROMPT_INJECTION_LINES) -> str:
        lines = self.get_prompt_lines(task_id, max_lines=max_lines)
        return " ; ".join(lines) if lines else "none"

    def get_task_records(self, task_id: str) -> list[EpisodeReflectionRecord]:
        with self._lock:
            buf = self._buffers.get(task_id)
            return list(buf) if buf else []

    def close(self) -> None:
        if self._flush_queue is not None and self._writer_thread is not None:
            try:
                self._flush_queue.put_nowait(None)
            except queue.Full:
                pass
            self._writer_thread.join(timeout=2.0)

    def _background_writer(self) -> None:
        assert self._flush_queue is not None and self.jsonl_path is not None
        while True:
            item = self._flush_queue.get()
            if item is None:
                break
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(item + "\n")
            except OSError:
                pass
