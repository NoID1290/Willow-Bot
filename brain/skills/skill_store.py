"""Module 4: Zero-VRAM Hybrid RAG Skill Library (BM25 + 384-dim CPU MurmurHash3/SIMD + Q-Value)."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import math
import mmap
from pathlib import Path
import re
import struct
import threading
from typing import Any, Callable

from curriculum.curriculum_dag import CurriculumNode
from execution.sandbox import ScriptSandbox
from rl_rewards.q_table import ContextStateHash, SkillQTable
from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot

LOG = logging.getLogger("localai.skill_store")

EMBEDDING_DIM = 384
BYTES_PER_VECTOR = EMBEDDING_DIM * 4  # 1,536 bytes per skill -> 1,000 skills = 1.5 MB RAM
TOKEN_RE = re.compile(r"[a-z0-9_]+")

try:
    import numpy as _np  # Optional BLAS acceleration if installed on system
except ImportError:
    _np = None


def _murmur3_32(data: bytes, seed: int = 0) -> int:
    """Pure-CPU 32-bit MurmurHash3 for deterministic 0-VRAM feature hashing."""
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    length = len(data)
    h1 = seed & 0xFFFFFFFF
    rounded_end = length & ~0x3

    for i in range(0, rounded_end, 4):
        k1 = (
            data[i]
            | (data[i + 1] << 8)
            | (data[i + 2] << 16)
            | (data[i + 3] << 24)
        )
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF

        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    k1 = 0
    rem = length & 3
    if rem == 3:
        k1 ^= data[rounded_end + 2] << 16
    if rem >= 2:
        k1 ^= data[rounded_end + 1] << 8
    if rem >= 1:
        k1 ^= data[rounded_end]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1


class CpuFeatureHasher:
    """Zero-VRAM 384-dim signed MurmurHash3 N-gram embedder with C/SIMD dot product."""

    @staticmethod
    def embed(text: str) -> array:
        vec = array("f", [0.0] * EMBEDDING_DIM)
        tokens = TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vec

        features: list[tuple[bytes, float]] = []
        for idx, tok in enumerate(tokens):
            tb = tok.encode("utf-8")
            features.append((tb, 1.5))
            # Bigram lexical context
            if idx + 1 < len(tokens):
                features.append((f"{tok}_{tokens[idx + 1]}".encode("utf-8"), 1.2))
            # Character 3-grams for morphological resilience ('iron_pickaxe' <-> 'pickaxe')
            if len(tok) >= 4:
                for c_idx in range(len(tok) - 2):
                    features.append((tok[c_idx : c_idx + 3].encode("utf-8"), 0.5))

        for feat_bytes, weight in features:
            h_idx = _murmur3_32(feat_bytes, seed=0x9747B28C)
            h_sgn = _murmur3_32(feat_bytes, seed=0x165667B1)
            bucket = h_idx % EMBEDDING_DIM
            sign = 1.0 if (h_sgn & 1) == 0 else -1.0
            vec[bucket] += sign * weight

        norm_sq = math.sumprod(vec, vec)
        if norm_sq > 1e-12:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            for i in range(EMBEDDING_DIM):
                vec[i] *= inv_norm
        return vec

    @staticmethod
    def simd_dot(vec_a: array | memoryview, vec_b: array | memoryview) -> float:
        """Executes vectorized C-level dot product (numpy BLAS if installed, else CPython math.sumprod)."""
        if _np is not None:
            arr_a = _np.frombuffer(vec_a, dtype=_np.float32)
            arr_b = _np.frombuffer(vec_b, dtype=_np.float32)
            return float(_np.dot(arr_a, arr_b))
        return float(math.sumprod(vec_a, vec_b))


def _bm25_lexical_similarity(query_tokens: list[str], doc_tokens: list[str], avg_dl: float = 16.0) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    k1 = 1.2
    b = 0.75
    dl = len(doc_tokens)
    counts: dict[str, int] = {}
    for t in doc_tokens:
        counts[t] = counts.get(t, 0) + 1
    raw = 0.0
    unique_q = dict.fromkeys(query_tokens)
    for q in unique_q:
        tf = counts.get(q, 0)
        if tf > 0:
            raw += (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / max(1.0, avg_dl))))
    max_possible = float(len(unique_q)) * (k1 + 1.0)
    return max(0.0, min(1.0, raw / max(1.0, max_possible)))


def _sigmoid(x: float) -> float:
    clamped = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-clamped))


class SkillStatus(str, Enum):
    Verified = "Verified"
    Probation = "Probation"
    NeedsRefinement = "NeedsRefinement"
    Deprecated = "Deprecated"


@dataclass(slots=True)
class SkillRecord:
    skill_id: str
    skill_name: str
    description: str
    task_id: str
    preconditions: dict[str, int] = field(default_factory=dict)
    postconditions: dict[str, int] = field(default_factory=dict)
    q_value: float = 0.0
    consecutive_failures: int = 0
    success_count: int = 0
    status: SkillStatus = SkillStatus.Probation
    embedding_index: int = 0
    version: int = 1
    signature: str = ""

    def precondition_match_ratio(self, state: BotTelemetrySnapshot) -> float:
        if not self.preconditions:
            return 1.0
        scores: list[float] = []
        for pat, req in self.preconditions.items():
            if pat == "crafting_table_nearby":
                ok = "crafting_table" in state.nearby_workstations or state.has_item("crafting_table")
                scores.append(1.0 if ok else 0.0)
            elif pat == "furnace_nearby":
                ok = "furnace" in state.nearby_workstations or state.has_item("furnace")
                scores.append(1.0 if ok else 0.0)
            else:
                have = state.count_item(pat)
                scores.append(min(1.0, have / float(max(1, req))))
        return (sum(scores) / len(scores)) if scores else 1.0

    def preconditions_satisfied(self, state: BotTelemetrySnapshot) -> bool:
        return self.precondition_match_ratio(state) >= 0.999

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "description": self.description,
            "task_id": self.task_id,
            "preconditions": dict(self.preconditions),
            "postconditions": dict(self.postconditions),
            "q_value": round(self.q_value, 4),
            "consecutive_failures": self.consecutive_failures,
            "success_count": self.success_count,
            "status": self.status.value,
            "embedding_index": self.embedding_index,
            "version": self.version,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SkillRecord:
        return cls(
            skill_id=str(raw["skill_id"]),
            skill_name=str(raw["skill_name"]),
            description=str(raw.get("description", "")),
            task_id=str(raw.get("task_id", "")),
            preconditions={str(k): int(v) for k, v in (raw.get("preconditions") or {}).items()},
            postconditions={str(k): int(v) for k, v in (raw.get("postconditions") or {}).items()},
            q_value=float(raw.get("q_value", 0.0)),
            consecutive_failures=int(raw.get("consecutive_failures", 0)),
            success_count=int(raw.get("success_count", 0)),
            status=SkillStatus(raw.get("status", SkillStatus.Probation.value)),
            embedding_index=int(raw.get("embedding_index", 0)),
            version=int(raw.get("version", 1)),
            signature=str(raw.get("signature", "")),
        )


@dataclass(frozen=True, slots=True)
class RetrievedSkillMatch:
    record: SkillRecord
    score: float
    cpu_similarity: float
    q_factor: float
    precondition_ratio: float


class SkillStore:
    """Persistent Zero-VRAM RAG repository backed by skill_registry.json, library/*.code, and mmap embeddings.bin."""

    MUTATION_Q_THRESHOLD = -15.0
    MUTATION_FAILURE_STREAK = 2

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        q_table: SkillQTable | None = None,
        sandbox: ScriptSandbox | None = None,
        cpu_embedder: Callable[[str], array] | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.library_dir = self.storage_dir / "library"
        self.registry_path = self.storage_dir / "skill_registry.json"
        self.embeddings_path = self.storage_dir / "embeddings.bin"
        self.q_table = q_table
        self.sandbox = sandbox
        self._embed_fn = cpu_embedder or CpuFeatureHasher.embed

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records: dict[str, SkillRecord] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if not self.registry_path.is_file():
            return
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        for entry in raw.get("skills", []):
            rec = SkillRecord.from_dict(entry)
            self._records[rec.skill_id] = rec
            self._doc_tokens[rec.skill_id] = TOKEN_RE.findall(
                f"{rec.skill_name} {rec.description} {rec.task_id} {' '.join(rec.postconditions.keys())}".lower()
            )
            if rec.status == SkillStatus.Verified and self.sandbox is not None:
                self._bind_into_sandbox(rec)

    def _save_registry_locked(self) -> None:
        payload = {
            "embedding_dim": EMBEDDING_DIM,
            "skills": [r.to_dict() for r in self._records.values()],
        }
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _append_or_write_embedding_locked(self, index: int, vec: array) -> None:
        raw_bytes = struct.pack(f"<{EMBEDDING_DIM}f", *vec)
        if not self.embeddings_path.exists():
            self.embeddings_path.write_bytes(b"")
        current_size = self.embeddings_path.stat().st_size
        target_offset = index * BYTES_PER_VECTOR
        with self.embeddings_path.open("r+b") as f:
            if target_offset > current_size:
                f.seek(0, 2)
                f.write(b"\x00" * (target_offset - current_size))
            else:
                f.seek(target_offset)
            f.write(raw_bytes)

    def _read_embedding_vector(self, index: int) -> array:
        with self._lock:
            if not self.embeddings_path.is_file():
                return array("f", [0.0] * EMBEDDING_DIM)
            size = self.embeddings_path.stat().st_size
            offset = index * BYTES_PER_VECTOR
            if offset + BYTES_PER_VECTOR > size or size == 0:
                return array("f", [0.0] * EMBEDDING_DIM)
            with self.embeddings_path.open("rb") as f:
                f.seek(offset)
                chunk = f.read(BYTES_PER_VECTOR)
                if len(chunk) < BYTES_PER_VECTOR:
                    return array("f", [0.0] * EMBEDDING_DIM)
                out = array("f")
                out.frombytes(chunk)
                return out

    def load_code(self, skill_id: str) -> str:
        code_file = self.library_dir / f"{skill_id}.code"
        if not code_file.is_file():
            raise FileNotFoundError(f"Skill code file missing: {code_file}")
        return code_file.read_text(encoding="utf-8")

    def register_skill(
        self,
        *,
        skill_id: str,
        skill_name: str,
        description: str,
        code: str,
        task_id: str,
        preconditions: dict[str, int] | None = None,
        postconditions: dict[str, int] | None = None,
        verified: bool = True,
        q_value: float = 10.0,
        version: int = 1,
    ) -> SkillRecord:
        sig = f"await {skill_name}(bot, ct) # {description[:80]}"
        embed_text = f"{skill_name} {description} {task_id} {' '.join((postconditions or {}).keys())}"
        vec = self._embed_fn(embed_text)

        code_path = self.library_dir / f"{skill_id}.code"
        tmp_code = code_path.with_suffix(".tmp")
        tmp_code.write_text(code, encoding="utf-8")
        tmp_code.replace(code_path)

        with self._lock:
            existing = self._records.get(skill_id)
            emb_idx = existing.embedding_index if existing is not None else len(self._records)
            rec = SkillRecord(
                skill_id=skill_id,
                skill_name=skill_name,
                description=description,
                task_id=task_id,
                preconditions=dict(preconditions or {}),
                postconditions=dict(postconditions or {}),
                q_value=float(q_value),
                consecutive_failures=0,
                success_count=1 if verified else 0,
                status=SkillStatus.Verified if verified else SkillStatus.Probation,
                embedding_index=emb_idx,
                version=version,
                signature=sig,
            )
            self._records[skill_id] = rec
            self._doc_tokens[skill_id] = TOKEN_RE.findall(embed_text.lower())
            self._append_or_write_embedding_locked(emb_idx, vec)
            self._save_registry_locked()

        if verified and self.sandbox is not None:
            self._bind_into_sandbox(rec)
        return rec

    def _bind_into_sandbox(self, rec: SkillRecord) -> None:
        if self.sandbox is None:
            return
        try:
            code = self.load_code(rec.skill_id)
            fn, _, _ = self.sandbox.compile_skill(code, skill_name_hint=rec.skill_name)
            self.sandbox.register_composed_skill(rec.skill_name, fn)
        except Exception as exc:
            LOG.debug("Deferred sandbox binding for skill %s: %s", rec.skill_id, exc)

    def get_verified_skills_for_task(
        self,
        task_id: str,
        state: BotTelemetrySnapshot,
    ) -> list[SkillRecord]:
        with self._lock:
            candidates = [
                r
                for r in self._records.values()
                if r.status == SkillStatus.Verified
                and (r.task_id == task_id or not r.task_id)
                and r.preconditions_satisfied(state)
            ]
        return candidates

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        with self._lock:
            return self._records.get(skill_id)

    def retrieve_relevant_skills(
        self,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        top_k: int = 3,
    ) -> list[RetrievedSkillMatch]:
        """Hybrid CPU retrieval: 0.50*Sim_CPU(q,k) + 0.30*sigmoid(Q(c,k)/50.0) + 0.20*PrecondMatch(S_t,k)."""
        capped_k = max(1, min(3, int(top_k)))
        query_text = f"{task.id} {task.description} {' '.join(task.target_items.keys())}"
        q_vec = self._embed_fn(query_text)
        q_tokens = TOKEN_RE.findall(query_text.lower())
        ctx_hash = ContextStateHash.encode(state)

        with self._lock:
            active_records = [
                r for r in self._records.values() if r.status in (SkillStatus.Verified, SkillStatus.Probation)
            ]
            avg_dl = (
                sum(len(self._doc_tokens.get(r.skill_id, ())) for r in active_records) / max(1, len(active_records))
                if active_records
                else 16.0
            )

        matches: list[RetrievedSkillMatch] = []
        for rec in active_records:
            k_vec = self._read_embedding_vector(rec.embedding_index)
            vec_sim = max(0.0, min(1.0, CpuFeatureHasher.simd_dot(q_vec, k_vec)))
            bm25_sim = _bm25_lexical_similarity(q_tokens, self._doc_tokens.get(rec.skill_id, []), avg_dl=avg_dl)
            sim_cpu = 0.65 * vec_sim + 0.35 * bm25_sim

            q_ck = (
                self.q_table.get_context_skill_q(ctx_hash, rec.skill_id)
                if self.q_table is not None
                else rec.q_value
            )
            q_sig = _sigmoid(q_ck / 50.0)
            precond_ratio = rec.precondition_match_ratio(state)

            total_score = (0.50 * sim_cpu) + (0.30 * q_sig) + (0.20 * precond_ratio)
            matches.append(
                RetrievedSkillMatch(
                    record=rec,
                    score=total_score,
                    cpu_similarity=sim_cpu,
                    q_factor=q_sig,
                    precondition_ratio=precond_ratio,
                )
            )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:capped_k]

    # Alias matching C# specification
    RetrieveRelevantSkills = retrieve_relevant_skills

    def format_rag_signatures(self, task: CurriculumNode, state: BotTelemetrySnapshot, top_k: int = 3) -> str:
        top = self.retrieve_relevant_skills(task, state, top_k=top_k)
        if not top:
            return "none"
        return " | ".join(f"{m.record.signature} [Q={m.record.q_value:+.1f}]" for m in top)

    def record_execution_feedback(
        self,
        skill_id: str,
        *,
        success: bool,
        updated_q: float,
    ) -> SkillRecord | None:
        """Marks skill Status = NeedsRefinement if Q < -15.0 or consecutive_failures >= 2."""
        with self._lock:
            rec = self._records.get(skill_id)
            if rec is None:
                return None
            rec.q_value = float(updated_q)
            if success:
                rec.consecutive_failures = 0
                rec.success_count += 1
                if rec.status in (SkillStatus.Probation, SkillStatus.NeedsRefinement):
                    rec.status = SkillStatus.Verified
            else:
                rec.consecutive_failures += 1
                if (
                    rec.q_value < self.MUTATION_Q_THRESHOLD
                    or rec.consecutive_failures >= self.MUTATION_FAILURE_STREAK
                ):
                    rec.status = SkillStatus.NeedsRefinement
            self._save_registry_locked()
            return rec

    def replace_with_mutated_skill(
        self,
        old_skill_id: str,
        *,
        new_skill_name: str,
        new_description: str,
        new_code: str,
    ) -> SkillRecord:
        """Replaces NeedsRefinement skill with <base_id>_v2 (or incremented version) and deprecates predecessor."""
        with self._lock:
            old = self._records.get(old_skill_id)
            if old is not None:
                old.status = SkillStatus.Deprecated
                next_ver = old.version + 1
                base_id = re.sub(r"_v\d+$", "", old_skill_id)
                new_id = f"{base_id}_v{next_ver}"
                task_id = old.task_id
                preconds = dict(old.preconditions)
                postconds = dict(old.postconditions)
                if self.sandbox is not None:
                    self.sandbox.unregister_composed_skill(old.skill_name)
                self._save_registry_locked()
            else:
                next_ver = 2
                new_id = f"{old_skill_id}_v2"
                task_id = ""
                preconds = {}
                postconds = {}

        return self.register_skill(
            skill_id=new_id,
            skill_name=new_skill_name,
            description=new_description,
            code=new_code,
            task_id=task_id,
            preconditions=preconds,
            postconditions=postconds,
            verified=False,
            q_value=0.0,
            version=next_ver,
        )
