"""Module 1: 32-Bit Contextual State Hash & Tabular Q-Learning Table with Fast-Path LLM Bypass."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from typing import Iterable

from .telemetry_snapshot import BotTelemetrySnapshot

LOG = logging.getLogger("localai.q_table")

DEFAULT_ALPHA = 0.25
DEFAULT_GAMMA = 0.85
FAST_PATH_Q_THRESHOLD = 10.0


class ContextStateHash:
    """Packs bot survival state into a compact 32-bit unsigned integer bitmask (14 active bits).

    Bit layout:
      [0..2]   ToolTier     (3 bits): None(0) | Wood(1) | Stone(2) | Iron(3) | Diamond+(4)
      [3..4]   ArmorTier    (2 bits): None(0) | PartialIron(1) | FullIron(2) | Diamond(3)
      [5..6]   HealthBucket (2 bits): Critical <8 (0) | Moderate 8-15 (1) | Healthy 16-20 (2)
      [7..8]   TimeBucket   (2 bits): Day(0) | Dusk(1) | Night(2)
      [9..10]  Dimension    (2 bits): Overworld_Surface(0) | Overworld_Cave Y<45(1) | Nether(2) | End(3)
      [11..13] Flags        (3 bits): HasCraftingTable(bit 11) | HasFurnace(bit 12) | HasShield(bit 13)
    """

    @staticmethod
    def encode(snapshot: BotTelemetrySnapshot) -> int:
        tool_raw = max(
            snapshot.highest_tool_tier("_pickaxe"),
            snapshot.highest_tool_tier("_sword"),
        )
        tool_tier = min(4, max(0, tool_raw))

        armor_pts = snapshot.vitals.armor_points
        has_diamond_armor = any(
            eq.item_id.split(":", 1)[-1].startswith(("diamond_", "netherite_"))
            for slot, eq in snapshot.equipment.items()
            if slot.value in ("Head", "Chest", "Legs", "Feet") and eq.item_id
        )
        if has_diamond_armor or armor_pts >= 16:
            armor_tier = 3
        elif armor_pts >= 15:
            armor_tier = 2
        elif armor_pts > 0:
            armor_tier = 1
        else:
            armor_tier = 0

        hp = snapshot.vitals.health
        if hp < 8.0:
            hp_bucket = 0
        elif hp < 16.0:
            hp_bucket = 1
        else:
            hp_bucket = 2

        tod = snapshot.time_of_day % 24000
        if 12542 <= tod <= 23460:
            time_bucket = 2
        elif 11800 <= tod < 12542 or 23460 < tod <= 24000:
            time_bucket = 1
        else:
            time_bucket = 0

        dim = snapshot.dimension
        if "the_nether" in dim:
            dim_bucket = 2
        elif "the_end" in dim:
            dim_bucket = 3
        elif snapshot.position.y < 45.0:
            dim_bucket = 1
        else:
            dim_bucket = 0

        has_table = int(
            "crafting_table" in snapshot.nearby_workstations
            or snapshot.has_item("minecraft:crafting_table")
        )
        has_furnace = int(
            "furnace" in snapshot.nearby_workstations
            or snapshot.has_item("minecraft:furnace")
        )
        has_shield = int(snapshot.has_equipped_or_inv("minecraft:shield"))
        flags = (has_table & 1) | ((has_furnace & 1) << 1) | ((has_shield & 1) << 2)

        mask = (
            (tool_tier & 0x7)
            | ((armor_tier & 0x3) << 3)
            | ((hp_bucket & 0x3) << 5)
            | ((time_bucket & 0x3) << 7)
            | ((dim_bucket & 0x3) << 9)
            | ((flags & 0x7) << 11)
        )
        return mask & 0xFFFFFFFF

    @staticmethod
    def decode(mask: int) -> dict[str, int | bool]:
        flags = (mask >> 11) & 0x7
        return {
            "tool_tier": mask & 0x7,
            "armor_tier": (mask >> 3) & 0x3,
            "health_bucket": (mask >> 5) & 0x3,
            "time_bucket": (mask >> 7) & 0x3,
            "dimension_bucket": (mask >> 9) & 0x3,
            "has_crafting_table": bool(flags & 0x1),
            "has_furnace": bool(flags & 0x2),
            "has_shield": bool(flags & 0x4),
        }


class SkillQTable:
    """Tabular Q-Learning engine over (context_hash, task_id, skill_id) with JSON persistence."""

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        fast_path_threshold: float = FAST_PATH_Q_THRESHOLD,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.fast_path_threshold = float(fast_path_threshold)
        self._lock = threading.Lock()
        # Nested map: context_hash -> task_id -> skill_id -> Q-value
        self._table: dict[int, dict[str, dict[str, float]]] = {}
        # Global aggregate per skill across contexts for RAG fallback scoring
        self._skill_ema: dict[str, float] = {}
        if self.storage_path is not None and self.storage_path.is_file():
            self.load()

    def get_q(self, context_hash: int, task_id: str, skill_id: str) -> float:
        with self._lock:
            return float(self._table.get(int(context_hash), {}).get(task_id, {}).get(skill_id, 0.0))

    def get_context_skill_q(self, context_hash: int, skill_id: str) -> float:
        """Returns max Q(c, *, k) in current context if seen, else global EMA Q(k)."""
        with self._lock:
            ctx_map = self._table.get(int(context_hash), {})
            vals = [skills[skill_id] for skills in ctx_map.values() if skill_id in skills]
            if vals:
                return float(max(vals))
            return float(self._skill_ema.get(skill_id, 0.0))

    def max_q(self, context_hash: int, task_id: str) -> float:
        with self._lock:
            skills_map = self._table.get(int(context_hash), {}).get(task_id, {})
            if not skills_map:
                return 0.0
            return max(0.0, float(max(skills_map.values())))

    def update(
        self,
        context_prev: int,
        task_prev: str,
        skill_id: str,
        reward_total: float,
        context_next: int,
        task_next: str,
    ) -> float:
        """Applies Q(c,n,k) <- Q(c,n,k) + alpha * [R_total + gamma * max_k' Q(c',n',k') - Q(c,n,k)]."""
        c_prev = int(context_prev)
        c_next = int(context_next)
        with self._lock:
            next_skills = self._table.get(c_next, {}).get(task_next, {})
            max_future_q = max(0.0, max(next_skills.values())) if next_skills else 0.0

            ctx_bucket = self._table.setdefault(c_prev, {})
            task_bucket = ctx_bucket.setdefault(task_prev, {})
            current_q = float(task_bucket.get(skill_id, 0.0))

            td_target = float(reward_total) + self.gamma * max_future_q
            updated_q = current_q + self.alpha * (td_target - current_q)
            task_bucket[skill_id] = updated_q

            prev_ema = float(self._skill_ema.get(skill_id, current_q))
            self._skill_ema[skill_id] = prev_ema + self.alpha * (float(reward_total) - prev_ema)

        if self.storage_path is not None:
            self.save()
        return updated_q

    def select_fast_path_skill(
        self,
        context_hash: int,
        task_id: str,
        eligible_skill_ids: Iterable[str],
    ) -> tuple[str | None, float]:
        """Returns (best_skill_id, q_value) if Q(c, n, k) >= +10.0, bypassing LLM inference completely."""
        best_skill: str | None = None
        best_q = float("-inf")
        with self._lock:
            task_bucket = self._table.get(int(context_hash), {}).get(task_id, {})
            for sid in eligible_skill_ids:
                q_val = float(task_bucket.get(sid, self._skill_ema.get(sid, 0.0) * 0.5 if sid in task_bucket else float("-inf")))
                if sid in task_bucket:
                    q_val = float(task_bucket[sid])
                else:
                    continue
                if q_val >= self.fast_path_threshold and q_val > best_q:
                    best_q = q_val
                    best_skill = sid

        if best_skill is not None:
            LOG.info(
                "[Q-FASTPATH] Bypassing LLM! Context=0x%04x Task=%s Skill=%s Q=%+.2f (>= +%.1f)",
                int(context_hash),
                task_id,
                best_skill,
                best_q,
                self.fast_path_threshold,
            )
            return best_skill, best_q
        return None, 0.0

    def save(self) -> None:
        if self.storage_path is None:
            return
        with self._lock:
            serializable = {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "fast_path_threshold": self.fast_path_threshold,
                "q_values": {str(c): tasks for c, tasks in self._table.items()},
                "skill_ema": dict(self._skill_ema),
            }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(serializable, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.storage_path)

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.is_file():
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        q_raw = raw.get("q_values", {})
        ema_raw = raw.get("skill_ema", {})
        with self._lock:
            self._table.clear()
            for c_str, tasks in q_raw.items():
                c_int = int(c_str)
                self._table[c_int] = {
                    str(t_id): {str(s_id): float(q_v) for s_id, q_v in skills.items()}
                    for t_id, skills in tasks.items()
                    if isinstance(skills, dict)
                }
            self._skill_ema = {str(k): float(v) for k, v in ema_raw.items()}
