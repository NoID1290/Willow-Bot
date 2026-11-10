"""Module 0: Zero-Allocation LiveReflexState, Reusable Buffer Pools & Collectible LRU-64 Skill Cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import threading
import time
from typing import Any, Callable

LOG = logging.getLogger("localai.vram_opt")

MAX_COMPILED_SKILLS_IN_RAM = 64
SNAPSHOT_MIN_INTERVAL_TICKS = 20


class LiveReflexState:
    """Pre-allocated, in-place updated mutable state for the 20Hz Fast Reactive Reflex Loop (0 GC churn)."""

    __slots__ = (
        "world_tick",
        "health",
        "food_level",
        "saturation",
        "oxygen",
        "fire_ticks",
        "fall_distance",
        "is_alive",
        "in_lava",
        "lava_distance",
        "suffocating",
        "nearest_hostile_id",
        "nearest_hostile_type",
        "nearest_hostile_distance",
        "creeper_primed",
        "skeleton_line_of_sight",
        "zombie_swarm_count",
        "has_shield",
        "shield_equipped",
        "has_water_bucket",
        "best_food_item",
        "pos_x",
        "pos_y",
        "pos_z",
        "combat_or_flee_active",
        "active_reflex_reason",
    )

    def __init__(self) -> None:
        self.world_tick: int = 0
        self.health: float = 20.0
        self.food_level: int = 20
        self.saturation: float = 5.0
        self.oxygen: int = 300
        self.fire_ticks: int = 0
        self.fall_distance: float = 0.0
        self.is_alive: bool = True
        self.in_lava: bool = False
        self.lava_distance: float = 99.0
        self.suffocating: bool = False
        self.nearest_hostile_id: int = -1
        self.nearest_hostile_type: str = ""
        self.nearest_hostile_distance: float = 99.0
        self.creeper_primed: bool = False
        self.skeleton_line_of_sight: bool = False
        self.zombie_swarm_count: int = 0
        self.has_shield: bool = False
        self.shield_equipped: bool = False
        self.has_water_bucket: bool = False
        self.best_food_item: str = ""
        self.pos_x: float = 0.0
        self.pos_y: float = 64.0
        self.pos_z: float = 0.0
        self.combat_or_flee_active: bool = False
        self.active_reflex_reason: str = ""

    def update_in_place(self, obs: dict[str, Any]) -> None:
        """Mutates fields directly from raw tick observation without allocating intermediate objects."""
        self.world_tick = int(obs.get("world_tick", obs.get("time", self.world_tick)))
        self.health = float(obs.get("health", 20.0))
        self.food_level = int(obs.get("food", 20))
        self.saturation = float(obs.get("saturation", 5.0))
        self.oxygen = int(obs.get("air", 300))
        self.fire_ticks = int(obs.get("fire_ticks", 0))
        self.fall_distance = float(obs.get("fall_distance", 0.0))
        self.is_alive = bool(obs.get("is_alive", self.health > 0.0))
        self.in_lava = bool(obs.get("in_lava", False))
        self.suffocating = bool(obs.get("suffocating", False))

        pos = obs.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            self.pos_x = float(pos[0])
            self.pos_y = float(pos[1])
            self.pos_z = float(pos[2])

        # Scan nearby blocks for lava within 2 blocks without list comprehensions
        lava_dist = 99.0
        if self.in_lava:
            lava_dist = 0.0
        else:
            nearby_blocks = obs.get("nearby_blocks")
            if isinstance(nearby_blocks, list):
                for blk in nearby_blocks:
                    if isinstance(blk, dict) and blk.get("id") in ("minecraft:lava", "lava"):
                        at = blk.get("at")
                        if isinstance(at, (list, tuple)) and len(at) == 3:
                            dx = float(at[0]) - self.pos_x
                            dy = float(at[1]) - self.pos_y
                            dz = float(at[2]) - self.pos_z
                            d = (dx * dx + dy * dy + dz * dz) ** 0.5
                            if d < lava_dist:
                                lava_dist = d
        self.lava_distance = lava_dist

        # Scan entities for immediate survival threats (< 6.0m, Creeper fuse, Skeleton LOS, Zombie swarm)
        nearest_id = -1
        nearest_type = ""
        nearest_dist = 99.0
        creeper_fuse = False
        skel_los = False
        zombies = 0

        hearing = obs.get("hearing") or obs.get("listening")
        if isinstance(hearing, dict):
            alerts = hearing.get("active_alerts")
            if isinstance(alerts, list):
                for alert in alerts:
                    if isinstance(alert, dict) and "creeper.primed" in str(alert.get("sound", "")):
                        if float(alert.get("distance", 99.0)) < 7.0:
                            creeper_fuse = True

        entities = obs.get("entities")
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, dict) or not ent.get("hostile"):
                    continue
                dist = float(ent.get("distance", 99.0))
                etype = str(ent.get("type", ""))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = int(ent.get("id", -1))
                    nearest_type = etype
                if "creeper" in etype and (ent.get("ignited") or ent.get("fuse", 0) > 0 or dist < 4.0):
                    if dist < 6.0:
                        creeper_fuse = True
                if "skeleton" in etype and dist < 12.0 and ent.get("line_of_sight", True):
                    skel_los = True
                if "zombie" in etype and dist < 6.0:
                    zombies += 1

        self.nearest_hostile_id = nearest_id
        self.nearest_hostile_type = nearest_type
        self.nearest_hostile_distance = nearest_dist
        self.creeper_primed = creeper_fuse
        self.skeleton_line_of_sight = skel_los
        self.zombie_swarm_count = zombies

        # Inventory scan for shield, water_bucket, and highest-value edible food
        has_shield = False
        has_water = False
        best_food = ""
        best_food_rank = -1
        inv = obs.get("inventory")
        if isinstance(inv, list):
            for raw in inv:
                if not isinstance(raw, str) or "=" not in raw:
                    continue
                item_id, cnt_str = raw.rsplit("=", 1)
                try:
                    cnt = int(cnt_str)
                except ValueError:
                    continue
                if cnt <= 0:
                    continue
                short = item_id.split(":", 1)[-1]
                if short == "shield":
                    has_shield = True
                elif short == "water_bucket":
                    has_water = True
                else:
                    rank = _food_saturation_rank(short)
                    if rank > best_food_rank:
                        best_food_rank = rank
                        best_food = item_id

        doing = obs.get("doing")
        shield_eq = False
        if isinstance(doing, dict):
            off = doing.get("off_hand")
            main = doing.get("main_hand")
            if isinstance(off, dict) and "shield" in str(off.get("item", "")):
                shield_eq = True
                has_shield = True
            if isinstance(main, dict) and "shield" in str(main.get("item", "")):
                shield_eq = True
                has_shield = True

        self.has_shield = has_shield
        self.shield_equipped = shield_eq
        self.has_water_bucket = has_water
        self.best_food_item = best_food

    def evaluate_hazard(self) -> tuple[bool, str, str]:
        """Returns (triggered, hazard_code, recommended_reflex_action) with 0 allocations."""
        if not self.is_alive:
            return False, "NONE", ""
        if self.fall_distance > 3.0 and self.has_water_bucket:
            return True, "FALL_MLG", "water_bucket_mlg"
        if (self.in_lava or self.lava_distance <= 2.0 or self.fire_ticks > 0):
            if self.has_water_bucket and (self.in_lava or self.fire_ticks > 0):
                return True, "LAVA_OR_FIRE", "extinguish_water"
            return True, "LAVA_PROXIMITY", "escape_lava"
        if self.suffocating:
            return True, "SUFFOCATION", "escape_suffocation"
        if self.oxygen < 60:
            return True, "LOW_OXYGEN", "surface_oxygen"
        if self.creeper_primed:
            return True, "CREEPER_FUSE", "shield_or_flee" if self.has_shield else "sprint_away"
        if self.nearest_hostile_distance < 6.0:
            if self.health <= 8.0:
                return True, "CRITICAL_HP_COMBAT", "shield_or_flee" if self.has_shield else "pillar_or_flee"
            if self.zombie_swarm_count >= 2 or (self.skeleton_line_of_sight and self.has_shield):
                return True, "HOSTILE_SWARM_OR_ARCHER", "defend_or_engage"
            return True, "HOSTILE_CLOSE", "engage_hostile"
        if self.health <= 8.0 and self.best_food_item and self.food_level < 20:
            return True, "CRITICAL_VITALS_HP", "eat_emergency"
        if self.food_level <= 6 and self.best_food_item:
            return True, "CRITICAL_VITALS_FOOD", "eat_emergency"
        return False, "NONE", ""


def _food_saturation_rank(short_id: str) -> int:
    if short_id in ("golden_carrot", "cooked_porkchop", "cooked_beef"):
        return 100
    if short_id in ("cooked_mutton", "cooked_salmon", "golden_apple"):
        return 85
    if short_id in ("cooked_chicken", "cooked_rabbit", "baked_potato", "bread"):
        return 70
    if short_id in ("cooked_cod", "carrot", "apple", "melon_slice", "sweet_berries"):
        return 45
    if short_id in ("beef", "porkchop", "mutton", "rabbit", "potato"):
        return 20
    return -1


@dataclass(slots=True)
class CompiledSkillEntry:
    code_hash: str
    skill_name: str
    fn: Callable[..., Any]
    scope: dict[str, Any]
    compiled_at: float


class CompiledSkillCache:
    """Thread-safe SHA256 LRU cache (max 64 skills) with explicit isolated-scope unloading."""

    def __init__(self, max_entries: int = MAX_COMPILED_SKILLS_IN_RAM) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, CompiledSkillEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._unloads = 0

    @staticmethod
    def compute_hash(source_code: str) -> str:
        return hashlib.sha256(source_code.encode("utf-8")).hexdigest()

    def get(self, code_hash: str) -> CompiledSkillEntry | None:
        with self._lock:
            entry = self._entries.get(code_hash)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(code_hash)
            self._hits += 1
            return entry

    def put(
        self,
        code_hash: str,
        skill_name: str,
        fn: Callable[..., Any],
        scope: dict[str, Any],
    ) -> CompiledSkillEntry:
        evicted: list[CompiledSkillEntry] = []
        entry = CompiledSkillEntry(
            code_hash=code_hash,
            skill_name=skill_name,
            fn=fn,
            scope=scope,
            compiled_at=time.monotonic(),
        )
        with self._lock:
            old = self._entries.pop(code_hash, None)
            if old is not None:
                evicted.append(old)
            self._entries[code_hash] = entry
            while len(self._entries) > self.max_entries:
                _, oldest = self._entries.popitem(last=False)
                evicted.append(oldest)
                self._unloads += 1

        for item in evicted:
            self._teardown_scope(item)
        return entry

    def unload(self, code_hash: str) -> bool:
        """Immediately unloads a failed or deprecated skill's isolated namespace from RAM."""
        with self._lock:
            removed = self._entries.pop(code_hash, None)
            if removed is not None:
                self._unloads += 1
        if removed is not None:
            self._teardown_scope(removed)
            return True
        return False

    def clear(self) -> None:
        with self._lock:
            items = list(self._entries.values())
            self._entries.clear()
            self._unloads += len(items)
        for item in items:
            self._teardown_scope(item)

    @staticmethod
    def _teardown_scope(entry: CompiledSkillEntry) -> None:
        try:
            entry.scope.clear()
        except Exception:
            pass

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "cached_skills": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "unloads": self._unloads,
            }


class MemoryPoolManager:
    """Centralized low-GC buffer pool, 1Hz snapshot throttle, and LRU-64 compiled skill cache."""

    def __init__(self, max_compiled_skills: int = MAX_COMPILED_SKILLS_IN_RAM) -> None:
        self.reflex_state = LiveReflexState()
        self.skill_cache = CompiledSkillCache(max_entries=max_compiled_skills)
        self._buffer_lock = threading.Lock()
        self._block_scan_pool: list[list[Any]] = []
        self._entity_scan_pool: list[list[Any]] = []
        self._last_snapshot_tick: int = -100
        self._last_snapshot: Any = None

    def rent_block_buffer(self) -> list[Any]:
        with self._buffer_lock:
            if self._block_scan_pool:
                buf = self._block_scan_pool.pop()
                buf.clear()
                return buf
        return []

    def return_block_buffer(self, buf: list[Any]) -> None:
        buf.clear()
        with self._buffer_lock:
            if len(self._block_scan_pool) < 8:
                self._block_scan_pool.append(buf)

    def rent_entity_buffer(self) -> list[Any]:
        with self._buffer_lock:
            if self._entity_scan_pool:
                buf = self._entity_scan_pool.pop()
                buf.clear()
                return buf
        return []

    def return_entity_buffer(self, buf: list[Any]) -> None:
        buf.clear()
        with self._buffer_lock:
            if len(self._entity_scan_pool) < 8:
                self._entity_scan_pool.append(buf)

    def should_capture_snapshot(self, world_tick: int, force_episode_boundary: bool = False) -> bool:
        if force_episode_boundary or self._last_snapshot is None:
            return True
        return abs(int(world_tick) - self._last_snapshot_tick) >= SNAPSHOT_MIN_INTERVAL_TICKS

    def record_snapshot(self, world_tick: int, snapshot: Any) -> Any:
        self._last_snapshot_tick = int(world_tick)
        self._last_snapshot = snapshot
        return snapshot

    @property
    def cached_snapshot(self) -> Any:
        return self._last_snapshot
