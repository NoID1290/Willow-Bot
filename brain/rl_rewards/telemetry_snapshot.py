"""Module 1: Immutable BotTelemetrySnapshot, Value-Type Perception Records & State Diffing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import fnmatch
import math
import time
from types import MappingProxyType
from typing import Any, Mapping


def normalize_id(raw_id: str) -> str:
    """Normalizes item/block/entity IDs to 'minecraft:<name>'."""
    if not raw_id:
        return ""
    cleaned = raw_id.strip().lower()
    if not cleaned:
        return ""
    if ":" in cleaned:
        return cleaned
    return f"minecraft:{cleaned}"


def matches_item_pattern(item_id: str, pattern: str) -> bool:
    """Matches exact IDs, wildcard patterns ('*_log', '*_planks'), or short names ('cobblestone')."""
    if not item_id or not pattern:
        return False
    norm_item = normalize_id(item_id)
    short_item = norm_item.split(":", 1)[-1]
    pat = pattern.strip().lower()
    if pat.startswith("#"):
        tag = pat[1:]
        if tag in ("logs", "log"):
            return short_item.endswith(("_log", "_stem"))
        if tag in ("planks", "plank"):
            return short_item.endswith("_planks")
        if tag in ("pickaxes", "pickaxe"):
            return short_item.endswith("_pickaxe")
        if tag in ("swords", "sword"):
            return short_item.endswith("_sword")
        if tag in ("axes", "axe"):
            return short_item.endswith("_axe")
        if tag in ("beds", "bed"):
            return short_item.endswith("_bed")
        if tag in ("wool",):
            return short_item.endswith("_wool")
        if tag in ("coals", "fuel_coal"):
            return short_item in ("coal", "charcoal")
    short_pat = pat.split(":", 1)[-1]
    if "*" in short_pat or "?" in short_pat:
        return fnmatch.fnmatchcase(short_item, short_pat) or fnmatch.fnmatchcase(norm_item, pat)
    return short_item == short_pat or norm_item == normalize_id(pat)


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def distance_squared(self, other: Vector3) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx * dx + dy * dy + dz * dz

    def distance_to(self, other: Vector3) -> float:
        return math.sqrt(self.distance_squared(other))

    def offset(self, dx: float, dy: float, dz: float) -> Vector3:
        return Vector3(self.x + dx, self.y + dy, self.z + dz)

    def to_block_tuple(self) -> tuple[int, int, int]:
        return (int(math.floor(self.x)), int(math.floor(self.y)), int(math.floor(self.z)))


@dataclass(frozen=True, slots=True)
class BlockInfo:
    block_id: str
    position: Vector3
    visible: bool = True
    reachable: bool = True
    requires_tool: bool = False
    has_tool: bool = True


@dataclass(frozen=True, slots=True)
class PerceivedEntity:
    entity_id: int
    entity_type: str
    position: Vector3
    distance: float
    health: float = 20.0
    is_hostile: bool = False
    is_alive: bool = True
    item_id: str = ""
    item_count: int = 0


class EquipmentSlot(str, Enum):
    MAIN_HAND = "MainHand"
    OFF_HAND = "OffHand"
    HEAD = "Head"
    CHEST = "Chest"
    LEGS = "Legs"
    FEET = "Feet"


@dataclass(frozen=True, slots=True)
class EquippedItemState:
    item_id: str
    current_durability: int = 0
    max_durability: int = 0


@dataclass(frozen=True, slots=True)
class VitalsState:
    health: float = 20.0
    max_health: float = 20.0
    absorption: float = 0.0
    food_level: int = 20
    saturation: float = 5.0
    armor_points: int = 0
    oxygen: int = 300
    is_alive: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionTelemetry:
    duration_ms: int = 0
    compilation_failed: bool = False
    runtime_exceptions: tuple[str, ...] = ()
    pathfinding_timeouts: int = 0
    blocks_mined_count: int = 0
    improper_tool_mine_attempts: int = 0
    tools_broken: tuple[str, ...] = ()


class ExecutionTelemetryRecorder:
    """Mutable low-allocation collector converted into an immutable ExecutionTelemetry on episode end."""

    __slots__ = (
        "started_ns",
        "compilation_failed",
        "runtime_exceptions",
        "pathfinding_timeouts",
        "blocks_mined_count",
        "improper_tool_mine_attempts",
        "tools_broken",
    )

    def __init__(self) -> None:
        self.started_ns: int = time.perf_counter_ns()
        self.compilation_failed: bool = False
        self.runtime_exceptions: list[str] = []
        self.pathfinding_timeouts: int = 0
        self.blocks_mined_count: int = 0
        self.improper_tool_mine_attempts: int = 0
        self.tools_broken: list[str] = []

    def reset(self) -> None:
        self.started_ns = time.perf_counter_ns()
        self.compilation_failed = False
        self.runtime_exceptions.clear()
        self.pathfinding_timeouts = 0
        self.blocks_mined_count = 0
        self.improper_tool_mine_attempts = 0
        self.tools_broken.clear()

    def record_exception(self, message: str) -> None:
        if len(self.runtime_exceptions) < 3:
            self.runtime_exceptions.append(str(message)[:200])

    def record_broken_tool(self, item_id: str) -> None:
        self.tools_broken.append(normalize_id(item_id))

    def freeze(self, override_duration_ms: int | None = None) -> ExecutionTelemetry:
        elapsed_ms = (
            int(override_duration_ms)
            if override_duration_ms is not None
            else max(0, int((time.perf_counter_ns() - self.started_ns) // 1_000_000))
        )
        return ExecutionTelemetry(
            duration_ms=elapsed_ms,
            compilation_failed=self.compilation_failed,
            runtime_exceptions=tuple(self.runtime_exceptions[:3]),
            pathfinding_timeouts=self.pathfinding_timeouts,
            blocks_mined_count=self.blocks_mined_count,
            improper_tool_mine_attempts=self.improper_tool_mine_attempts,
            tools_broken=tuple(self.tools_broken),
        )


WORKSTATION_TYPES = (
    "minecraft:crafting_table",
    "minecraft:furnace",
    "minecraft:blast_furnace",
    "minecraft:smoker",
    "minecraft:chest",
    "minecraft:nether_portal",
)

EDIBLE_FOOD_ITEMS = frozenset(
    {
        "minecraft:apple",
        "minecraft:baked_potato",
        "minecraft:beef",
        "minecraft:beetroot",
        "minecraft:beetroot_soup",
        "minecraft:bread",
        "minecraft:carrot",
        "minecraft:chicken",
        "minecraft:cod",
        "minecraft:cooked_beef",
        "minecraft:cooked_chicken",
        "minecraft:cooked_cod",
        "minecraft:cooked_mutton",
        "minecraft:cooked_porkchop",
        "minecraft:cooked_rabbit",
        "minecraft:cooked_salmon",
        "minecraft:cookie",
        "minecraft:dried_kelp",
        "minecraft:glow_berries",
        "minecraft:golden_apple",
        "minecraft:enchanted_golden_apple",
        "minecraft:golden_carrot",
        "minecraft:melon_slice",
        "minecraft:mushroom_stew",
        "minecraft:mutton",
        "minecraft:porkchop",
        "minecraft:potato",
        "minecraft:pumpkin_pie",
        "minecraft:rabbit",
        "minecraft:rabbit_stew",
        "minecraft:salmon",
        "minecraft:sweet_berries",
    }
)

COOKED_FOOD_ITEMS = frozenset(
    {
        "minecraft:cooked_beef",
        "minecraft:cooked_porkchop",
        "minecraft:cooked_mutton",
        "minecraft:cooked_chicken",
        "minecraft:cooked_rabbit",
        "minecraft:cooked_cod",
        "minecraft:cooked_salmon",
        "minecraft:baked_potato",
        "minecraft:bread",
        "minecraft:golden_carrot",
    }
)

VALUABLE_ITEM_KEYWORDS = (
    "diamond",
    "netherite",
    "iron_ingot",
    "raw_iron",
    "iron_pickaxe",
    "iron_sword",
    "iron_axe",
    "iron_helmet",
    "iron_chestplate",
    "iron_leggings",
    "iron_boots",
    "shield",
    "water_bucket",
    "bucket",
    "obsidian",
    "blaze_rod",
    "ender_pearl",
    "ender_eye",
    "golden_apple",
)


@dataclass(frozen=True, slots=True)
class BotTelemetrySnapshot:
    timestamp_ms: int
    world_tick: int
    time_of_day: int
    position: Vector3
    dimension: str
    biome: str
    light_level: int
    vitals: VitalsState
    item_counts: Mapping[str, int]
    empty_slots_count: int
    equipment: Mapping[EquipmentSlot, EquippedItemState]
    unlocked_advancements: frozenset[str]
    discovered_unique_items: frozenset[str]
    crafted_unique_items: frozenset[str]
    nearby_entities: tuple[PerceivedEntity, ...]
    nearby_workstations: Mapping[str, Vector3]
    execution_telemetry: ExecutionTelemetry = field(default_factory=ExecutionTelemetry)
    nearby_blocks: tuple[BlockInfo, ...] = ()
    enclosed_lit_shelter: bool = False

    def count_item(self, pattern: str) -> int:
        total = 0
        for item_id, cnt in self.item_counts.items():
            if cnt > 0 and matches_item_pattern(item_id, pattern):
                total += cnt
        return total

    def has_item(self, pattern: str, min_count: int = 1) -> bool:
        return self.count_item(pattern) >= min_count

    def has_equipped_or_inv(self, pattern: str) -> bool:
        if self.count_item(pattern) >= 1:
            return True
        for eq in self.equipment.values():
            if eq.item_id and matches_item_pattern(eq.item_id, pattern):
                return True
        return False

    def food_items_count(self) -> int:
        return sum(cnt for item_id, cnt in self.item_counts.items() if item_id in EDIBLE_FOOD_ITEMS and cnt > 0)

    def cooked_food_count(self) -> int:
        return sum(cnt for item_id, cnt in self.item_counts.items() if item_id in COOKED_FOOD_ITEMS and cnt > 0)

    def has_edible_food(self) -> bool:
        return self.food_items_count() > 0

    def highest_tool_tier(self, suffix: str = "_pickaxe") -> int:
        """0=None, 1=Wood/Gold, 2=Stone, 3=Iron, 4=Diamond, 5=Netherite."""
        tier_map = {
            "wooden": 1,
            "golden": 1,
            "stone": 2,
            "iron": 3,
            "diamond": 4,
            "netherite": 5,
        }
        best = 0
        for item_id, cnt in self.item_counts.items():
            if cnt <= 0:
                continue
            short = item_id.split(":", 1)[-1]
            if short.endswith(suffix):
                prefix = short[: -len(suffix)]
                best = max(best, tier_map.get(prefix, 0))
        for eq in self.equipment.values():
            if not eq.item_id:
                continue
            short = eq.item_id.split(":", 1)[-1]
            if short.endswith(suffix):
                prefix = short[: -len(suffix)]
                best = max(best, tier_map.get(prefix, 0))
        return best

    @classmethod
    def from_observation(
        cls,
        obs: Mapping[str, Any],
        *,
        discovered_items: set[str] | frozenset[str] | None = None,
        crafted_items: set[str] | frozenset[str] | None = None,
        execution_telemetry: ExecutionTelemetry | None = None,
        timestamp_ms: int | None = None,
    ) -> BotTelemetrySnapshot:
        now_ms = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
        raw_pos = obs.get("position") or [0, 64, 0]
        pos = Vector3(float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
        dimension = normalize_id(str(obs.get("dimension") or "minecraft:overworld"))
        biome = normalize_id(str(obs.get("biome") or "minecraft:plains"))

        sight = obs.get("sight") or obs.get("seeing") or {}
        lighting = sight.get("lighting") if isinstance(sight, dict) else {}
        if isinstance(lighting, dict) and ("block" in lighting or "sky" in lighting):
            light_level = max(int(lighting.get("block", 0)), int(lighting.get("sky", 0)))
        else:
            light_level = int(obs.get("light_level", 15))
        light_level = max(0, min(15, light_level))

        health = float(obs.get("health", 20.0))
        vitals = VitalsState(
            health=health,
            max_health=float(obs.get("max_health", 20.0)),
            absorption=float(obs.get("absorption", 0.0)),
            food_level=int(obs.get("food", 20)),
            saturation=float(obs.get("saturation", 5.0)),
            armor_points=int(obs.get("armor_points", _estimate_armor_points(obs))),
            oxygen=int(obs.get("air", 300)),
            is_alive=bool(obs.get("is_alive", health > 0.0)),
        )

        counts: dict[str, int] = {}
        raw_inv = obs.get("inventory")
        if isinstance(raw_inv, list):
            for entry in raw_inv:
                if isinstance(entry, str) and "=" in entry:
                    k, v = entry.rsplit("=", 1)
                    nk = normalize_id(k)
                    try:
                        c = int(v)
                    except ValueError:
                        continue
                    if c > 0:
                        counts[nk] = counts.get(nk, 0) + c
        elif isinstance(raw_inv, Mapping):
            for k, v in raw_inv.items():
                nk = normalize_id(str(k))
                c = int(v)
                if c > 0:
                    counts[nk] = counts.get(nk, 0) + c

        empty_slots = int(obs.get("free_slots", max(0, 36 - len(counts))))

        equipment: dict[EquipmentSlot, EquippedItemState] = {}
        raw_equip = obs.get("equipment")
        if isinstance(raw_equip, Mapping):
            for slot in EquipmentSlot:
                entry = raw_equip.get(slot.value) or raw_equip.get(slot.name.lower())
                if isinstance(entry, Mapping):
                    equipment[slot] = EquippedItemState(
                        item_id=normalize_id(str(entry.get("item_id") or entry.get("item") or "minecraft:air")),
                        current_durability=int(entry.get("current_durability", 0)),
                        max_durability=int(entry.get("max_durability", 0)),
                    )
        if EquipmentSlot.MAIN_HAND not in equipment:
            doing = obs.get("doing") if isinstance(obs.get("doing"), Mapping) else {}
            mh = doing.get("main_hand") if isinstance(doing, Mapping) else None
            oh = doing.get("off_hand") if isinstance(doing, Mapping) else None
            equipment[EquipmentSlot.MAIN_HAND] = EquippedItemState(
                item_id=normalize_id(str(mh.get("item", "minecraft:air"))) if isinstance(mh, Mapping) else "minecraft:air"
            )
            equipment[EquipmentSlot.OFF_HAND] = EquippedItemState(
                item_id=normalize_id(str(oh.get("item", "minecraft:air"))) if isinstance(oh, Mapping) else "minecraft:air"
            )
        for slot in EquipmentSlot:
            equipment.setdefault(slot, EquippedItemState("minecraft:air", 0, 0))

        adv_set = {normalize_id(str(a)) for a in (obs.get("unlocked_advancements") or []) if a}
        disc_set = set(discovered_items or ()) | {k for k, c in counts.items() if c > 0}
        craft_set = {normalize_id(str(x)) for x in (crafted_items or ())}
        for extra_crafted in obs.get("crafted_unique_items") or ():
            if extra_crafted:
                craft_set.add(normalize_id(str(extra_crafted)))

        entities_list: list[PerceivedEntity] = []
        for raw_ent in obs.get("entities") or ():
            if not isinstance(raw_ent, Mapping):
                continue
            at = raw_ent.get("at") or [pos.x, pos.y, pos.z]
            epos = Vector3(float(at[0]), float(at[1]), float(at[2]))
            dist = float(raw_ent.get("distance", pos.distance_to(epos)))
            if dist > 16.0:
                continue
            entities_list.append(
                PerceivedEntity(
                    entity_id=int(raw_ent.get("id", 0)),
                    entity_type=normalize_id(str(raw_ent.get("type", "minecraft:unknown"))),
                    position=epos,
                    distance=dist,
                    health=float(raw_ent.get("health", 20.0)),
                    is_hostile=bool(raw_ent.get("hostile", False)),
                    is_alive=bool(raw_ent.get("alive", True)),
                    item_id=normalize_id(str(raw_ent.get("item", ""))) if raw_ent.get("item") else "",
                    item_count=int(raw_ent.get("count", 0)),
                )
            )
        entities_list.sort(key=lambda e: e.distance)
        top_entities = tuple(entities_list[:12])

        blocks_list: list[BlockInfo] = []
        workstations: dict[str, Vector3] = {}
        raw_ws = obs.get("nearby_workstations")
        if isinstance(raw_ws, Mapping):
            for ws_key, coords in raw_ws.items():
                if isinstance(coords, (list, tuple)) and len(coords) == 3:
                    wvec = Vector3(float(coords[0]), float(coords[1]), float(coords[2]))
                    if pos.distance_to(wvec) <= 16.0:
                        workstations[_canonical_station_key(str(ws_key))] = wvec

        for raw_blk in obs.get("nearby_blocks") or ():
            if not isinstance(raw_blk, Mapping):
                continue
            bid = normalize_id(str(raw_blk.get("id", "")))
            at = raw_blk.get("at")
            if not bid or not isinstance(at, (list, tuple)) or len(at) != 3:
                continue
            bvec = Vector3(float(at[0]), float(at[1]), float(at[2]))
            blocks_list.append(
                BlockInfo(
                    block_id=bid,
                    position=bvec,
                    visible=bool(raw_blk.get("visible", True)),
                    reachable=bool(raw_blk.get("reachable", True)),
                    requires_tool=bool(raw_blk.get("requires_tool", False)),
                    has_tool=bool(raw_blk.get("has_tool", True)),
                )
            )
            station_key = _station_key_for_block(bid)
            if station_key and pos.distance_to(bvec) <= 16.0:
                existing = workstations.get(station_key)
                if existing is None or pos.distance_to(bvec) < pos.distance_to(existing):
                    workstations[station_key] = bvec

        crafting_meta = obs.get("crafting") if isinstance(obs.get("crafting"), Mapping) else {}
        if isinstance(crafting_meta, Mapping):
            if crafting_meta.get("table_reachable") and "crafting_table" not in workstations:
                workstations["crafting_table"] = pos
            if crafting_meta.get("furnace_reachable") and "furnace" not in workstations:
                workstations["furnace"] = pos
            if crafting_meta.get("chest_reachable") and "chest" not in workstations:
                workstations["chest"] = pos
            if crafting_meta.get("bed_reachable") and "bed" not in workstations:
                workstations["bed"] = pos

        exec_tel = execution_telemetry
        if exec_tel is None:
            raw_exec = obs.get("execution_telemetry")
            if isinstance(raw_exec, Mapping):
                exec_tel = ExecutionTelemetry(
                    duration_ms=int(raw_exec.get("duration_ms", 0)),
                    compilation_failed=bool(raw_exec.get("compilation_failed", False)),
                    runtime_exceptions=tuple(str(x) for x in (raw_exec.get("runtime_exceptions") or ())[:3]),
                    pathfinding_timeouts=int(raw_exec.get("pathfinding_timeouts", 0)),
                    blocks_mined_count=int(raw_exec.get("blocks_mined_count", 0)),
                    improper_tool_mine_attempts=int(raw_exec.get("improper_tool_mine_attempts", 0)),
                    tools_broken=tuple(normalize_id(str(x)) for x in (raw_exec.get("tools_broken") or ())),
                )
            else:
                last_res = obs.get("last_result") if isinstance(obs.get("last_result"), Mapping) else {}
                dur_ticks = int(last_res.get("duration_ticks", 0)) if isinstance(last_res, Mapping) else 0
                exec_tel = ExecutionTelemetry(duration_ms=dur_ticks * 50)

        enclosed_shelter = bool(obs.get("enclosed_lit_shelter", False))
        if not enclosed_shelter:
            terrain = obs.get("terrain") if isinstance(obs.get("terrain"), Mapping) else {}
            exits = terrain.get("exits") if isinstance(terrain, Mapping) else None
            if isinstance(exits, list) and exits:
                non_standable = sum(1 for e in exits if isinstance(e, Mapping) and not e.get("standable", True))
                if non_standable >= 6 and light_level >= 8:
                    enclosed_shelter = True

        time_of_day = int(obs.get("time", 0)) % 24000
        world_tick = int(obs.get("world_tick", obs.get("time", 0)))

        return cls(
            timestamp_ms=now_ms,
            world_tick=world_tick,
            time_of_day=time_of_day,
            position=pos,
            dimension=dimension,
            biome=biome,
            light_level=light_level,
            vitals=vitals,
            item_counts=MappingProxyType(counts),
            empty_slots_count=empty_slots,
            equipment=MappingProxyType(equipment),
            unlocked_advancements=frozenset(adv_set),
            discovered_unique_items=frozenset(disc_set),
            crafted_unique_items=frozenset(craft_set),
            nearby_entities=top_entities,
            nearby_workstations=MappingProxyType(workstations),
            execution_telemetry=exec_tel,
            nearby_blocks=tuple(blocks_list),
            enclosed_lit_shelter=enclosed_shelter,
        )


def _canonical_station_key(key: str) -> str:
    short = key.split(":", 1)[-1].lower()
    if short in ("blast_furnace", "smoker"):
        return "furnace"
    if short.endswith("_bed"):
        return "bed"
    if short in ("trapped_chest", "barrel"):
        return "chest"
    return short


def _station_key_for_block(block_id: str) -> str:
    short = block_id.split(":", 1)[-1]
    if short == "crafting_table":
        return "crafting_table"
    if short in ("furnace", "blast_furnace", "smoker"):
        return "furnace"
    if short in ("chest", "trapped_chest", "barrel"):
        return "chest"
    if short.endswith("_bed"):
        return "bed"
    if short == "nether_portal":
        return "nether_portal"
    return ""


def _estimate_armor_points(obs: Mapping[str, Any]) -> int:
    armor_values = {
        "iron_helmet": 2,
        "iron_chestplate": 6,
        "iron_leggings": 5,
        "iron_boots": 2,
        "diamond_helmet": 3,
        "diamond_chestplate": 8,
        "diamond_leggings": 6,
        "diamond_boots": 3,
        "netherite_helmet": 3,
        "netherite_chestplate": 8,
        "netherite_leggings": 6,
        "netherite_boots": 3,
        "leather_helmet": 1,
        "leather_chestplate": 3,
        "leather_leggings": 2,
        "leather_boots": 1,
    }
    total = 0
    raw_equip = obs.get("equipment")
    if isinstance(raw_equip, Mapping):
        for slot_name in ("Head", "Chest", "Legs", "Feet", "head", "chest", "legs", "feet"):
            entry = raw_equip.get(slot_name)
            if isinstance(entry, Mapping):
                short = str(entry.get("item_id") or entry.get("item") or "").split(":", 1)[-1]
                total += armor_values.get(short, 0)
        if total > 0:
            return min(20, total)
    raw_inv = obs.get("inventory")
    if isinstance(raw_inv, list):
        seen_slots: set[str] = set()
        for entry in raw_inv:
            if isinstance(entry, str) and "=" in entry:
                item_id, cnt_s = entry.rsplit("=", 1)
                if int(cnt_s) <= 0:
                    continue
                short = item_id.split(":", 1)[-1]
                if short in armor_values:
                    kind = short.split("_", 1)[-1]
                    if kind not in seen_slots:
                        seen_slots.add(kind)
                        total += armor_values[short]
    return min(20, total)


@dataclass(frozen=True, slots=True)
class TelemetryStateDiff:
    inventory_deltas: Mapping[str, int]
    new_advancements: frozenset[str]
    new_discovered_items: frozenset[str]
    new_crafted_items: frozenset[str]
    lost_valuable_items_count: int
    position_delta_l2: float
    health_delta: float
    food_delta: int

    @classmethod
    def compute(cls, s_prev: BotTelemetrySnapshot, s_curr: BotTelemetrySnapshot) -> TelemetryStateDiff:
        all_keys = set(s_prev.item_counts.keys()) | set(s_curr.item_counts.keys())
        deltas: dict[str, int] = {}
        lost_valuables = 0
        for k in all_keys:
            diff = s_curr.item_counts.get(k, 0) - s_prev.item_counts.get(k, 0)
            if diff != 0:
                deltas[k] = diff
            if diff < 0 and any(val_kw in k for val_kw in VALUABLE_ITEM_KEYWORDS):
                lost_valuables += -diff

        return cls(
            inventory_deltas=MappingProxyType(deltas),
            new_advancements=s_curr.unlocked_advancements - s_prev.unlocked_advancements,
            new_discovered_items=s_curr.discovered_unique_items - s_prev.discovered_unique_items,
            new_crafted_items=s_curr.crafted_unique_items - s_prev.crafted_unique_items,
            lost_valuable_items_count=lost_valuables,
            position_delta_l2=s_curr.position.distance_to(s_prev.position),
            health_delta=s_curr.vitals.health - s_prev.vitals.health,
            food_delta=s_curr.vitals.food_level - s_prev.vitals.food_level,
        )
