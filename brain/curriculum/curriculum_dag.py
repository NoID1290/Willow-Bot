"""Module 2: 20-Node Minecraft Survival & Advancement Curriculum DAG with Native Predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot, normalize_id

DEFAULT_DAG_JSON_PATH = Path(__file__).with_name("advancement_dag.json")


def _has_unlocked(state: BotTelemetrySnapshot, adv_suffix: str) -> bool:
    norm_target = normalize_id(adv_suffix)
    short_target = adv_suffix.split(":", 1)[-1]
    for unlocked in state.unlocked_advancements:
        if unlocked == norm_target or unlocked.endswith(short_target):
            return True
    return False


def _has_table(state: BotTelemetrySnapshot) -> bool:
    return "crafting_table" in state.nearby_workstations or state.count_item("minecraft:crafting_table") >= 1


def _has_furnace(state: BotTelemetrySnapshot) -> bool:
    return "furnace" in state.nearby_workstations or state.count_item("minecraft:furnace") >= 1


def _has_fuel(state: BotTelemetrySnapshot) -> bool:
    return (
        state.count_item("minecraft:coal")
        + state.count_item("minecraft:charcoal")
        + state.count_item("*_log")
        + state.count_item("*_planks")
        >= 1
    )


PRECONDITION_PREDICATES: dict[str, Callable[[BotTelemetrySnapshot], bool]] = {
    "t1_gather_logs": lambda s: True,
    "t1_craft_planks_sticks": lambda s: s.count_item("*_log") >= 3 or (s.count_item("*_planks") >= 8 and s.count_item("stick") >= 4),
    "t1_craft_table": lambda s: s.count_item("*_planks") >= 4 or s.count_item("*_log") >= 1,
    "t1_wooden_pickaxe": lambda s: _has_table(s) and s.count_item("*_planks") >= 3 and s.count_item("stick") >= 2,
    "t2_mine_cobblestone": lambda s: s.highest_tool_tier("_pickaxe") >= 1,
    "t2_stone_tools": lambda s: (
        (s.count_item("cobblestone") + s.count_item("cobbled_deepslate")) >= 11
        and s.count_item("stick") >= 6
    ),
    "t2_acquire_raw_food": lambda s: s.highest_tool_tier("_sword") >= 2,
    "t2_furnace_and_fuel": lambda s: (s.count_item("cobblestone") + s.count_item("cobbled_deepslate")) >= 8,
    "t2_cook_food_and_torches": lambda s: _has_furnace(s) and _has_fuel(s),
    "t2_survive_night_or_bed": lambda s: s.count_item("torch") >= 4,
    "t3_mine_iron_ore": lambda s: s.highest_tool_tier("_pickaxe") >= 2 and s.count_item("torch") >= 8,
    "t3_smelt_iron_ingots": lambda s: s.count_item("raw_iron") >= 1 or s.count_item("iron_ingot") >= 12,
    "t3_shield_and_water_bucket": lambda s: s.count_item("iron_ingot") >= 4 and s.count_item("*_planks") >= 6,
    "t3_iron_tools_and_armor": lambda s: s.count_item("iron_ingot") >= 13 and s.count_item("stick") >= 4,
    "t3_slay_hostile_mob": lambda s: s.highest_tool_tier("_sword") >= 3 and s.has_equipped_or_inv("shield"),
    "t4_deep_mining_prep": lambda s: s.highest_tool_tier("_pickaxe") >= 3,
    "t4_mine_diamonds": lambda s: s.highest_tool_tier("_pickaxe") >= 3 and s.count_item("water_bucket") >= 1,
    "t4_craft_diamond_gear": lambda s: s.count_item("diamond") >= 5 and s.count_item("stick") >= 4,
    "t4_harvest_obsidian": lambda s: s.count_item("diamond_pickaxe") >= 1 and s.count_item("water_bucket") >= 1,
    "t5_enter_the_nether": lambda s: s.count_item("obsidian") >= 10 and s.count_item("flint_and_steel") >= 1,
}

POSTCONDITION_PREDICATES: dict[str, Callable[[BotTelemetrySnapshot], bool]] = {
    "t1_gather_logs": lambda s: s.count_item("*_log") >= 4,
    "t1_craft_planks_sticks": lambda s: s.count_item("*_planks") >= 8 and s.count_item("stick") >= 4,
    "t1_craft_table": lambda s: s.count_item("crafting_table") >= 1 or "crafting_table" in s.nearby_workstations,
    "t1_wooden_pickaxe": lambda s: s.count_item("wooden_pickaxe") >= 1 or s.highest_tool_tier("_pickaxe") >= 1,
    "t2_mine_cobblestone": lambda s: (s.count_item("cobblestone") + s.count_item("cobbled_deepslate")) >= 16,
    "t2_stone_tools": lambda s: (
        (s.count_item("stone_pickaxe") >= 1 or s.highest_tool_tier("_pickaxe") >= 2)
        and (s.count_item("stone_axe") >= 1 or s.highest_tool_tier("_axe") >= 2)
        and (s.count_item("stone_sword") >= 1 or s.highest_tool_tier("_sword") >= 2)
    ),
    "t2_acquire_raw_food": lambda s: s.food_items_count() >= 5,
    "t2_furnace_and_fuel": lambda s: (
        (s.count_item("furnace") >= 1 or "furnace" in s.nearby_workstations)
        and (s.count_item("coal") + s.count_item("charcoal") >= 4)
    ),
    "t2_cook_food_and_torches": lambda s: s.cooked_food_count() >= 4 and s.count_item("torch") >= 8,
    "t2_survive_night_or_bed": lambda s: _has_unlocked(s, "adventure/sleep_in_bed") or s.enclosed_lit_shelter,
    "t3_mine_iron_ore": lambda s: (s.count_item("raw_iron") + s.count_item("iron_ingot")) >= 12,
    "t3_smelt_iron_ingots": lambda s: s.count_item("iron_ingot") >= 12,
    "t3_shield_and_water_bucket": lambda s: s.has_equipped_or_inv("shield") and s.count_item("water_bucket") >= 1,
    "t3_iron_tools_and_armor": lambda s: (
        (s.count_item("iron_pickaxe") >= 1 or s.highest_tool_tier("_pickaxe") >= 3)
        and (s.count_item("iron_sword") >= 1 or s.highest_tool_tier("_sword") >= 3)
        and s.vitals.armor_points >= 10
    ),
    "t3_slay_hostile_mob": lambda s: _has_unlocked(s, "adventure/kill_a_mob"),
    "t4_deep_mining_prep": lambda s: (
        s.count_item("torch") >= 32
        and s.cooked_food_count() >= 12
        and (s.count_item("iron_pickaxe") >= 2 or s.highest_tool_tier("_pickaxe") >= 4)
    ),
    "t4_mine_diamonds": lambda s: s.count_item("diamond") >= 5,
    "t4_craft_diamond_gear": lambda s: (
        (s.count_item("diamond_pickaxe") >= 1 or s.highest_tool_tier("_pickaxe") >= 4)
        and (s.count_item("diamond_sword") >= 1 or s.highest_tool_tier("_sword") >= 4)
    ),
    "t4_harvest_obsidian": lambda s: s.count_item("obsidian") >= 10,
    "t5_enter_the_nether": lambda s: s.dimension == "minecraft:the_nether",
}


@dataclass(slots=True)
class CurriculumNode:
    id: str
    tier: int
    description: str
    advancement_id: str | None
    prerequisites: tuple[str, ...]
    precondition_expr: str
    verification_expr: str
    base_reward: float
    target_items: dict[str, int]
    requires_movement: bool = True
    persistent_milestone: bool = False
    consecutive_failures: int = 0
    is_micro_task: bool = False
    parent_node_id: str | None = None
    _precondition_fn: Callable[[BotTelemetrySnapshot], bool] | None = field(default=None, repr=False)
    _postcondition_fn: Callable[[BotTelemetrySnapshot], bool] | None = field(default=None, repr=False)

    @property
    def verification_description(self) -> str:
        return self.verification_expr

    def check_preconditions(self, state: BotTelemetrySnapshot) -> bool:
        if self._precondition_fn is not None:
            return bool(self._precondition_fn(state))
        fn = PRECONDITION_PREDICATES.get(self.id)
        return bool(fn(state)) if fn is not None else True

    def verify_postcondition(self, state: BotTelemetrySnapshot) -> bool:
        if self._postcondition_fn is not None:
            return bool(self._postcondition_fn(state))
        fn = POSTCONDITION_PREDICATES.get(self.id)
        if fn is not None:
            return bool(fn(state))
        if not self.target_items:
            return False
        return all(state.count_item(pat) >= cnt for pat, cnt in self.target_items.items())

    def verify(self, s_prev: BotTelemetrySnapshot, s_curr: BotTelemetrySnapshot) -> bool:
        _ = s_prev
        return self.verify_postcondition(s_curr)

    def progress_ratio(self, s_prev: BotTelemetrySnapshot, s_curr: BotTelemetrySnapshot) -> float:
        if not self.target_items:
            return 1.0 if self.verify(s_prev, s_curr) else 0.0
        total_needed = 0
        gained = 0
        for pat, req in self.target_items.items():
            r = max(1, int(req))
            before = min(r, s_prev.count_item(pat))
            after = min(r, s_curr.count_item(pat))
            delta = max(0, after - before)
            gained += delta
            total_needed += max(1, r - before)
        if total_needed <= 0:
            return 0.0
        return max(0.0, min(1.0, gained / float(total_needed)))

    def precondition_match_ratio(self, state: BotTelemetrySnapshot) -> float:
        if self.check_preconditions(state):
            return 1.0
        if not self.target_items:
            return 0.5
        ratios = [
            min(1.0, state.count_item(pat) / float(max(1, req)))
            for pat, req in self.target_items.items()
        ]
        return sum(ratios) / len(ratios)


class CurriculumDAG:
    """Directed Acyclic Graph of the 20 progression milestones across Tiers 1-5."""

    def __init__(self, json_path: str | Path = DEFAULT_DAG_JSON_PATH) -> None:
        self.json_path = Path(json_path)
        self._nodes: dict[str, CurriculumNode] = {}
        self._topological_order: list[str] = []
        self._load_from_json(self.json_path)

    def _load_from_json(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw.get("nodes", []):
            nid = str(entry["id"])
            node = CurriculumNode(
                id=nid,
                tier=int(entry["tier"]),
                description=str(entry["description"]),
                advancement_id=entry.get("advancement_id"),
                prerequisites=tuple(str(p) for p in entry.get("prerequisites", [])),
                precondition_expr=str(entry.get("precondition_expr", "None")),
                verification_expr=str(entry.get("verification_expr", "")),
                base_reward=float(entry.get("base_reward", 50.0)),
                target_items={str(k): int(v) for k, v in (entry.get("target_items") or {}).items()},
                requires_movement=bool(entry.get("requires_movement", True)),
                persistent_milestone=bool(entry.get("persistent_milestone", False)),
                _precondition_fn=PRECONDITION_PREDICATES.get(nid),
                _postcondition_fn=POSTCONDITION_PREDICATES.get(nid),
            )
            self._nodes[nid] = node
        self._topological_order = self._compute_topological_sort()

    def _compute_topological_sort(self) -> list[str]:
        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            if nid in visiting:
                raise ValueError(f"Cycle detected in CurriculumDAG involving '{nid}'")
            visiting.add(nid)
            node = self._nodes[nid]
            for prereq in node.prerequisites:
                if prereq not in self._nodes:
                    raise KeyError(f"Unknown prerequisite '{prereq}' on node '{nid}'")
                dfs(prereq)
            visiting.remove(nid)
            visited.add(nid)
            order.append(nid)

        for nid in self._nodes:
            dfs(nid)
        return order

    @property
    def nodes(self) -> dict[str, CurriculumNode]:
        return self._nodes

    @property
    def ordered_nodes(self) -> list[CurriculumNode]:
        return [self._nodes[nid] for nid in self._topological_order]

    def get_node(self, node_id: str) -> CurriculumNode:
        return self._nodes[node_id]
