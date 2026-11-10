"""Module 1: Exact Mathematical Hybrid Extrinsic/Intrinsic RL Reward & Punishment Calculator."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from .telemetry_snapshot import BotTelemetrySnapshot, TelemetryStateDiff, normalize_id

LOG = logging.getLogger("localai.rl_reward")


class TaskLike(Protocol):
    id: str
    advancement_id: str | None
    base_reward: float
    target_items: dict[str, int]
    requires_movement: bool

    def verify(self, s_prev: BotTelemetrySnapshot, s_curr: BotTelemetrySnapshot) -> bool: ...
    def progress_ratio(self, s_prev: BotTelemetrySnapshot, s_curr: BotTelemetrySnapshot) -> float: ...


class CriticEvaluationLike(Protocol):
    success: bool
    failure_category: str
    summary: str


ADVANCEMENT_BASE_VALUES: dict[str, float] = {
    # Tier 1 (+50.0)
    "minecraft:story/root": 50.0,
    "minecraft:story/mine_stone": 50.0,
    "minecraft:husbandry/root": 50.0,
    "minecraft:adventure/root": 50.0,
    # Tier 2 (+100.0)
    "minecraft:story/upgrade_tools": 100.0,
    "minecraft:story/smelt_iron": 100.0,
    "minecraft:husbandry/plant_seed": 100.0,
    "minecraft:adventure/sleep_in_bed": 100.0,
    "minecraft:adventure/kill_a_mob": 100.0,
    # Tier 3 (+200.0)
    "minecraft:story/obtain_armor": 200.0,
    "minecraft:story/iron_tools": 200.0,
    "minecraft:story/deflect_arrow": 200.0,
    "minecraft:story/lava_bucket": 200.0,
    # Tier 4 (+350.0)
    "minecraft:story/mine_diamond": 350.0,
    "minecraft:story/shiny_gear": 350.0,
    "minecraft:story/form_obsidian": 350.0,
    "minecraft:story/enchant_item": 350.0,
    # Tier 5 (+600.0)
    "minecraft:story/enter_the_nether": 600.0,
    "minecraft:nether/obtain_blaze_rod": 600.0,
    "minecraft:story/follow_ender_eye": 600.0,
    "minecraft:story/enter_the_end": 600.0,
}

VALUABLE_TOOL_PREFIXES = ("iron_", "diamond_", "netherite_")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    r_total: float
    r_adv: float
    r_task: float
    r_shaping: float
    r_intrinsic: float
    p_vital: float
    p_exec: float
    p_time: float

    def to_dict(self) -> dict[str, float]:
        return {
            "r_total": round(self.r_total, 3),
            "r_adv": round(self.r_adv, 3),
            "r_task": round(self.r_task, 3),
            "r_shaping": round(self.r_shaping, 3),
            "r_intrinsic": round(self.r_intrinsic, 3),
            "p_vital": round(self.p_vital, 3),
            "p_exec": round(self.p_exec, 3),
            "p_time": round(self.p_time, 3),
        }


class RewardCalculator:
    """Computes R_total = R_adv + R_task + R_shaping + R_intrinsic - P_vital - P_exec - P_time."""

    @staticmethod
    def advancement_base_value(adv_id: str) -> float:
        norm = normalize_id(adv_id)
        if norm in ADVANCEMENT_BASE_VALUES:
            return ADVANCEMENT_BASE_VALUES[norm]
        if "/nether/" in norm or norm.startswith("minecraft:nether/") or norm.startswith("minecraft:end/"):
            return 350.0
        return 50.0

    @classmethod
    def calculate_episode_reward(
        cls,
        s_prev: BotTelemetrySnapshot,
        s_curr: BotTelemetrySnapshot,
        task: TaskLike,
        critic: CriticEvaluationLike,
    ) -> RewardBreakdown:
        diff = TelemetryStateDiff.compute(s_prev, s_curr)

        # 1. Extrinsic Advancement Rewards (R_adv)
        r_adv = 0.0
        target_adv = normalize_id(task.advancement_id) if task.advancement_id else ""
        for adv in diff.new_advancements:
            norm_adv = normalize_id(adv)
            bonus = 50.0 if (target_adv and norm_adv == target_adv) else 0.0
            r_adv += cls.advancement_base_value(norm_adv) + bonus

        # 2. Task Completion Reward (R_task)
        verified = bool(task.verify(s_prev, s_curr))
        if verified and critic.success:
            r_task = float(task.base_reward)
        else:
            prog = max(0.0, min(1.0, float(task.progress_ratio(s_prev, s_curr))))
            r_task = 0.25 * float(task.base_reward) * prog if prog > 0.0 else 0.0

        # 3. Dense Inventory & Gear Shaping (R_shaping)
        r_shaping = 0.0
        for item_pattern, req_count in task.target_items.items():
            req = max(1, int(req_count))
            prev_capped = min(req, s_prev.count_item(item_pattern))
            curr_capped = min(req, s_curr.count_item(item_pattern))
            delta_n = curr_capped - prev_capped
            if delta_n > 0:
                r_shaping += 5.0 * float(delta_n)

        armor_upgraded = s_curr.vitals.armor_points > s_prev.vitals.armor_points
        pick_upgraded = s_curr.highest_tool_tier("_pickaxe") > s_prev.highest_tool_tier("_pickaxe")
        sword_upgraded = s_curr.highest_tool_tier("_sword") > s_prev.highest_tool_tier("_sword")
        if armor_upgraded or pick_upgraded or sword_upgraded:
            r_shaping += 15.0

        if s_prev.vitals.food_level <= 12 and s_curr.vitals.food_level >= 18:
            r_shaping += 5.0

        # 4. Intrinsic Curiosity & Novelty Bonus (R_intrinsic)
        r_intrinsic = (3.0 * float(len(diff.new_discovered_items))) + (8.0 * float(len(diff.new_crafted_items)))

        # 5. Vitality & Survival Punishments (P_vital)
        p_vital = 0.0
        if (not s_curr.vitals.is_alive) or s_curr.vitals.health <= 0.0:
            p_vital += 250.0 + 10.0 * float(diff.lost_valuable_items_count)
        damage_taken = max(0.0, s_prev.vitals.health - s_curr.vitals.health)
        p_vital += 3.5 * damage_taken

        if s_curr.vitals.food_level == 0:
            p_vital += 15.0
        if (
            s_curr.vitals.food_level <= 6
            and s_prev.has_edible_food()
            and s_curr.food_items_count() >= s_prev.food_items_count()
            and s_curr.vitals.food_level <= s_prev.vitals.food_level
        ):
            p_vital += 5.0

        # 6. Execution, Tool & Pathing Punishments (P_exec)
        tel = s_curr.execution_telemetry
        p_exec = 0.0
        if tel.compilation_failed:
            p_exec += 15.0
        if tel.runtime_exceptions:
            p_exec += 20.0 * float(min(3, len(tel.runtime_exceptions)))
        for broken in tel.tools_broken:
            short_broken = broken.split(":", 1)[-1]
            if short_broken.startswith(VALUABLE_TOOL_PREFIXES):
                p_exec += 35.0
        if tel.improper_tool_mine_attempts > 0:
            p_exec += 8.0 * float(tel.improper_tool_mine_attempts)
        if tel.pathfinding_timeouts > 0 or (
            task.requires_movement and diff.position_delta_l2 < 0.5 and tel.duration_ms > 10_000
        ):
            p_exec += 12.0

        # 7. Temporal Cost (P_time)
        p_time = 0.08 * (float(tel.duration_ms) / 1000.0)

        r_total = r_adv + r_task + r_shaping + r_intrinsic - p_vital - p_exec - p_time
        breakdown = RewardBreakdown(
            r_total=r_total,
            r_adv=r_adv,
            r_task=r_task,
            r_shaping=r_shaping,
            r_intrinsic=r_intrinsic,
            p_vital=p_vital,
            p_exec=p_exec,
            p_time=p_time,
        )
        LOG.info(
            "[RL-REWARD] task=%s total=%+.2f (adv=+%.1f task=+%.1f shape=+%.1f intr=+%.1f | vital=-%.1f exec=-%.1f time=-%.2f)",
            task.id,
            breakdown.r_total,
            breakdown.r_adv,
            breakdown.r_task,
            breakdown.r_shaping,
            breakdown.r_intrinsic,
            breakdown.p_vital,
            breakdown.p_exec,
            breakdown.p_time,
        )
        return breakdown

    # PascalCase alias matching C# architectural specification
    CalculateEpisodeReward = calculate_episode_reward
