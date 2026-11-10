"""Module 1: Telemetry & Reinforcement Learning Reward/Punishment Engine."""

from .q_table import DEFAULT_ALPHA, DEFAULT_GAMMA, FAST_PATH_Q_THRESHOLD, ContextStateHash, SkillQTable
from .reflexion_log import EpisodeReflectionRecord, ReflexionMemoryLog
from .reward_calculator import ADVANCEMENT_BASE_VALUES, RewardBreakdown, RewardCalculator
from .telemetry_snapshot import (
    COOKED_FOOD_ITEMS,
    EDIBLE_FOOD_ITEMS,
    BlockInfo,
    BotTelemetrySnapshot,
    EquipmentSlot,
    EquippedItemState,
    ExecutionTelemetry,
    ExecutionTelemetryRecorder,
    PerceivedEntity,
    TelemetryStateDiff,
    Vector3,
    VitalsState,
    matches_item_pattern,
    normalize_id,
)

__all__ = [
    "ADVANCEMENT_BASE_VALUES",
    "COOKED_FOOD_ITEMS",
    "DEFAULT_ALPHA",
    "DEFAULT_GAMMA",
    "EDIBLE_FOOD_ITEMS",
    "FAST_PATH_Q_THRESHOLD",
    "BlockInfo",
    "BotTelemetrySnapshot",
    "ContextStateHash",
    "EpisodeReflectionRecord",
    "EquipmentSlot",
    "EquippedItemState",
    "ExecutionTelemetry",
    "ExecutionTelemetryRecorder",
    "PerceivedEntity",
    "ReflexionMemoryLog",
    "RewardBreakdown",
    "RewardCalculator",
    "SkillQTable",
    "TelemetryStateDiff",
    "Vector3",
    "VitalsState",
    "matches_item_pattern",
    "normalize_id",
]
