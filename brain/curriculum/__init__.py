"""Module 2: Minecraft Base Game Mechanics & 20-Node Survival Curriculum DAG."""

from .curriculum_dag import DEFAULT_DAG_JSON_PATH, CurriculumDAG, CurriculumNode
from .curriculum_manager import CurriculumDecomposer, CurriculumManager
from .mechanics_registry import (
    PICKAXE_TIER_NAMES,
    CraftingRecipeSpec,
    HarvestRule,
    RecipeKnowledgeGraph,
    SurvivalMechanicsRegistry,
)

__all__ = [
    "DEFAULT_DAG_JSON_PATH",
    "PICKAXE_TIER_NAMES",
    "CraftingRecipeSpec",
    "CurriculumDAG",
    "CurriculumDecomposer",
    "CurriculumManager",
    "CurriculumNode",
    "HarvestRule",
    "RecipeKnowledgeGraph",
    "SurvivalMechanicsRegistry",
]
