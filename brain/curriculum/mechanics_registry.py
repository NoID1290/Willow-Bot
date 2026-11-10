"""Module 2: Deterministic Minecraft Survival Mechanics Knowledge Base & Task-Filtered Rule Injector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot, normalize_id

if TYPE_CHECKING:
    from .curriculum_dag import CurriculumNode


@dataclass(frozen=True, slots=True)
class HarvestRule:
    block_id: str
    min_pickaxe_tier: int  # 0=hand, 1=wooden, 2=stone, 3=iron, 4=diamond
    optimal_tool: str
    drops_item: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CraftingRecipeSpec:
    output_item: str
    output_count: int
    ingredients: dict[str, int]
    requires_crafting_table: bool


PICKAXE_TIER_NAMES: dict[int, str] = {
    0: "hand",
    1: "wooden_pickaxe",
    2: "stone_pickaxe",
    3: "iron_pickaxe",
    4: "diamond_pickaxe",
    5: "netherite_pickaxe",
}


class RecipeKnowledgeGraph:
    """Exact crafting dependency tree for 2x2 inventory and 3x3 crafting_table recipes."""

    RECIPES: dict[str, CraftingRecipeSpec] = {
        "planks": CraftingRecipeSpec("minecraft:oak_planks", 4, {"*_log": 1}, False),
        "oak_planks": CraftingRecipeSpec("minecraft:oak_planks", 4, {"*_log": 1}, False),
        "birch_planks": CraftingRecipeSpec("minecraft:birch_planks", 4, {"*_log": 1}, False),
        "spruce_planks": CraftingRecipeSpec("minecraft:spruce_planks", 4, {"*_log": 1}, False),
        "stick": CraftingRecipeSpec("minecraft:stick", 4, {"*_planks": 2}, False),
        "crafting_table": CraftingRecipeSpec("minecraft:crafting_table", 1, {"*_planks": 4}, False),
        "torch": CraftingRecipeSpec("minecraft:torch", 4, {"#coals": 1, "minecraft:stick": 1}, False),
        "flint_and_steel": CraftingRecipeSpec(
            "minecraft:flint_and_steel", 1, {"minecraft:iron_ingot": 1, "minecraft:flint": 1}, False
        ),
        "wooden_pickaxe": CraftingRecipeSpec(
            "minecraft:wooden_pickaxe", 1, {"*_planks": 3, "minecraft:stick": 2}, True
        ),
        "wooden_axe": CraftingRecipeSpec(
            "minecraft:wooden_axe", 1, {"*_planks": 3, "minecraft:stick": 2}, True
        ),
        "wooden_sword": CraftingRecipeSpec(
            "minecraft:wooden_sword", 1, {"*_planks": 2, "minecraft:stick": 1}, True
        ),
        "stone_pickaxe": CraftingRecipeSpec(
            "minecraft:stone_pickaxe", 1, {"minecraft:cobblestone": 3, "minecraft:stick": 2}, True
        ),
        "stone_axe": CraftingRecipeSpec(
            "minecraft:stone_axe", 1, {"minecraft:cobblestone": 3, "minecraft:stick": 2}, True
        ),
        "stone_sword": CraftingRecipeSpec(
            "minecraft:stone_sword", 1, {"minecraft:cobblestone": 2, "minecraft:stick": 1}, True
        ),
        "furnace": CraftingRecipeSpec("minecraft:furnace", 1, {"minecraft:cobblestone": 8}, True),
        "chest": CraftingRecipeSpec("minecraft:chest", 1, {"*_planks": 8}, True),
        "bed": CraftingRecipeSpec("minecraft:white_bed", 1, {"*_wool": 3, "*_planks": 3}, True),
        "white_bed": CraftingRecipeSpec("minecraft:white_bed", 1, {"*_wool": 3, "*_planks": 3}, True),
        "shield": CraftingRecipeSpec(
            "minecraft:shield", 1, {"*_planks": 6, "minecraft:iron_ingot": 1}, True
        ),
        "bucket": CraftingRecipeSpec("minecraft:bucket", 1, {"minecraft:iron_ingot": 3}, True),
        "iron_pickaxe": CraftingRecipeSpec(
            "minecraft:iron_pickaxe", 1, {"minecraft:iron_ingot": 3, "minecraft:stick": 2}, True
        ),
        "iron_axe": CraftingRecipeSpec(
            "minecraft:iron_axe", 1, {"minecraft:iron_ingot": 3, "minecraft:stick": 2}, True
        ),
        "iron_sword": CraftingRecipeSpec(
            "minecraft:iron_sword", 1, {"minecraft:iron_ingot": 2, "minecraft:stick": 1}, True
        ),
        "iron_helmet": CraftingRecipeSpec("minecraft:iron_helmet", 1, {"minecraft:iron_ingot": 5}, True),
        "iron_chestplate": CraftingRecipeSpec("minecraft:iron_chestplate", 1, {"minecraft:iron_ingot": 8}, True),
        "iron_leggings": CraftingRecipeSpec("minecraft:iron_leggings", 1, {"minecraft:iron_ingot": 7}, True),
        "iron_boots": CraftingRecipeSpec("minecraft:iron_boots", 1, {"minecraft:iron_ingot": 4}, True),
        "diamond_pickaxe": CraftingRecipeSpec(
            "minecraft:diamond_pickaxe", 1, {"minecraft:diamond": 3, "minecraft:stick": 2}, True
        ),
        "diamond_sword": CraftingRecipeSpec(
            "minecraft:diamond_sword", 1, {"minecraft:diamond": 2, "minecraft:stick": 1}, True
        ),
    }

    @classmethod
    def get_recipe(cls, item_id: str) -> CraftingRecipeSpec | None:
        short = item_id.split(":", 1)[-1].lower()
        if short in cls.RECIPES:
            return cls.RECIPES[short]
        if short.endswith("_planks"):
            return CraftingRecipeSpec(normalize_id(short), 4, {"*_log": 1}, False)
        if short.endswith("_bed"):
            return CraftingRecipeSpec(normalize_id(short), 1, {"*_wool": 3, "*_planks": 3}, True)
        return None

    @classmethod
    def format_recipe_compact(cls, item_id: str) -> str:
        spec = cls.get_recipe(item_id)
        if spec is None:
            return ""
        short_out = spec.output_item.split(":", 1)[-1]
        ing = "+".join(f"{cnt}x {k.split(':', 1)[-1]}" for k, cnt in spec.ingredients.items())
        grid = "table" if spec.requires_crafting_table else "2x2"
        return f"{short_out}(x{spec.output_count})<-{ing}[{grid}]"


class SurvivalMechanicsRegistry:
    """Static deterministic registry of harvest tiers, crafting trees, fuels, and Y-level rules."""

    HARVEST_MATRIX: dict[str, HarvestRule] = {
        # Hand-harvestable (tier 0)
        "oak_log": HarvestRule("minecraft:oak_log", 0, "axe", "minecraft:oak_log"),
        "birch_log": HarvestRule("minecraft:birch_log", 0, "axe", "minecraft:birch_log"),
        "spruce_log": HarvestRule("minecraft:spruce_log", 0, "axe", "minecraft:spruce_log"),
        "jungle_log": HarvestRule("minecraft:jungle_log", 0, "axe", "minecraft:jungle_log"),
        "acacia_log": HarvestRule("minecraft:acacia_log", 0, "axe", "minecraft:acacia_log"),
        "dark_oak_log": HarvestRule("minecraft:dark_oak_log", 0, "axe", "minecraft:dark_oak_log"),
        "cherry_log": HarvestRule("minecraft:cherry_log", 0, "axe", "minecraft:cherry_log"),
        "mangrove_log": HarvestRule("minecraft:mangrove_log", 0, "axe", "minecraft:mangrove_log"),
        "dirt": HarvestRule("minecraft:dirt", 0, "shovel", "minecraft:dirt"),
        "grass_block": HarvestRule("minecraft:grass_block", 0, "shovel", "minecraft:dirt"),
        "sand": HarvestRule("minecraft:sand", 0, "shovel", "minecraft:sand"),
        "gravel": HarvestRule("minecraft:gravel", 0, "shovel", "minecraft:gravel", "Chance to drop flint"),
        # Requires >= wooden_pickaxe (tier 1)
        "stone": HarvestRule("minecraft:stone", 1, "wooden_pickaxe", "minecraft:cobblestone"),
        "cobblestone": HarvestRule("minecraft:cobblestone", 1, "wooden_pickaxe", "minecraft:cobblestone"),
        "deepslate": HarvestRule("minecraft:deepslate", 1, "wooden_pickaxe", "minecraft:cobbled_deepslate"),
        "cobbled_deepslate": HarvestRule("minecraft:cobbled_deepslate", 1, "wooden_pickaxe", "minecraft:cobbled_deepslate"),
        "coal_ore": HarvestRule("minecraft:coal_ore", 1, "wooden_pickaxe", "minecraft:coal", "Y in [48,136]"),
        "deepslate_coal_ore": HarvestRule("minecraft:deepslate_coal_ore", 1, "wooden_pickaxe", "minecraft:coal"),
        "furnace": HarvestRule("minecraft:furnace", 1, "wooden_pickaxe", "minecraft:furnace"),
        # Requires >= stone_pickaxe (tier 2)
        "iron_ore": HarvestRule("minecraft:iron_ore", 2, "stone_pickaxe", "minecraft:raw_iron", "Y in [-16,48], peak Y=16"),
        "deepslate_iron_ore": HarvestRule("minecraft:deepslate_iron_ore", 2, "stone_pickaxe", "minecraft:raw_iron"),
        "copper_ore": HarvestRule("minecraft:copper_ore", 2, "stone_pickaxe", "minecraft:raw_copper"),
        "deepslate_copper_ore": HarvestRule("minecraft:deepslate_copper_ore", 2, "stone_pickaxe", "minecraft:raw_copper"),
        "lapis_ore": HarvestRule("minecraft:lapis_ore", 2, "stone_pickaxe", "minecraft:lapis_lazuli"),
        "deepslate_lapis_ore": HarvestRule("minecraft:deepslate_lapis_ore", 2, "stone_pickaxe", "minecraft:lapis_lazuli"),
        # Requires >= iron_pickaxe (tier 3)
        "gold_ore": HarvestRule("minecraft:gold_ore", 3, "iron_pickaxe", "minecraft:raw_gold"),
        "deepslate_gold_ore": HarvestRule("minecraft:deepslate_gold_ore", 3, "iron_pickaxe", "minecraft:raw_gold"),
        "diamond_ore": HarvestRule("minecraft:diamond_ore", 3, "iron_pickaxe", "minecraft:diamond", "Y in [-59,-50], mine at Y=-53"),
        "deepslate_diamond_ore": HarvestRule("minecraft:deepslate_diamond_ore", 3, "iron_pickaxe", "minecraft:diamond", "Mine at Y=-53"),
        "redstone_ore": HarvestRule("minecraft:redstone_ore", 3, "iron_pickaxe", "minecraft:redstone"),
        "deepslate_redstone_ore": HarvestRule("minecraft:deepslate_redstone_ore", 3, "iron_pickaxe", "minecraft:redstone"),
        "emerald_ore": HarvestRule("minecraft:emerald_ore", 3, "iron_pickaxe", "minecraft:emerald"),
        "deepslate_emerald_ore": HarvestRule("minecraft:deepslate_emerald_ore", 3, "iron_pickaxe", "minecraft:emerald"),
        # Requires >= diamond_pickaxe (tier 4)
        "obsidian": HarvestRule("minecraft:obsidian", 4, "diamond_pickaxe", "minecraft:obsidian"),
        "crying_obsidian": HarvestRule("minecraft:crying_obsidian", 4, "diamond_pickaxe", "minecraft:crying_obsidian"),
        "ancient_debris": HarvestRule("minecraft:ancient_debris", 4, "diamond_pickaxe", "minecraft:ancient_debris"),
    }

    SMELTING_OUTPUTS: dict[str, str] = {
        "minecraft:raw_iron": "minecraft:iron_ingot",
        "minecraft:iron_ore": "minecraft:iron_ingot",
        "minecraft:deepslate_iron_ore": "minecraft:iron_ingot",
        "minecraft:raw_gold": "minecraft:gold_ingot",
        "minecraft:raw_copper": "minecraft:copper_ingot",
        "minecraft:beef": "minecraft:cooked_beef",
        "minecraft:porkchop": "minecraft:cooked_porkchop",
        "minecraft:mutton": "minecraft:cooked_mutton",
        "minecraft:chicken": "minecraft:cooked_chicken",
        "minecraft:rabbit": "minecraft:cooked_rabbit",
        "minecraft:cod": "minecraft:cooked_cod",
        "minecraft:salmon": "minecraft:cooked_salmon",
        "minecraft:potato": "minecraft:baked_potato",
        "minecraft:oak_log": "minecraft:charcoal",
        "minecraft:spruce_log": "minecraft:charcoal",
        "minecraft:birch_log": "minecraft:charcoal",
    }

    SMELTING_FUELS: dict[str, float] = {
        "minecraft:lava_bucket": 100.0,
        "minecraft:coal_block": 80.0,
        "minecraft:blaze_rod": 12.0,
        "minecraft:coal": 8.0,
        "minecraft:charcoal": 8.0,
        "minecraft:oak_log": 1.5,
        "minecraft:spruce_log": 1.5,
        "minecraft:birch_log": 1.5,
        "minecraft:oak_planks": 1.5,
        "minecraft:spruce_planks": 1.5,
        "minecraft:birch_planks": 1.5,
        "minecraft:stick": 0.5,
    }

    ORE_Y_LEVELS: dict[str, tuple[int, int, int]] = {
        "coal_ore": (48, 136, 96),
        "iron_ore": (-16, 48, 16),
        "diamond_ore": (-59, -50, -53),
    }

    @classmethod
    def required_pickaxe_tier(cls, block_id_or_pattern: str) -> int:
        short = block_id_or_pattern.split(":", 1)[-1].lower().strip("*")
        if short in cls.HARVEST_MATRIX:
            return cls.HARVEST_MATRIX[short].min_pickaxe_tier
        if short.endswith(("_log", "_stem", "_leaves", "_planks")):
            return 0
        if "obsidian" in short or "ancient_debris" in short:
            return 4
        if any(k in short for k in ("diamond_ore", "gold_ore", "redstone_ore", "emerald_ore")):
            return 3
        if any(k in short for k in ("iron_ore", "copper_ore", "lapis_ore")):
            return 2
        if any(k in short for k in ("stone", "cobblestone", "deepslate", "coal_ore", "furnace")):
            return 1
        return 0

    @classmethod
    def validate_harvest_tool(
        cls,
        block_id_or_pattern: str,
        snapshot: BotTelemetrySnapshot,
    ) -> tuple[bool, int, str]:
        """Returns (has_sufficient_tool, required_tier, required_tool_name)."""
        req_tier = cls.required_pickaxe_tier(block_id_or_pattern)
        if req_tier <= 0:
            return True, 0, "hand"
        owned_tier = snapshot.highest_tool_tier("_pickaxe")
        req_name = PICKAXE_TIER_NAMES.get(req_tier, "wooden_pickaxe")
        return owned_tier >= req_tier, req_tier, req_name

    @classmethod
    def filter_rules_for_task(cls, task: CurriculumNode) -> str:
        """Dynamically prunes mechanics rules to ONLY those matching the active task (saves ~800 tokens)."""
        rules: list[str] = []
        tid = task.id
        tier = task.tier

        for target_pat in task.target_items:
            short = target_pat.split(":", 1)[-1].lstrip("*_")
            rec = RecipeKnowledgeGraph.format_recipe_compact(target_pat)
            if rec:
                rules.append(rec)
            elif short == "raw_iron":
                rules.append("iron_ore requires >=stone_pickaxe (drops raw_iron; peak Y=16 in [-16,48])")
            elif short == "iron_ingot":
                rules.append("smelt raw_iron + coal/charcoal in placed furnace -> iron_ingot")
            elif short == "cobblestone":
                rules.append("stone requires >=wooden_pickaxe (hand yields 0 drops)")
            elif short == "diamond":
                rules.append("diamond_ore requires >=iron_pickaxe; dig stepped staircase(dY=-1,dX=+1) to Y=-53")
            elif short == "obsidian":
                rules.append("obsidian requires >=diamond_pickaxe; pour water_bucket near lava source")
            elif short == "water_bucket":
                rules.append(RecipeKnowledgeGraph.format_recipe_compact("bucket") + "; UseBucketAtAsync on water")

        if tier == 1 and not rules:
            rules.append("logs harvestable by hand; planks(x4)<-1x log[2x2]; stick(x4)<-2x planks[2x2]")
        elif "iron_tools_and_armor" in tid:
            rules.append(
                "iron_chestplate<-8x iron_ingot[table]; iron_leggings<-7x[table]; iron_boots<-4x[table]; iron_helmet<-5x[table]"
            )
        elif "cook_food" in tid or "furnace" in tid:
            rules.append("furnace<-8x cobblestone[table]; smelt raw meat + coal/charcoal(8x)/logs(1.5x); torch(x4)<-1x coal+1x stick[2x2]")
        elif "nether" in tid:
            rules.append("flint_and_steel<-1x iron_ingot+1x flint(mine gravel); build 4x5 obsidian frame & ignite")

        if task.requires_movement and tier >= 3:
            rules.append("SafeMining: NEVER dig straight down/up; use DigStaircaseToYAsync checking lava/water")

        compact_line = " | ".join(dict.fromkeys(r for r in rules if r))
        return compact_line[:360] if compact_line else "Follow precondition checks and equip best tool."
