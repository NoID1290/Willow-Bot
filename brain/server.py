#!/usr/bin/env python3
"""Loopback-only actor and bounded background strategist, using the standard library."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import hmac
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
import urllib.error
import urllib.request
import uuid

from skills import (
    SkillLibrary,
    dry_run,
    extract_description,
    normalize,
    parse,
    rank_skills,
    retrieve_few_shot_examples,
)
from experience import Experience, inventory
from memory import Memory
import modelprofile
from modelprofile import (
    DECISION_LIMITS,
    DEFAULT_ACTOR_TIMEOUT as ACTOR_TIMEOUT,
    DEFAULT_ACTOR_TOKENS as OUTPUT_TOKENS,
    DEFAULT_CONTEXT_CAP as CONTEXT,
    DEFAULT_MODEL as MODEL,
    DEFAULT_PLAN_MAX_AGE as PLAN_MAX_AGE,
    DEFAULT_PLAN_TIMEOUT as PLAN_TIMEOUT,
    DEFAULT_PROMPT_BYTES as PROMPT_BYTES,
    GAMEPLAY_LIMITS,
)
from multimodal import ConcurrentPipeline
from orchestrator import WillowBotOrchestrator
from hardware_opt.llm_client import InferencePriority
from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot
from rl_rewards.q_table import ContextStateHash

LOG = logging.getLogger("localai")
STEP = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": [
        "explore", "goto", "run", "sprint", "jump", "sneak", "pillar",
        "mine", "craft", "place", "eat", "attack", "pickup", "wait", "escape",
        "smelt", "chest", "sleep", "trade", "boat", "tunnel", "bridge", "shelter", "portal"]},
        "target": {"type": "string", "maxLength": DECISION_LIMITS["target"]},
        "x": {"type": "integer"},
        "y": {"type": "integer"}, "z": {"type": "integer"}},
    "required": ["action", "target", "x", "y", "z"], "additionalProperties": False,
}
ACTIONS = ["explore", "goto", "run", "sprint", "jump", "sneak", "pillar",
           "mine", "craft", "place", "eat", "attack", "pickup", "wait", "escape",
           "smelt", "chest", "sleep", "trade", "boat", "tunnel", "bridge", "shelter", "portal"]
TARGET_ACTIONS = ("craft", "mine", "pickup", "place", "attack", "pillar", "smelt", "chest", "trade")


class DecisionRejected(ValueError):
    """No valid current action was obtained; no action was executed."""


SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "target": {"type": "string", "maxLength": DECISION_LIMITS["target"]},
        "x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"},
        "goal": {"type": "string", "maxLength": DECISION_LIMITS["goal"]},
        "say": {"type": "string", "maxLength": DECISION_LIMITS["say"]},
        "memory": {"type": "string", "maxLength": DECISION_LIMITS["memory"]},
        "waypoint": {"type": "string", "maxLength": DECISION_LIMITS["waypoint"]},
        "plan": {"type": "array", "items": STEP, "maxItems": DECISION_LIMITS["plan"]},
    },
    "required": ["action", "target", "x", "y", "z", "goal", "say", "memory", "waypoint"],
    "additionalProperties": False,
}
SYSTEM = """You are Willow, an autonomous Minecraft survival player, not a chatbot.
ACTIVE TOOLS:
- active_tools.vision: seeing.looking_at (crosshair up to 12m), seeing.visible_entities (90° FOV with distance/bearing/LOS), seeing.retina (7x5 depth grid), seeing.visual_hazards (lava/hostiles).
- active_tools.hearing: listening.recent_sounds and listening.active_alerts (3D sound bearings/distances; evade CRITICAL creeper hisses or explosions immediately).
- active_tools.actions: explore, goto, run, sprint, jump, sneak, pillar, mine, craft, place, eat, attack, pickup, wait, escape, smelt, chest, sleep, trade, boat, tunnel, bridge, shelter, portal.
DECISION & PLAN CHAINING:
Choose an immediate action plus a plan of up to 8 sequential follow-up steps (action,target,x,y,z) that run WITHOUT another model call while preconditions hold (stopping on failure, danger, chat or changed surroundings). Do not duplicate the initial action in plan.
Always chain 2-6 steps when gathering or crafting (e.g. mine log -> pickup log -> mine next log -> craft planks).
SURVIVAL RULES:
1. Priority: survive, food (if food < 14 and holding food, eat now), logs, planks, crafting_table, sticks, wooden_pickaxe, cobblestone, stone tools, furnace/smelt iron/food, shelter, sleep at night, exploration. Pursue saved_goal; update goal when complete or blocked.
2. Trees in sight: when logs appear in nearby_blocks or seeing.looking_at, mine them directly (with exact x/y/z or 0,0,0). NEVER repeat goto or circle when a tree is visible and wood is needed.
3. No duplicates: milestones.owned_tools and inventory show completed tools/tables; owned tools are finished milestones, not missing prerequisites. NEVER craft duplicate pickaxes, axes, swords, shovels, hoes, furnaces or crafting_tables already owned, or unrequested redstone/decorative items (levers, buttons, slabs, stairs, walls). Follow crafting.next_step to bootstrap tools; place an inventory crafting_table at crafting.placement_sites for 3x3 recipes. Basics: 1 log -> 4 planks; a crafting_table is 4 planks in the 2x2 grid (no table needed for planks, sticks or the table itself); a wooden_pickaxe is 3 planks + 2 sticks and needs a placed table.
4. Caves, pits, tunnels & bridges: blocked_actions failed twice here—change target, move to terrain.exits, or alter conditions. If stuck in a cave/hole/pit, use pillar (places a block under your feet to get higher), tunnel (dig 1x2 path forward), bridge (place horizontal path across gaps/water), or escape (may take fall damage or break the block supporting you).
5. Stockpile & full inventory: stop mining bulk stone/dirt once milestones.enough_building_blocks lists it (those items are ENOUGH; never gather more) or milestones.inventory_free_slots == 0. Use chest to store excess items or retrieve supplies. Collect needed item drops with pickup only when entities contains item drops.
6. Action semantics: mine (block ID + observed x/y/z or 0,0,0; logs by hand, stone needs pickaxe; blocks with reachable=false have no standing spot—tunnel toward them instead); craft (exact ID from craftable); smelt (item ID from smeltable); chest (item ID or ""); sleep (use/place bed at night); trade (villager trade item or ""); boat (use/place boat on water); shelter (build protective walls/roof); portal (light/enter portal); place (ID from placeable at exact crafting.placement_sites x/y/z); goto (walk to x/y/z or head toward far global waypoints); run/sprint (fast move, needs food > 6); tunnel (dig a 1x2 passage toward x/y/z; a y below your position digs a safe staircase DOWN—the normal way to reach stone when none is reachable); attack (numeric non-player entity ID). Do not put explore or wait in plan: they need a fresh look each time.
7. Chat & permanent memory: if chat is nonempty, ALWAYS reply briefly in say and adopt reasonable requests; otherwise state brief sensory/action observations in say. Save new durable facts in memory (shown in my_notes) and named locations (home, base, table, cave, iron) in waypoint. Return JSON only."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "maxLength": DECISION_LIMITS["goal"]},
        "reason": {"type": "string", "maxLength": DECISION_LIMITS["memory"]},
        "code": {"type": "string", "maxLength": DECISION_LIMITS["code"]},
    },
    "required": ["goal", "reason", "code"], "additionalProperties": False,
}
PLAN_SYSTEM = """You are Willow's background strategist and skill author, NOT its immediate actor.
Study the snapshot, lookup failures, and skill_programs. Propose a short goal, reason, and reusable Python skill.
1. Check milestones.owned_tools and inventory first: NEVER propose crafting tools or crafting_tables already owned, or gathering blocks listed in milestones.enough_building_blocks.
2. Table & furnace bootstrap: logs -> planks -> crafting_table -> place_table() -> sticks -> pickaxe -> cobblestone -> furnace -> place_furnace() -> smelt(). Built-in foundation() already handles crafting.next_step, smelting, and wood drops; complement it rather than duplicating it.
3. Restricted Python syntax: exactly `def skill():` with local variables (<=16), if/elif/else, return, comparisons, and/or/not, integer +/-, pass, docstrings, and literals (<=4096 UTF-8 bytes). FORBIDDEN: loops (for/while), imports, other def/lambda, lists, dicts, indexing, attributes, f-strings, `in`, print(), len(), markdown fences. Called once per turn to return ONE action or None.
4. Read helpers: count("minecraft:item") or group tags count("#logs"), count("#planks"), count("#pickaxes"), count("#raw_ores"), count("#ingots"), count("#beds"), count("#boats") (never "minecraft:logs"/"minecraft:planks"/"minecraft:sticks"); can_craft("minecraft:recipe"), can_smelt("minecraft:item"), can_sleep(), can_trade("minecraft:item"), can_pillar(), can_pillar_up(), health(), food(), free_slots(), is_night(), in_water(), table_reachable(), furnace_reachable(), chest_reachable(), bed_reachable(), stuck(), nearby("minecraft:block"), drops("minecraft:item") (ground items only), failures(action,target), successes(action,target).
5. Action helpers: foundation(), place_table(), place_furnace(), place_chest(), smelt("minecraft:raw_iron"), chest("minecraft:cobblestone" or ""), sleep(), trade("minecraft:emerald" or ""), boat(), tunnel(), bridge(), shelter(), portal(), pillar("minecraft:dirt" or ""), craft("minecraft:recipe") (always guard with can_craft), mine("minecraft:block") (guard with nearby and cap with count, e.g. <64 cobblestone), pickup_item("minecraft:item"), pickup(), explore(), escape().
Return JSON with goal, reason, and code (empty string if no skill needed)."""
SYSTEM_COMPACT = """You are Willow, an autonomous Minecraft survival player.
Use active_tools (vision looking_at/visible_entities/hazards, hearing alerts, actions). Choose action + up to 8 plan steps.
Priority: survive, eat if food < 14, logs, planks, crafting_table, sticks, wooden_pickaxe, cobblestone, stone tools, smelt, shelter, sleep.
Mine visible logs directly; never repeat goto or circle trees. Never craft duplicate tools/tables in milestones.owned_tools or gather items in milestones.enough_building_blocks.
Use pillar, tunnel, bridge, or escape when stuck. Reply in say when chat is nonempty. Return JSON only."""
PLAN_SYSTEM_COMPACT = """You are Willow's background strategist and skill author.
Propose goal, reason, and restricted Python `def skill():` (only local vars, if/elif/else, comparisons, return; no loops, imports, lists, indexing, attributes).
Helpers: count(item/#logs/#planks/#pickaxes/#raw_ores/#ingots/#beds/#boats), nearby(block), drops(item), can_craft(recipe), can_smelt(item), can_sleep(), can_trade(item), can_pillar_up(), table_reachable(), furnace_reachable(), chest_reachable(), bed_reachable(), free_slots(), is_night(), in_water(), stuck(), failures(a,t), foundation(), mine(block), craft(recipe), smelt(item), chest(item), sleep(), trade(item), boat(), tunnel(), bridge(), shelter(), portal(), place_table(), place_furnace(), place_chest(), pillar(block), pickup_item(item), pickup(), explore(), escape().
Never craft duplicate tools or gather milestones.enough_building_blocks. Return JSON with goal, reason, code."""


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def rate(count, nanoseconds):
    if not isinstance(count, int) or not isinstance(nanoseconds, int) or nanoseconds <= 0:
        return "-"
    return f"{count / (nanoseconds / 1e9):.1f}"


def validate_decision(value):
    if isinstance(value, dict):
        value.setdefault("waypoint", "")
    if (not isinstance(value, dict) or not set(SCHEMA["required"]).issubset(value)
            or set(value) - set(SCHEMA["properties"])):
        raise ValueError("Model response must contain the decision fields and optional plan")
    if value["action"] not in ACTIONS:
        raise ValueError("Unknown action")
    for key in ("x", "y", "z"):
        if type(value[key]) is not int or abs(value[key]) > 30_000_000:
            raise ValueError(f"Invalid coordinate {key}")
    for key in ("goal", "say", "memory", "waypoint"):
        limit = DECISION_LIMITS[key]
        # Free text is advisory; trim small models' long prose instead of discarding a valid action.
        if isinstance(value[key], str):
            value[key] = re.sub(r"[\x00-\x1f]+", " ", value[key]).strip()[:limit]
    for key in ("target", "goal", "say", "memory", "waypoint"):
        limit = DECISION_LIMITS[key]
        if not isinstance(value[key], str) or len(value[key]) > limit:
            raise ValueError(f"Invalid or oversized {key}")
        if any(ord(c) < 32 for c in value[key]):
            raise ValueError(f"Control character in {key}")
    if isinstance(value["target"], str):
        target = value["target"].strip()
        if value["action"] in ("mine", "craft", "place", "pillar", "smelt", "chest", "trade") and target:
            match = re.search(r"[a-zA-Z0-9_.-]+:[a-zA-Z0-9_./-]+", target)
            if match:
                target = match.group(0).lower().rstrip(".,;:!?")
            elif re.fullmatch(r"[a-zA-Z0-9_.-]+", target):
                target = f"minecraft:{target.lower()}"
        elif value["action"] == "attack":
            match = re.search(r"[0-9]{1,10}", target)
            if match:
                target = match.group(0)
        value["target"] = target
    if value["action"] in ("mine", "craft", "place", "smelt") and not re.fullmatch(
            r"[a-z0-9_.-]+:[a-z0-9_./-]+", value["target"]):
        raise ValueError("Resource target must be a namespaced ID without coordinates")
    if value["action"] in ("pillar", "chest", "trade") and value["target"] and not re.fullmatch(
            r"[a-z0-9_.-]+:[a-z0-9_./-]+", value["target"]):
        raise ValueError("Optional target must be a namespaced ID or empty")
    if value["action"] in ("jump", "sneak", "sleep", "boat", "tunnel", "bridge", "shelter", "portal") and value["target"]:
        value["target"] = ""
    if value["action"] == "attack" and not re.fullmatch(r"[0-9]{1,10}", value["target"]):
        raise ValueError("Attack target must be a numeric entity ID")
    plan = value.get("plan", [])
    if not isinstance(plan, list) or len(plan) > DECISION_LIMITS["plan"]:
        raise ValueError("Plan must contain at most eight steps")
    valid_plan = []
    for step in plan:
        if not isinstance(step, dict) or set(step) != set(STEP["required"]):
            raise ValueError("Invalid follow-up step")
        step_clean = {k: step[k] for k in STEP["required"]}
        try:
            validated_step = validate_decision(dict(step_clean, goal=value["goal"], say="", memory=""))
        except (ValueError, TypeError, KeyError):
            break
        # Undirected explore/wait steps queued blind become a random walk; they need a fresh observation.
        if validated_step["action"] == "wait" or (validated_step["action"] == "explore"
                                                   and [validated_step[k] for k in "xyz"] == [0, 0, 0]):
            continue
        valid_plan.append({k: validated_step[k] for k in STEP["required"]})
    value["plan"] = valid_plan
    return value


def allowed_actions(observation):
    available = list(ACTIONS)
    last = observation.get("last_result", {})
    if (observation.get("diagnosis", {}).get("stuck")
            and not (last.get("action") == "escape" and last.get("status") == "failed")
            and not blocked(observation, "escape", "")):
        return ["escape"]
    if not observation.get("craftable"):
        available.remove("craft")
    if not observation.get("smeltable"):
        available.remove("smelt")
    if "entities" in observation:
        entities = observation["entities"]
        if not any(isinstance(entity, dict) and entity.get("type") == "minecraft:item" for entity in entities):
            available.remove("pickup")
        if not any(isinstance(entity, dict) and entity.get("type") not in ("minecraft:item", "minecraft:player")
                   for entity in entities):
            available.remove("attack")
    if "nearby_blocks" in observation and not observation["nearby_blocks"]:
        available.remove("mine")
    if "inventory" in observation and not observation["inventory"]:
        available.remove("eat")
        available.remove("place")
        available.remove("pillar")

    counts = inventory(observation)
    nearby_ids = {b.get("id") for b in observation.get("nearby_blocks", []) if isinstance(b, dict)}
    crafting_info = observation.get("crafting", {})
    sites = crafting_info.get("placement_sites", [])
    chat = " ".join(observation.get("chat", [])).lower()

    if not (crafting_info.get("chest_reachable")
            or any(b in ("minecraft:chest", "minecraft:barrel", "minecraft:trapped_chest") for b in nearby_ids)
            or ((counts.get("minecraft:chest", 0) > 0 or counts.get("minecraft:barrel", 0) > 0) and sites)):
        available.remove("chest")

    has_bed_inv = any(k.endswith("_bed") and v > 0 for k, v in counts.items())
    has_bed_near = crafting_info.get("bed_reachable") or any(isinstance(b, str) and b.endswith("_bed") for b in nearby_ids)
    if not (has_bed_near or (has_bed_inv and sites)):
        available.remove("sleep")

    has_villager = any(
        isinstance(e, dict) and e.get("type") in ("minecraft:villager", "minecraft:wandering_trader")
        for e in observation.get("entities", [])
    )
    if not (observation.get("trades") or has_villager):
        available.remove("trade")

    has_boat_inv = any(k.endswith(("_boat", "_raft")) and v > 0 for k, v in counts.items())
    has_boat_ent = any(
        isinstance(e, dict) and isinstance(e.get("type"), str) and ("boat" in e["type"] or "raft" in e["type"])
        for e in observation.get("entities", [])
    )
    if not ((observation.get("in_water") or "minecraft:water" in nearby_ids) and (has_boat_inv or has_boat_ent)):
        available.remove("boat")

    exits = observation.get("terrain", {}).get("exits", [])
    has_blocked_exit = any(isinstance(c, dict) and not c.get("standable", True) for c in exits)
    pos = observation.get("position")
    underground = isinstance(pos, list) and len(pos) == 3 and isinstance(pos[1], (int, float)) and pos[1] < 60
    has_pickaxe = any(k.endswith("_pickaxe") and v > 0 for k, v in counts.items())
    if not (nearby_ids and (observation.get("diagnosis", {}).get("stuck") or has_blocked_exit or underground
                            or has_pickaxe or "tunnel" in chat or "dig" in chat)):
        available.remove("tunnel")

    has_placeable = any(counts.get(item, 0) > 0 for item in observation.get("placeable", []))
    if not (has_placeable and (observation.get("in_water") or "minecraft:water" in nearby_ids
                               or "minecraft:lava" in nearby_ids or has_blocked_exit or "bridge" in chat)):
        available.remove("bridge")

    time_val = observation.get("time", 0)
    is_night = isinstance(time_val, (int, float)) and 12000 <= (int(time_val) % 24000) <= 23500
    if not (has_placeable and sites and (is_night or "shelter" in chat or "house" in chat)):
        available.remove("shelter")

    if not ("minecraft:nether_portal" in nearby_ids or "minecraft:obsidian" in nearby_ids
            or (("portal" in chat or "nether" in chat) and (counts.get("minecraft:obsidian", 0) >= 10
                                                            or counts.get("minecraft:flint_and_steel", 0) > 0))):
        available.remove("portal")

    for action in TARGET_ACTIONS:
        if action in available and not action_targets(observation, action):
            available.remove(action)
    for action in ("escape", "explore", "wait", "goto", "run", "sprint", "jump", "sneak", "eat", "pillar",
                   "sleep", "boat", "tunnel", "bridge", "shelter", "portal"):
        if action in available and blocked(observation, action, ""):
            available.remove(action)
    return available


def blocked(observation, action, target):
    return any(row["action"] == action and (row["target"] == target or not row["target"])
               for row in observation.get("blocked_actions", []))


MOVEMENT_ACTIONS = ("goto", "run", "sprint")
COORD_TARGET = re.compile(r"-?\d+,-?\d+,-?\d+")


def stored_target(decision):
    """Target recorded in the decisions table: movement actions keep their destination so repeated
    failures block only that spot, not all movement."""
    target = decision.get("target", "")
    if target or decision.get("action") not in MOVEMENT_ACTIONS:
        return target
    if not all(k in decision and decision[k] is not None for k in "xyz"):
        return ""
    coords = [decision[k] for k in "xyz"]
    return f"{coords[0]},{coords[1]},{coords[2]}"


def memory_target(action, target):
    """Permanent knowledge is per action kind for movement, not per coordinate."""
    if action in MOVEMENT_ACTIONS and isinstance(target, str) and COORD_TARGET.fullmatch(target):
        return ""
    return target


TOOL_TIERS = {
    "wooden": 1,
    "golden": 1,
    "stone": 2,
    "iron": 3,
    "diamond": 4,
    "netherite": 5,
}

TOOL_TYPES = ("pickaxe", "axe", "shovel", "sword", "hoe")

JUNK_UNREQUESTED_ITEMS = {
    "minecraft:lever",
}


def is_tool_redundant(target, counts, table_reachable=False):
    name = target.split(":")[-1]
    for tool in TOOL_TYPES:
        for tier, level in TOOL_TIERS.items():
            if name == f"{tier}_{tool}":
                highest_owned = 0
                for owned_item, count in counts.items():
                    if count > 0:
                        owned_name = owned_item.split(":")[-1]
                        for o_tier, o_level in TOOL_TIERS.items():
                            if owned_name == f"{o_tier}_{tool}":
                                highest_owned = max(highest_owned, o_level)
                if highest_owned >= level:
                    return True
                return False

    if name == "crafting_table":
        return counts.get("minecraft:crafting_table", 0) > 0 or table_reachable

    if name == "furnace":
        return counts.get("minecraft:furnace", 0) > 0

    if target in JUNK_UNREQUESTED_ITEMS or name.endswith((
            "_button", "_pressure_plate", "_stairs", "_slab", "_wall",
            "_wood", "_hyphae", "_sign", "_hanging_sign", "_fence", "_fence_gate",
            "_door", "_trapdoor", "_carpet", "_banner", "_dye")):
        return True

    if name == "stick" and counts.get("minecraft:stick", 0) >= 8:
        return True

    planks = sum(c for k, c in counts.items() if k.endswith("_planks"))
    pickaxes = sum(c for k, c in counts.items() if k.endswith("_pickaxe"))
    if name.endswith("_planks") and (planks >= 32 or (pickaxes > 0 and planks >= 8)):
        return True

    return False


# Common building blocks: mined block -> item it drops. Ores, logs and rare blocks are never capped.
BULK_DROPS = {
    "minecraft:stone": "minecraft:cobblestone", "minecraft:cobblestone": "minecraft:cobblestone",
    "minecraft:deepslate": "minecraft:cobbled_deepslate",
    "minecraft:cobbled_deepslate": "minecraft:cobbled_deepslate",
    "minecraft:grass_block": "minecraft:dirt", "minecraft:dirt": "minecraft:dirt",
    "minecraft:coarse_dirt": "minecraft:coarse_dirt", "minecraft:sand": "minecraft:sand",
    "minecraft:red_sand": "minecraft:red_sand", "minecraft:gravel": "minecraft:gravel",
    "minecraft:granite": "minecraft:granite", "minecraft:diorite": "minecraft:diorite",
    "minecraft:andesite": "minecraft:andesite", "minecraft:tuff": "minecraft:tuff",
    "minecraft:calcite": "minecraft:calcite", "minecraft:sandstone": "minecraft:sandstone",
    "minecraft:netherrack": "minecraft:netherrack", "minecraft:blackstone": "minecraft:blackstone",
}
BULK_ITEMS = set(BULK_DROPS.values())
STOCKPILE_ITEM_LIMIT = GAMEPLAY_LIMITS["stockpile_plank_equiv"]
STOCKPILE_TOTAL_LIMIT = GAMEPLAY_LIMITS["stockpile_cobble"]
TOOL_SUFFIXES = ("_pickaxe", "_axe", "_shovel", "_sword", "_hoe", "_helmet", "_chestplate",
                 "_leggings", "_boots", "_bucket", "bow", "shield", "flint_and_steel", "shears")


def free_slots(observation):
    """Empty main-inventory slots, reported by the mod or conservatively estimated."""
    value = observation.get("free_slots")
    if type(value) is int:
        return value
    if "inventory" not in observation:
        return None
    used = sum(count if item.endswith(TOOL_SUFFIXES) else -(-count // 64)
               for item, count in inventory(observation).items())
    return max(0, 36 - used)


def exit_blocks(observation):
    return {cell.get(key) for cell in observation.get("terrain", {}).get("exits", [])
            if isinstance(cell, dict) for key in ("feet", "head")}


def exit_dig_cells(observation, block):
    """Coordinates where mining `block` opens one of the surrounding exit cells (feet or head block)."""
    cells = []
    for cell in observation.get("terrain", {}).get("exits", []):
        if not isinstance(cell, dict) or not (isinstance(cell.get("at"), list) and len(cell["at"]) == 3):
            continue
        x, y, z = cell["at"]
        if cell.get("feet") == block:
            cells.append([x, y, z])
        if cell.get("head") == block:
            cells.append([x, y + 1, z])
    return cells


def enclosed(observation):
    """No exit cell can be walked into, so digging a wall is the only way out."""
    return bool(observation.get("diagnosis", {}).get("stuck")) or not any(
        isinstance(cell, dict) and cell.get("standable")
        for cell in observation.get("terrain", {}).get("exits", []))


def stockpile_full(observation, block):
    """Reason why mining this block gains nothing useful, or an empty string."""
    counts = inventory(observation)
    chat = " ".join(observation.get("chat", [])).lower()
    if isinstance(block, str) and block.endswith(("_log", "_stem")) and not any(
            w in chat for w in ("log", "wood", "tree", "plank")):
        logs = sum(c for k, c in counts.items() if k.endswith(("_log", "_stem")))
        planks = sum(c for k, c in counts.items() if k.endswith("_planks"))
        pickaxes = sum(c for k, c in counts.items() if k.endswith("_pickaxe"))
        if logs >= 16 or (pickaxes > 0 and logs >= 8 and planks >= 8):
            return (f"Already carrying {logs} logs and {planks} planks; that is enough wood. "
                    "Use your pickaxe to mine stone/coal/iron or explore instead of chopping more trees")
    drop = BULK_DROPS.get(block)
    if not drop:
        return ""
    limits = (observation.get("_gameplay_limits") if isinstance(observation, dict) else None) or GAMEPLAY_LIMITS
    item_limit = int(limits.get("stockpile_plank_equiv", STOCKPILE_ITEM_LIMIT))
    total_limit = int(limits.get("stockpile_cobble", STOCKPILE_TOTAL_LIMIT))
    have = counts.get(drop, 0)
    total = sum(count for item, count in counts.items() if item in BULK_ITEMS)
    if have >= item_limit or total >= total_limit:
        return (f"Already carrying {have} {drop} ({total} building blocks); that is enough. "
                "Build with it (place) or pursue wood, food, coal, iron or exploration")
    if free_slots(observation) == 0 and have % 64 == 0:
        return f"Inventory is full; mined {drop} cannot be picked up. Place or use blocks, or pursue another goal"
    return ""


def craft_targets(observation):
    targets = list(dict.fromkeys(observation.get("craftable", [])))
    chat = " ".join(observation.get("chat", [])).lower()
    requested = [target for target in targets if target in chat
                 or target.split(":")[-1].replace("_", " ") in chat]
    next_target = observation.get("crafting", {}).get("next_step", {}).get("target")
    counts = inventory(observation)
    table_reachable = observation.get("crafting", {}).get("table_reachable", False)

    unrequested = []
    for target in targets:
        if target in requested:
            continue
        if target == next_target:
            unrequested.append(target)
            continue
        if not is_tool_redundant(target, counts, table_reachable):
            unrequested.append(target)

    targets = requested + unrequested
    if next_target in targets:
        targets.remove(next_target)
        targets.insert(0, next_target)
    return [target for target in targets if not blocked(observation, "craft", target)][:32]


NON_RESOURCE_FOLIAGE = {
    "minecraft:leaf_litter", "minecraft:wildflowers", "minecraft:pink_petals",
    "minecraft:bush", "minecraft:firefly_bush", "minecraft:short_dry_grass",
    "minecraft:tall_dry_grass", "minecraft:short_grass", "minecraft:tall_grass",
    "minecraft:fern", "minecraft:large_fern", "minecraft:dead_bush",
    "minecraft:seagrass", "minecraft:tall_seagrass",
}

JUNK_DROPS = {
    "minecraft:rotten_flesh", "minecraft:spider_eye", "minecraft:poisonous_potato",
    "minecraft:pufferfish", "minecraft:leaf_litter",
}


def action_targets(observation, action):
    chat = " ".join(observation.get("chat", [])).lower()
    if action == "craft":
        return craft_targets(observation)
    if action == "smelt":
        targets = [item for item in observation.get("smeltable", []) if isinstance(item, str)]
        return [target for target in dict.fromkeys(targets) if not blocked(observation, "smelt", target)]
    if action == "chest":
        owned = inventory(observation)
        targets = [item for item, cnt in owned.items() if cnt > 0]
        targets.append("")
        return [target for target in dict.fromkeys(targets) if not blocked(observation, "chest", target)]
    if action == "trade":
        targets = [item for item in observation.get("trades", []) if isinstance(item, str)]
        targets.append("")
        return [target for target in dict.fromkeys(targets) if not blocked(observation, "trade", target)]
    if action == "mine":
        exits = exit_blocks(observation)
        stuck = bool(observation.get("diagnosis", {}).get("stuck"))
        leaves_requested = any(w in chat for w in ("leaves", "leaf", "sapling", "apple"))
        # Walls of a stockpiled block (cave stone, dirt) are only worth digging when there is no way to walk out.
        dig_out = enclosed(observation)
        # Blocks the game marks unreachable (e.g. a lake bed) cannot be mined from any standing spot.
        raw_blocks = [block["id"] for block in observation.get("nearby_blocks", []) if isinstance(block, dict)
                      and ((block["id"] in exits and (dig_out or not stockpile_full(observation, block["id"])))
                           or (block.get("reachable", True) and not stockpile_full(observation, block["id"])))]
        filtered = [
            b for b in raw_blocks
            if b in exits or (
                b not in NON_RESOURCE_FOLIAGE
                and (not b.endswith("_leaves") or stuck or leaves_requested)
            ) or b in chat or b.split(":")[-1].replace("_", " ") in chat
        ]
        targets = filtered if filtered else [b for b in raw_blocks if not b.endswith("_leaves") or stuck or leaves_requested]
        # Prioritize blocks in direct line of sight / crosshairs
        looking_id = (observation.get("sight", {}).get("looking_at", {}).get("id")
                      or observation.get("seeing", {}).get("looking_at", {}).get("id"))
        if looking_id and looking_id in targets:
            targets.remove(looking_id)
            targets.insert(0, looking_id)
        if chat:
            requested = [t for t in targets if t in chat or t.split(":")[-1].replace("_", " ") in chat
                         or (("wood" in chat or "log" in chat) and "log" in t)
                         or (("stone" in chat or "cobble" in chat) and ("stone" in t or "cobble" in t))]
            unrequested = [t for t in targets if t not in requested]
            targets = requested + unrequested
    elif action == "place":
        if not observation.get("crafting", {}).get("placement_sites"):
            return []
        owned = inventory(observation)
        targets = [item for item in observation.get("placeable", []) if owned.get(item, 0) > 0]
    elif action == "pillar":
        owned = inventory(observation)
        targets = [item for item in observation.get("placeable", []) if owned.get(item, 0) > 0]
        if targets:
            targets = list(targets) + [""]
    elif action == "attack":
        targets = [str(entity["id"]) for entity in observation.get("entities", [])
                   if isinstance(entity, dict) and "id" in entity
                   and entity.get("type") not in ("minecraft:item", "minecraft:player")]
        # Prioritize hostiles seen in direct line of sight
        seen = [str(e.get("id")) for e in (observation.get("sight", {}).get("visible_entities", [])
                                           or observation.get("seeing", {}).get("visible_entities", []))
                if isinstance(e, dict) and e.get("hostile")]
        if seen:
            prioritized = [t for t in targets if t in seen]
            other = [t for t in targets if t not in seen]
            targets = prioritized + other
    else:
        owned = inventory(observation)
        slots = free_slots(observation)
        raw_items = [entity.get("item", "") for entity in observation.get("entities", [])
                     if isinstance(entity, dict) and entity.get("type") == "minecraft:item"]
        targets = []
        for item in raw_items:
            if item and item in JUNK_DROPS and item not in chat and item.split(":")[-1].replace("_", " ") not in chat:
                continue
            if item and item.endswith("_sapling") and "sapling" not in chat and (
                    owned.get(item, 0) >= 4 or any(k.endswith("_pickaxe") and v > 0 for k, v in owned.items())):
                continue
            if item and item.endswith(("_log", "_stem")) and stockpile_full(observation, item):
                continue
            if item and slots == 0:
                have = owned.get(item, 0)
                if item.endswith(TOOL_SUFFIXES) or have == 0 or have % 64 == 0:
                    continue
            targets.append(item)
        # Prioritize visible item drops
        seen_items = [e.get("item") for e in (observation.get("sight", {}).get("visible_entities", [])
                                              or observation.get("seeing", {}).get("visible_entities", []))
                      if isinstance(e, dict) and e.get("item")]
        if seen_items:
            prioritized = [t for t in targets if t in seen_items]
            other = [t for t in targets if t not in seen_items]
            targets = prioritized + other
    return [target for target in dict.fromkeys(targets) if not blocked(observation, action, target)]


def action_schema(observation):
    schema = json.loads(compact(SCHEMA))
    available = allowed_actions(observation)
    schema["properties"]["action"]["enum"] = available
    branches = []
    for action in TARGET_ACTIONS:
        if action not in available:
            continue
        branch = json.loads(compact(SCHEMA))
        branch["properties"]["action"]["enum"] = [action]
        branch["properties"]["target"]["enum"] = action_targets(observation, action)
        if action == "place":
            for site in observation["crafting"]["placement_sites"][:4]:
                positioned = json.loads(compact(branch))
                for key, coordinate in zip(("x", "y", "z"), site):
                    positioned["properties"][key]["enum"] = [coordinate]
                branches.append(positioned)
        else:
            branches.append(branch)
    schema["properties"]["action"]["enum"] = [action for action in available if action not in TARGET_ACTIONS]
    if schema["properties"]["action"]["enum"]:
        schema["properties"]["target"]["enum"] = [""]
        branches.append(schema)
    if not branches:
        raise DecisionRejected("No unblocked actions are currently available; waiting for the situation or cooldowns to change")
    return branches[0] if len(branches) == 1 else {"oneOf": branches}


def validate_action(decision, observation):
    validate_decision(decision)
    here = observation.get("position")
    limits = (observation.get("_gameplay_limits") if isinstance(observation, dict) else None) or GAMEPLAY_LIMITS
    goto_range = int(limits.get("goto_max_range", GAMEPLAY_LIMITS["goto_max_range"]))
    goto_min = int(limits.get("goto_min_distance", GAMEPLAY_LIMITS["goto_min_distance"]))
    has_coords = all(k in decision and decision[k] is not None for k in "xyz")
    dest = [decision[k] for k in "xyz"] if has_coords else None
    if decision["action"] == "goto" and dest is not None:
        if (isinstance(here, list) and len(here) == 3 and all(isinstance(v, (int, float)) for v in here)
                and max(abs(dest[i] - here[i]) for i in range(3)) < goto_min):
            raise ValueError("Already at or adjacent to goto destination; choose mine, place, pickup or explore")
        coord_csv = f"{dest[0]},{dest[1]},{dest[2]}"
        coord_space = f"{dest[0]} {dest[1]} {dest[2]}"
        coord_bracket = f"[{dest[0]}, {dest[1]}, {dest[2]}]"
        # recent_actions is most-recent-first (see Brain.perceive); count the streak from the newest entry.
        recent_actions = (observation.get("recent_actions") or [])[:4]
        consecutive_gotos = 0
        for recent in recent_actions:
            r_act = recent.get("action") if isinstance(recent, dict) else ("goto" if isinstance(recent, str) and "goto" in recent else "")
            if r_act == "goto":
                consecutive_gotos += 1
            else:
                break
        if consecutive_gotos >= 2:
            raise ValueError("Already executed 2 consecutive goto steps; choose mine, place, craft, pickup or explore instead of circling")
        coord_short = f"{dest[0]}, {dest[1]}, {dest[2]}"  # BlockPos.toShortString()
        for recent in recent_actions:
            if isinstance(recent, dict) and recent.get("action") == "goto":
                if (recent.get("at") == dest or [recent.get("x"), recent.get("y"), recent.get("z")] == dest
                        or recent.get("target") == coord_csv):
                    raise ValueError("Repeated goto to the same coordinates; choose mine, place, craft or explore")
                detail = str((recent.get("result") or {}).get("detail", ""))
                if coord_csv in detail or coord_space in detail or coord_bracket in detail or coord_short in detail:
                    raise ValueError("Repeated goto to the same coordinates; choose mine, place, craft or explore")
            elif isinstance(recent, str) and "goto" in recent and (
                coord_csv in recent or coord_space in recent or coord_bracket in recent
            ):
                raise ValueError("Repeated goto to the same coordinates; choose mine, place, craft or explore")
    if (decision["action"] in ("goto", "run", "sprint") and isinstance(here, list) and len(here) == 3
            and all(isinstance(v, (int, float)) for v in here)
            and max(abs(decision[k] - here[i]) for i, k in enumerate("xyz")) > goto_range
            and "explore" in allowed_actions(observation)):
        known_coords = {
            tuple(item["at"])
            for key in ("waypoints", "remembered_resources")
            for item in (observation.get(key) or [])
            if isinstance(item, dict) and isinstance(item.get("at"), list) and len(item["at"]) == 3
        }
        # Guard against zero-axis coordinate hallucinations (e.g. x=0 when here[0]=-625, or [0,99,0])
        if tuple(dest) not in known_coords and (
            (decision["x"] == 0 and abs(here[0]) > 64) or (decision["z"] == 0 and abs(here[2]) > 64)
        ):
            LOG.info("Clearing zero-axis coordinate hallucination %s near %s to local explore", dest, here)
            decision.update(action="explore", target="", x=0, y=0, z=0)
        else:
            # Far destinations (waypoints, remembered places) become a walk heading toward them.
            LOG.info("%s %s is outside the %d-block range; heading toward it",
                     decision["action"], dest, goto_range)
            decision.update(action="explore", target="")
    if decision["action"] == "pillar":
        if decision["target"] and decision["target"] not in action_targets(observation, "pillar"):
            raise ValueError("Pillar target is not an owned placeable block")
    if decision["action"] == "smelt":
        if decision["target"] not in action_targets(observation, "smelt"):
            raise ValueError("Smelt target is not a currently smeltable item")
    if decision["action"] == "chest":
        if decision["target"] and decision["target"] not in action_targets(observation, "chest"):
            raise ValueError("Chest target is not an owned unblocked item")
    if decision["action"] == "trade":
        if decision["target"] and decision["target"] not in action_targets(observation, "trade"):
            raise ValueError("Trade target is not an available trade item")
    if decision["action"] == "mine":
        reason = stockpile_full(observation, decision["target"])
        position = [decision[key] for key in ("x", "y", "z")]
        walls = exit_dig_cells(observation, decision["target"])
        if reason:
            if not walls or decision["target"] not in action_targets(observation, "mine"):
                raise ValueError(reason)
            if position not in walls:
                # The schema offers a stockpiled block only as the wall of an enclosed spot; dig that wall
                # rather than the floor or ceiling the model happens to be looking at.
                wall = next((c for c in walls if c[0] == position[0] and c[2] == position[2]), walls[0])
                LOG.info("Mine %s at %s redirected to the exit wall at %s", decision["target"], position, wall)
                decision.update(x=wall[0], y=wall[1], z=wall[2])
                position = wall
        digging_exit = position in walls
        chat = " ".join(observation.get("chat", [])).lower()
        if (decision["target"].endswith("_leaves") and not digging_exit
                and not observation.get("diagnosis", {}).get("stuck")
                and not any(w in chat for w in ("leaves", "leaf", "sapling", "apple"))):
            raise ValueError("Do not mine tree leaves; mine logs, stone, or ores, or explore instead")
    if decision["action"] not in allowed_actions(observation):
        raise ValueError("Selected an unavailable action")
    if blocked(observation, decision["action"], stored_target(decision)):
        raise ValueError("This action failed repeatedly here; change target or position first")
    if decision["action"] == "craft":
        if decision["target"] not in observation.get("craftable", []):
            raise ValueError("Craft target is not an available recipe")
        if decision["target"] not in craft_targets(observation):
            raise ValueError("Craft target is already owned or duplicate; do not craft duplicate tools")
    if decision["action"] == "mine" and decision["target"] not in {
            block["id"] for block in observation.get("nearby_blocks", []) if isinstance(block, dict)}:
        raise ValueError("Mine target is not currently observed")
    if decision["action"] == "mine" and decision["target"] not in action_targets(observation, "mine"):
        raise ValueError("Mine target has no reachable standing position from here; tunnel toward it "
                         "(a lower y digs a staircase down) or choose another block")
    if decision["action"] == "mine" and has_coords:
        position = [decision[key] for key in ("x", "y", "z")]
        seen = any(block.get("id") == decision["target"] and block.get("at") == position
                   and block.get("reachable", True)
                   for block in observation.get("nearby_blocks", []) if isinstance(block, dict))
        exit_block = position in exit_dig_cells(observation, decision["target"])
        if not seen and not exit_block:
            # A stale coordinate still names an observed block type: let the bot pick the nearest one.
            LOG.info("Mine coordinate %s is no longer observed; using automatic %s selection", position,
                     decision["target"])
            for key in ("x", "y", "z"):
                decision[key] = 0
    if decision["action"] == "pickup" and decision["target"] and decision["target"] not in action_targets(observation, "pickup"):
        raise ValueError("Pickup target is not an observed unblocked item")
    if decision["action"] == "pickup" and not decision["target"] and any(
            row["action"] == "pickup" for row in observation.get("blocked_actions", [])):
        raise ValueError("Use a specific unblocked pickup item instead of repeating a failed generic pickup")
    if decision["action"] == "place":
        if decision["target"] not in action_targets(observation, "place"):
            raise ValueError("Placement target is not an owned unblocked block")
        sites = observation.get("crafting", {}).get("placement_sites", [])
        if [decision.get(k) for k in ("x", "y", "z")] not in sites:
            raise ValueError("Block placement is not a currently usable site")
    if decision["action"] == "attack" and decision["target"] not in action_targets(observation, "attack"):
        raise ValueError("Attack target is not an observed unblocked non-player entity")
    return decision


_STRATA_PROCESS = None


def ensure_strata_running(strata_url="http://127.0.0.1:8080", timeout=45, ollama="http://127.0.0.1:11434"):
    """Checks if STRATA endpoint is responsive; if not, unloads Ollama VRAM and launches local run-q2_0.sh."""
    global _STRATA_PROCESS
    try:
        req = urllib.request.Request(f"{strata_url}/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    unload_active_model(model=None, ollama=ollama)
    script = Path("/mnt/LINUX-NVME/STRATA/Strata-main/run-q2_0.sh")
    if not script.is_file():
        script = Path.home() / "STRATA/Strata-main/run-q2_0.sh"
    if script.is_file():
        LOG.info("STRATA not responding on %s. Launching %s in background...", strata_url, script)
        try:
            proc_check = subprocess.run(["pgrep", "-f", "serve/server.py"], capture_output=True, text=True)
            if not proc_check.stdout.strip():
                _STRATA_PROCESS = subprocess.Popen(["/bin/bash", str(script)], cwd=str(script.parent),
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                   start_new_session=True)
        except Exception as e:
            LOG.warning("Failed to spawn STRATA run script: %s", e)

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                req = urllib.request.Request(f"{strata_url}/health")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        LOG.info("STRATA server is ready at %s!", strata_url)
                        return True
            except Exception:
                time.sleep(1.5)
    return False


def unload_active_model(model=None, ollama="http://127.0.0.1:11434", strata_url="http://127.0.0.1:8080"):
    """Unloads the active model from VRAM/RAM when stopping Willow or shutting down."""
    global _STRATA_PROCESS
    if model and ("strata" in model.lower()):
        # Unload STRATA engine via /unload endpoint and terminate server process
        try:
            req = urllib.request.Request(f"{strata_url}/unload", data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                LOG.info("STRATA model unloaded: HTTP %d", resp.status)
        except Exception as e:
            LOG.debug("STRATA unload request error: %s", e)
        if _STRATA_PROCESS is not None:
            try:
                if _STRATA_PROCESS.poll() is None:
                    _STRATA_PROCESS.terminate()
                    try:
                        _STRATA_PROCESS.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        _STRATA_PROCESS.kill()
                        _STRATA_PROCESS.wait(timeout=2)
                LOG.info("Stopped tracked STRATA background process on model unload")
            except Exception as e:
                LOG.debug("STRATA process termination error: %s", e)
            finally:
                _STRATA_PROCESS = None
        return

    # Unload Ollama models
    try:
        models_to_unload = [model] if model else []
        if not models_to_unload:
            req = urllib.request.Request(f"{ollama}/api/ps")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models_to_unload = [m.get("model") or m.get("name") for m in data.get("models", []) if m]
        for m in models_to_unload:
            if not m:
                continue
            body = json.dumps({"model": m, "keep_alive": 0}).encode()
            req = urllib.request.Request(f"{ollama}/api/generate", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                LOG.info("Unloaded Ollama model %s (keep_alive=0): HTTP %d", m, resp.status)
    except Exception as e:
        LOG.debug("Ollama unload error: %s", e)


def _scaled_count(base, scale, low=0, high=32):
    if base <= 0:
        return 0
    return max(low, min(high, int(round(base * scale))))


class Brain:
    def __init__(self, database, model=MODEL, ollama="http://127.0.0.1:11434", planning=True, settings=None):
        self.model = model
        self.ollama = ollama
        self.settings = modelprofile.validate_settings(settings)
        self.profile = None
        self.profile_path = Path(database).parent / "model-profile.json"
        self.status_path = None if str(database) == ":memory:" else Path(database).parent / "status.json"
        self.started_at = time.time()
        self.db = None
        self.skills = None
        self.experience = None
        self.memory = None
        self.planner = None
        self.pipeline = None
        self.orchestrator = None
        self.decisions_since_export = 0
        self.pending_plan = None
        self.next_plan = 0
        self.plan_started = 0
        self.plan_dimension = None
        self.chat_revision = 0
        self.plan_revision = 0
        self.skill_hold_until = 0
        self.planner_status = "idle"
        self.steps = []
        self.steps_goal = ""
        self.steps_dimension = None
        self.steps_deadline = 0
        self.steps_last_id = ""
        self.steps_health = 20
        self.steps_food = 20
        self.last_speech = ""
        self.last_speech_time = 0
        self.strategy_error = ""
        self.thinking = None
        self.last_rejection = ""
        self.last_rejection_result_id = None
        self.consecutive_skill_id = None
        self.consecutive_skill_action = None
        self.consecutive_skill_count = 0
        self.consecutive_skill_no_progress = 0
        self._inference_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._active_inferences = 0
        self._active_actor_inferences = 0
        self._hardware = None

        try:
            self.db = sqlite3.connect(database, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions
                  (id TEXT PRIMARY KEY, created REAL, action TEXT, target TEXT,
                   goal TEXT, hypothesis TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS lessons
                  (id TEXT PRIMARY KEY, text TEXT NOT NULL, used REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS places
                  (kind TEXT, x INTEGER, y INTEGER, z INTEGER, seen REAL,
                   PRIMARY KEY(kind,x,y,z));
            """)
            self.db.commit()
            self.skills = SkillLibrary(self.db, Path(database).parent / "skills")
            self.experience = Experience(self.db)
            self.memory = Memory(self.db, Path(database).parent / "memory.json")
            self.planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="willow-strategist") if planning else None
            self.pipeline = ConcurrentPipeline(model=self.model, profile=self.profile, ollama=self.ollama)
            self.orchestrator = WillowBotOrchestrator(
                storage_dir=Path(database).parent,
                resident_model=self.model,
                ollama_url=self.ollama,
            )
            if hasattr(self.orchestrator, "optimizer") and self.orchestrator.optimizer:
                self.orchestrator.optimizer.align_with_profile(self.profile, keep_alive=self.keep_alive)
            if hasattr(self.orchestrator, "q_table") and self.orchestrator.q_table:
                self.experience.q_table = self.orchestrator.q_table
            if hasattr(self.orchestrator, "llm_client") and self.orchestrator.llm_client:
                self.inference_gate = self.orchestrator.llm_client.gate
        except Exception:
            self._cleanup_resources()
            raise

    def _cleanup_resources(self):
        if getattr(self, "orchestrator", None):
            try:
                self.orchestrator.stop()
            except Exception:
                pass
            self.orchestrator = None
        if getattr(self, "pipeline", None):
            try:
                self.pipeline.close()
            except Exception:
                pass
            self.pipeline = None
        if getattr(self, "planner", None):
            try:
                self.planner.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.planner = None
        if getattr(self, "db", None):
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    def close(self):
        with self._state_lock:
            if hasattr(self, "orchestrator") and self.orchestrator:
                self.orchestrator.stop()
                self.orchestrator = None
            if hasattr(self, "pipeline") and self.pipeline:
                self.pipeline.close()
                self.pipeline = None
            if self.planner:
                self.planner.shutdown(wait=True, cancel_futures=True)
                self.planner = None
            if self.memory:
                try:
                    self.memory.export()
                except (OSError, sqlite3.Error) as exc:
                    LOG.warning("Could not export memory.json: %s", exc)
            if self.db:
                self.db.close()
                self.db = None
            try:
                unload_active_model(self.model, ollama=self.ollama)
            except Exception as exc:
                LOG.debug("Error during model unload: %s", exc)

    def role_prompt_bytes(self, role="actor"):
        if self.profile:
            role_cfg = (self.profile.get("roles") or {}).get(role) or {}
            if "prompt_bytes" in role_cfg:
                return int(role_cfg["prompt_bytes"])
            return int(self.profile.get("prompt_bytes", PROMPT_BYTES))
        return PROMPT_BYTES

    @property
    def prompt_bytes(self):
        return self.role_prompt_bytes("actor")

    @prompt_bytes.setter
    def prompt_bytes(self, value):
        if self.profile is None:
            self.profile = {}
        self.profile["prompt_bytes"] = int(value)

    @property
    def v2_step_queue(self):
        return self.steps

    @v2_step_queue.setter
    def v2_step_queue(self, val):
        self.steps = val

    def queue_steps(self, steps, goal="plan", observation=None):
        self.steps = list(steps)
        self.steps_goal = goal
        self.steps_dimension = observation.get("dimension") if observation else "minecraft:overworld"
        self.steps_deadline = time.monotonic() + self.steps_deadline_seconds
        self.steps_health = observation.get("health", 20) if observation else 20
        self.steps_food = observation.get("food", 20) if observation else 20
        self.steps_last_id = observation.get("last_result", {}).get("id") if observation else None

    def call_model(self, messages, schema=None, role="actor", **kwargs):
        if schema is None:
            schema = SCHEMA if role == "actor" else PLAN_SCHEMA
        return self.inference(messages, schema, role=role, **kwargs)

    @property
    def bytes_per_token(self):
        val = (self.profile.get("bytes_per_token") if self.profile else None) or modelprofile.BYTES_PER_TOKEN
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = modelprofile.BYTES_PER_TOKEN
        return max(0.1, val)

    @property
    def prompt_tokens(self):
        if self.profile and "prompt_tokens" in self.profile:
            return int(self.profile["prompt_tokens"])
        return int(self.prompt_bytes / self.bytes_per_token)

    @property
    def trim_scale(self):
        if self.profile and "trim_scale" in self.profile:
            return float(self.profile["trim_scale"])
        return round(max(0.5, min(3.0, self.prompt_tokens / 5700.0)), 2)

    @property
    def compact_threshold_tokens(self):
        if self.profile and "compact_prompt_threshold_tokens" in self.profile:
            return int(self.profile["compact_prompt_threshold_tokens"])
        return 3428

    @property
    def actor_timeout(self):
        return int((self.profile.get("timeouts") or {}).get("actor_timeout") or ACTOR_TIMEOUT) if self.profile else ACTOR_TIMEOUT

    @property
    def plan_timeout(self):
        return int((self.profile.get("timeouts") or {}).get("plan_timeout") or PLAN_TIMEOUT) if self.profile else PLAN_TIMEOUT

    @property
    def plan_max_age(self):
        return int((self.profile.get("timeouts") or {}).get("plan_max_age") or PLAN_MAX_AGE) if self.profile else PLAN_MAX_AGE

    @property
    def steps_deadline_seconds(self):
        return int((self.profile.get("timeouts") or {}).get("steps_deadline") or 300) if self.profile else 300

    @property
    def next_plan_interval(self):
        return int((self.profile.get("timeouts") or {}).get("next_plan_interval") or 60) if self.profile else 60

    @property
    def skill_hold_seconds(self):
        return int((self.profile.get("timeouts") or {}).get("skill_hold_seconds") or 60) if self.profile else 60

    @property
    def strata_startup_timeout(self):
        return int((self.profile.get("timeouts") or {}).get("strata_startup_timeout") or 45) if self.profile else 45

    @property
    def keep_alive(self):
        return str((self.profile.get("timeouts") or {}).get("keep_alive") or "10m") if self.profile else "10m"

    @property
    def gameplay_limits(self):
        if self.profile and isinstance(self.profile.get("gameplay_limits"), dict):
            merged = dict(GAMEPLAY_LIMITS)
            merged.update(self.profile["gameplay_limits"])
            return merged
        return dict(GAMEPLAY_LIMITS)

    @property
    def response_limit_bytes(self):
        return int(self.gameplay_limits["response_limit_bytes"])

    @property
    def request_limit_bytes(self):
        return int(self.gameplay_limits["request_limit_bytes"])

    @property
    def lessons_cap(self):
        return int(self.gameplay_limits["lessons_cap"])

    @property
    def decisions_cap(self):
        return int(self.gameplay_limits["decisions_cap"])

    def _save_profile(self, profile):
        try:
            temporary = self.profile_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile, indent=2))
            temporary.replace(self.profile_path)
        except OSError as exc:
            LOG.warning("Could not save model profile: %s", exc)

    def _export_status(self, observation, decision, source, elapsed):
        if not self.status_path:
            return
        milestones = observation.get("milestones") if isinstance(observation.get("milestones"), dict) else {}
        diagnosis = observation.get("diagnosis") if isinstance(observation.get("diagnosis"), dict) else {}
        curriculum = observation.get("v2_curriculum") if isinstance(observation.get("v2_curriculum"), dict) else {}
        snapshot = {
            "updated_at": time.time(),
            "started_at": self.started_at,
            "ollama": self.ollama,
            "model": self.model,
            "health": observation.get("health"),
            "max_health": observation.get("max_health", 20),
            "food": observation.get("food"),
            "armor": observation.get("armor_points", 0),
            "position": observation.get("position"),
            "dimension": observation.get("dimension", ""),
            "biome": observation.get("biome", ""),
            "time": observation.get("time"),
            "weather": observation.get("weather", "clear"),
            "stuck": bool(diagnosis.get("stuck")),
            "stuck_reason": str(diagnosis.get("reason") or ""),
            "owned_tools": milestones.get("owned_tools") or {},
            "free_slots": milestones.get("inventory_free_slots"),
            "curriculum": curriculum,
            "goal": decision.get("goal", ""),
            "action": decision.get("action", ""),
            "target": decision.get("target", ""),
            "source": source,
            "elapsed": round(float(elapsed), 2),
            "plan_queued": len(self.steps),
            "planner_status": self.planner_status,
        }
        try:
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(compact(snapshot), encoding="utf-8")
            temporary.replace(self.status_path)
        except OSError:
            pass

    def autotune(self, run_benchmark=True):
        """Run hardware detection + live Ollama benchmark at startup, persist model-profile.json, and log all derived values."""
        try:
            self._hardware = modelprofile.detect_hardware()
            profile = modelprofile.autotune(
                self.model,
                ollama=self.ollama,
                settings=self.settings,
                run_benchmark=run_benchmark,
                save_path=self.profile_path,
            )
            self.profile = profile
            self.thinking = profile.get("thinking")
            if hasattr(self, "pipeline") and self.pipeline:
                self.pipeline.profile = profile
            LOG.info(modelprofile.format_startup_log(profile))
            return profile
        except Exception as exc:
            LOG.warning("Startup autotune fallback (%s); will build profile on first inference", exc)
            return None

    def ensure_profile(self, timeout=10):
        if self.profile is not None:
            return self.profile
        try:
            show = modelprofile.fetch_show(self.model, self.ollama, timeout=min(timeout, 10))
            profile = modelprofile.build(self.model, show, self.settings, hardware=self._hardware)
        except ValueError as exc:
            raise OSError(str(exc)) from exc
        self.profile = profile
        self.thinking = profile["thinking"]
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.profile = profile
        if hasattr(self, "orchestrator") and self.orchestrator and hasattr(self.orchestrator, "optimizer"):
            self.orchestrator.optimizer.align_with_profile(profile, keep_alive=self.keep_alive)
        LOG.info("Model profile model=%s arch=%s params=%s quant=%s trained_ctx=%s ctx=%s kv_cache=%sMiB "
                 "moe_experts=%s thinking=%s actor_tokens=%s plan_tokens=%s prompt_bytes=%s options=%s",
                 self.model, profile["architecture"], profile["parameter_size"], profile["quantization"],
                 profile["trained_context"], profile["context"], profile["kv_cache_mib"], profile["moe_experts"],
                 profile["thinking"], profile["actor_tokens"], profile["plan_tokens"], profile["prompt_bytes"],
                 profile["options"])
        self._save_profile(profile)
        return profile

    def _update_live_metrics(self, profile, prompt_byte_len, payload):
        p_tok = payload.get("prompt_eval_count")
        p_dur = payload.get("prompt_eval_duration")
        g_tok = payload.get("eval_count")
        g_dur = payload.get("eval_duration")
        if not (isinstance(p_tok, int) and p_tok > 20 and isinstance(p_dur, int) and p_dur > 0
                and isinstance(g_tok, int) and g_tok > 0 and isinstance(g_dur, int) and g_dur > 0):
            return
        measured_bpt = max(2.5, min(5.5, prompt_byte_len / p_tok))
        old_bpt = float(profile.get("bytes_per_token") or modelprofile.BYTES_PER_TOKEN)
        new_bpt = round(0.7 * old_bpt + 0.3 * measured_bpt, 3)
        profile["bytes_per_token"] = new_bpt
        if "prompt_tokens" in profile:
            new_pb = int(min(
                modelprofile.PROMPT_BYTES_CAP,
                max(modelprofile.PROMPT_BYTES_MIN, profile["prompt_tokens"] * new_bpt),
            ))
            profile["prompt_bytes"] = new_bpt_prompt = new_pb
            if "roles" in profile:
                for r_name, r_dict in profile["roles"].items():
                    if isinstance(r_dict, dict) and "prompt_tokens" in r_dict:
                        r_dict["prompt_bytes"] = int(min(
                            modelprofile.PROMPT_BYTES_CAP,
                            max(modelprofile.PROMPT_BYTES_MIN, r_dict["prompt_tokens"] * new_bpt),
                        ))
                    elif isinstance(r_dict, dict):
                        r_dict["prompt_bytes"] = new_bpt_prompt
        live_p_tps = p_tok / (p_dur / 1e9)
        live_g_tps = g_tok / (g_dur / 1e9)
        old_p_tps = float(profile.get("prompt_tps") or live_p_tps)
        old_g_tps = float(profile.get("gen_tps") or live_g_tps)
        profile["prompt_tps"] = round(0.7 * old_p_tps + 0.3 * live_p_tps, 1)
        profile["gen_tps"] = round(0.7 * old_g_tps + 0.3 * live_g_tps, 1)
        profile["timeouts"] = modelprofile.derive_timeouts(
            prompt_tokens=min(int(profile.get("prompt_tokens") or 4096), 6000),
            actor_tokens=int(profile.get("actor_tokens") or 512),
            plan_tokens=int(profile.get("plan_tokens") or 1024),
            prompt_tps=profile["prompt_tps"],
            gen_tps=profile["gen_tps"],
            is_strata=bool(profile.get("is_strata")),
        )
        if "roles" in profile:
            if "actor" in profile["roles"]:
                profile["roles"]["actor"]["timeout"] = profile["timeouts"]["actor_timeout"]
            if "strategist" in profile["roles"]:
                profile["roles"]["strategist"]["timeout"] = profile["timeouts"]["plan_timeout"]

    def inference(self, messages, schema, role, output=None, timeout=None, images=None, temperature_boost=0.0):
        with self._inference_lock:
            self._active_inferences += 1
            if role == "actor":
                self._active_actor_inferences += 1
        try:
            return self._run_inference(messages, schema, role, output=output, timeout=timeout, images=images,
                                       temperature_boost=temperature_boost)
        finally:
            with self._inference_lock:
                self._active_inferences = max(0, self._active_inferences - 1)
                if role == "actor":
                    self._active_actor_inferences = max(0, self._active_actor_inferences - 1)

    def _run_inference(self, messages, schema, role, output=None, timeout=None, images=None, temperature_boost=0.0):
        started = time.monotonic()
        profile = self.ensure_profile(timeout or (self.plan_timeout if role == "strategist" else self.actor_timeout))
        role_cfg = (profile.get("roles") or {}).get(role) or {}

        if timeout is None:
            timeout = role_cfg.get("timeout") or (self.plan_timeout if role == "strategist" else self.actor_timeout)
        if output is None:
            output = role_cfg.get("num_predict") or (profile["plan_tokens"] if role == "strategist" else profile["actor_tokens"])
        prompt_budget = role_cfg.get("prompt_bytes") or profile.get("prompt_bytes", PROMPT_BYTES)

        msg_bytes = len(compact(messages).encode())
        schema_bytes = len(compact(schema).encode())
        if msg_bytes + schema_bytes > prompt_budget:
            raise OSError(f"{role} prompt exceeds context budget")
        msgs = copy.deepcopy(messages)
        resp_limit = self.response_limit_bytes
        is_strata = profile.get("is_strata") or self.model.lower().startswith("strata")
        if is_strata:
            strata_url = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("STRATA inference deadline exceeded")
            ensure_strata_running(strata_url, timeout=max(5, min(self.strata_startup_timeout, int(remaining))), ollama=self.ollama)
            strata_msgs = []
            for m in msgs:
                if m.get("role") == "user" and images and profile.get("vision_enabled", profile.get("multimodal_capabilities", {}).get("vision")):
                    content = [{"type": "text", "text": str(m.get("content", ""))}]
                    for img in images:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    strata_msgs.append({"role": m["role"], "content": content})
                else:
                    strata_msgs.append(m)
            strata_schema = copy.deepcopy(schema)
            if strata_schema.get("type") != "object":
                # STRATA / OpenAI JSON schema specification requires {"type": "object"} at root
                strata_schema = copy.deepcopy(SCHEMA)
                if isinstance(schema, dict) and "oneOf" in schema:
                    actions = set()
                    for branch in schema["oneOf"]:
                        if isinstance(branch, dict) and "properties" in branch and "action" in branch["properties"]:
                            enums = branch["properties"]["action"].get("enum", [])
                            actions.update(enums)
                    if actions:
                        strata_schema["properties"]["action"]["enum"] = sorted(actions)

            strata_temp = role_cfg.get("temperature", profile["options"].get("temperature", 0.2))
            if temperature_boost:
                strata_temp = round(min(1.2, float(strata_temp or 0.2) + float(temperature_boost)), 2)
            strata_req = {
                "model": profile.get("model", "qwen3.8-flash-next-q2_0"),
                "messages": strata_msgs,
                "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": strata_schema}},
                "temperature": strata_temp,
                "max_tokens": max(1024, output),
                "stream": False,
            }
            req = urllib.request.Request(f"{strata_url}/v1/chat/completions",
                                         data=compact(strata_req).encode(),
                                         headers={"Content-Type": "application/json"})
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("STRATA inference deadline exceeded")
            try:
                with urllib.request.urlopen(req, timeout=remaining) as response:
                    raw = response.read(resp_limit + 1)
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", "replace")
                except Exception:
                    error_body = ""
                LOG.error("STRATA HTTP %s error for role %s: %s", exc.code, role, error_body)
                raise OSError(f"STRATA HTTP {exc.code} for {role}: {error_body}") from exc
            except urllib.error.URLError as exc:
                LOG.error("STRATA connection error for role %s: %s", role, exc)
                raise OSError(f"Cannot connect to STRATA on {strata_url}: {exc.reason}. Ensure STRATA server is running.") from exc
            if len(raw) > resp_limit:
                raise OSError("STRATA response exceeds size limit")
            payload = json.loads(raw)
            if "error" in payload:
                err_msg = payload["error"].get("message", str(payload["error"]))
                LOG.error("STRATA returned error: %s", err_msg)
                raise OSError(f"STRATA error: {err_msg}")
            content = payload["choices"][0]["message"]["content"]
            LOG.info("role=%s STRATA elapsed=%.2fs", role, time.monotonic() - started)
            return json.loads(content)

        if images and profile.get("vision_enabled", profile.get("multimodal_capabilities", {}).get("vision")) and msgs:
            if isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user":
                msgs[-1]["images"] = images

        req_options = dict(profile["options"], num_predict=output)
        if "temperature" in role_cfg and role_cfg["temperature"] is not None:
            req_options["temperature"] = role_cfg["temperature"]
        if temperature_boost:
            # Retries after a rejection: a deterministic low temperature would reproduce the same answer.
            base_temp = req_options.get("temperature")
            base_temp = float(base_temp) if isinstance(base_temp, (int, float)) else 0.6
            req_options["temperature"] = round(min(1.2, base_temp + float(temperature_boost)), 2)

        request = {
            "model": self.model, "messages": msgs, "format": schema,
            "stream": False, "keep_alive": self.keep_alive,
            "options": req_options,
        }
        role_think_val = role_cfg.get("think_value", profile.get("think_value"))
        if role_think_val is not None:
            request["think"] = role_think_val

        def _send_ollama(req_dict):
            req_obj = urllib.request.Request(self.ollama + "/api/chat", data=compact(req_dict).encode(),
                                             headers={"Content-Type": "application/json"})
            rem = timeout - (time.monotonic() - started)
            if rem <= 0:
                raise TimeoutError("Model metadata consumed inference deadline")
            try:
                with urllib.request.urlopen(req_obj, timeout=rem) as response:
                    raw_bytes = response.read(resp_limit + 1)
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", "replace")
                except Exception:
                    error_body = ""
                LOG.error("Ollama HTTP %s error for role %s: %s", exc.code, role, error_body)
                raise OSError(f"Ollama HTTP {exc.code} for {role}: {error_body}") from exc
            if len(raw_bytes) > resp_limit:
                raise OSError("Ollama response exceeds size limit")
            return json.loads(raw_bytes)

        gate = getattr(self, "inference_gate", None)
        if gate is None and hasattr(self, "orchestrator") and self.orchestrator:
            gate = getattr(getattr(self.orchestrator, "llm_client", None), "gate", None)
            self.inference_gate = gate

        priority = InferencePriority.REFLEX_DECISION if role == "actor" else InferencePriority.TASK_DECOMPOSITION
        if gate is not None:
            gate.acquire(priority=priority, timeout=timeout)
        try:
            payload = _send_ollama(request)
            if payload.get("done_reason") == "length" and request.get("think") and (timeout - (time.monotonic() - started)) > 3.0:
                non_think_cap = max(384, min(1024, output))
                LOG.warning("role=%s exhausted num_predict=%s while thinking; retrying with think=False (num_predict=%s)",
                            role, output, non_think_cap)
                retry_req = dict(request, think=False, options=dict(request["options"], num_predict=non_think_cap))
                payload = _send_ollama(retry_req)
                if payload.get("done_reason") != "length":
                    # Only the role that ran out of budget stops thinking; the other role keeps its own settings.
                    role_profile = (profile.get("roles") or {}).get(role)
                    if isinstance(role_profile, dict):
                        role_profile["think"] = False
                        role_profile["think_value"] = False
                        role_profile["num_predict"] = non_think_cap
                    if role == "actor":
                        profile["thinking"] = False
                        profile["think_value"] = False
                        profile["actor_tokens"] = non_think_cap
                    else:
                        profile["plan_tokens"] = non_think_cap
            if payload.get("done_reason") == "length":
                raise ValueError("Model exhausted output budget; refusing partial response")
            self._update_live_metrics(profile, msg_bytes, payload)
            LOG.info("role=%s prompt_tokens=%s cached=%s generated_tokens=%s elapsed=%.2fs prompt_tps=%s gen_tps=%s", role,
                     payload.get("prompt_eval_count"), payload.get("prompt_eval_cached_count", "-"),
                     payload.get("eval_count"), time.monotonic() - started,
                     rate(payload.get("prompt_eval_count"), payload.get("prompt_eval_duration")),
                     rate(payload.get("eval_count"), payload.get("eval_duration")))
            return json.loads(payload["message"]["content"])
        finally:
            if gate is not None:
                gate.release()

    def think_plan(self, messages):
        value = self.inference(messages, PLAN_SCHEMA, "strategist", timeout=self.plan_timeout)
        value = self.check_plan(value)
        error = value.pop("code_error", "")
        rejected = value.pop("rejected_code", "")
        planner_mode = (self.profile or {}).get("planner_mode", "idle_only")
        if error and not (planner_mode == "idle_only" and self._active_actor_inferences > 0):
            # One focused repair: show the model its own code plus the exact validator message.
            LOG.info("Asking strategist to repair its skill once: %s", error)
            repair = messages + [
                {"role": "assistant", "content": compact(dict(value, code=rejected))},
                {"role": "user", "content": (
                    f"Your skill code was REJECTED by the validator: {error}\n"
                    "Rewrite it in the restricted skill language: one def skill():, only local variables, "
                    "if/elif/else, return, comparisons, and/or/not, integer +/-, and the listed helpers. "
                    "No loops, imports, lists, indexing, attributes, f-strings, or other functions. "
                    "count() reads INVENTORY; drops() checks items on the ground. Return the same JSON fields.")}]
            try:
                repaired = self.check_plan(self.inference(repair, PLAN_SCHEMA, "strategist", timeout=self.plan_timeout))
                error = repaired.pop("code_error", "")
                repaired.pop("rejected_code", None)
                if repaired["code"]:
                    LOG.info("Strategist repaired its skill code")
                    value = repaired
            except (ValueError, KeyError, TypeError, OSError) as exc:
                error = f"{error}; repair failed: {exc}"
        if error:
            value["code_error"] = error[:DECISION_LIMITS["memory"]]
        return value

    def check_plan(self, value):
        if not isinstance(value, dict) or set(value) != set(PLAN_SCHEMA["required"]):
            raise ValueError("Strategist response has invalid fields")
        for key, limit in (("goal", DECISION_LIMITS["goal"]), ("reason", DECISION_LIMITS["memory"])):
            if isinstance(value.get(key), str):
                encoded = value[key].encode("utf-8")
                if len(encoded) > limit:
                    value[key] = encoded[:limit].decode("utf-8", "ignore")
            if not isinstance(value.get(key), str) or len(value[key].encode("utf-8")) > limit:
                raise ValueError("Oversized strategist field: " + key)
        value["code"] = normalize(value.get("code"))
        if not isinstance(value.get("code"), str) or len(value["code"].encode("utf-8")) > DECISION_LIMITS["code"]:
            LOG.warning("Strategist generated invalid or oversized code; discarding code")
            value["code"] = ""
        elif value["code"]:
            allow_bounded = bool(self.settings.get("bounded_repetition"))
            try:
                parse(value["code"], allow_bounded_repetition=allow_bounded)
            except ValueError as exc:
                LOG.warning("Strategist generated invalid skill code: %s; discarding code", exc)
                LOG.warning("Rejected skill source:\n%s", value["code"][:DECISION_LIMITS["code"]])
                value["rejected_code"] = value["code"]
                value["code"] = ""
                value["code_error"] = str(exc)
        return value

    def update_planner(self, observation):
        with self._state_lock:
            return self._update_planner_locked(observation)

    def _update_planner_locked(self, observation):
        if not self.planner:
            return
        now = time.monotonic()
        if observation.get("chat"):
            self.chat_revision += 1
            self.skill_hold_until = now + self.skill_hold_seconds
        if self.pending_plan is not None and self.pending_plan.done():
            future, self.pending_plan = self.pending_plan, None
            try:
                plan = future.result()
                if (now - self.plan_started > self.plan_max_age or self.plan_dimension != observation.get("dimension")
                        or self.plan_revision != self.chat_revision):
                    self.planner_status = "discarded stale strategy"
                    LOG.warning(self.planner_status)
                else:
                    counts = inventory(observation)
                    pickaxes = sum(v for k, v in counts.items() if k.endswith("_pickaxe"))
                    planks = sum(v for k, v in counts.items() if k.endswith("_planks"))
                    stone_picks = counts.get("minecraft:stone_pickaxe", 0) + counts.get("minecraft:iron_pickaxe", 0)
                    cobble = counts.get("minecraft:cobblestone", 0)
                    goal_lower = (plan.get("goal") or "").lower()
                    if (pickaxes > 0 and stone_picks == 0 and cobble < 3 and planks >= 8
                            and not observation.get("chat")
                            and any(w in goal_lower for w in ("plank", "wooden pickaxe", "wooden_pickaxe", "oak log", "gather wood"))):
                        plan["goal"] = "Find and mine 3 cobblestone with wooden pickaxe to craft a stone pickaxe."
                        plan["reason"] = "Wooden pickaxe and ample wood/planks are already in inventory; next progression step is mining stone for cobblestone."
                    if plan["code"]:
                        # Task B2: Dry-run candidate skill against recent past observations before saving
                        past_obs = []
                        try:
                            rows = self.db.execute(
                                "SELECT hypothesis FROM decisions WHERE hypothesis IS NOT NULL "
                                "AND hypothesis != '' ORDER BY created DESC LIMIT 20"
                            ).fetchall()
                            for (h_json,) in rows:
                                try:
                                    parsed = json.loads(h_json)
                                    if isinstance(parsed, dict) and "observed" in parsed and isinstance(parsed["observed"], dict):
                                        past_obs.append(parsed["observed"])
                                except Exception:
                                    continue
                        except Exception:
                            pass
                        if not past_obs:
                            # Use current observation as baseline test case
                            past_obs = [observation]

                        allow_rep = bool(self.settings.get("bounded_repetition"))
                        passed, reason = dry_run(
                            plan["code"], past_obs, validate_action_fn=validate_action, allow_bounded_repetition=allow_rep
                        )
                        if passed:
                            self.skills.install(plan["code"], dry_run_passed=1, allow_bounded_repetition=allow_rep)
                            LOG.info("Skill passed dry-run and installed to library")
                        else:
                            LOG.warning("Skill dry-run failed: %s; discarding code", reason)
                            plan["code_error"] = f"Dry-run rejected: {reason}"
                            plan["code"] = ""
                    with self.db:
                        self.db.execute("INSERT OR REPLACE INTO state VALUES ('strategy',?)",
                                        (compact({"goal": plan["goal"], "reason": plan["reason"]}),))
                    self.planner_status = "strategy updated"
                    self.strategy_error = ("Last skill code rejected: " + plan["code_error"])[:DECISION_LIMITS["memory"]] \
                        if plan.get("code_error") else ""
                    LOG.info("Strategist goal=%s reason=%s", plan["goal"], plan["reason"])
            except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
                self.planner_status = "strategist failed; see brain.log"
                self.strategy_error = str(exc)[:DECISION_LIMITS["memory"]]
                LOG.error("Background strategist failed: %s", exc)
                LOG.debug("Background strategist failure details:", exc_info=True)

    def schedule_planner(self, observation, messages):
        with self._state_lock:
            return self._schedule_planner_locked(observation, messages)

    def _schedule_planner_locked(self, observation, messages):
        if not self.planner:
            return
        planner_mode = (self.profile or {}).get("planner_mode")
        if planner_mode == "disabled":
            self.planner_status = "disabled (low VRAM budget)"
            return
        now = time.monotonic()
        if self.pending_plan is not None or now < self.next_plan:
            return
        if planner_mode == "idle_only":
            limits = self.gameplay_limits
            in_danger = (
                observation.get("food", 20) < limits["danger_food"]
                or observation.get("health", 20) < limits["danger_health"]
                or observation.get("diagnosis", {}).get("stuck")
                or any(
                    isinstance(e, dict) and e.get("hostile") and e.get("distance", 99) < limits["danger_hostile_distance"]
                    for e in observation.get("entities", [])
                )
            )
            if self._active_inferences > 0 or self._active_actor_inferences > 0 or in_danger:
                self.planner_status = "deferred (actor busy or survival priority)"
                return
        snapshot = json.loads(messages[-1]["content"])
        scale = self.trim_scale
        # Task B4: Retrieve only top 3-5 relevant skills instead of full list
        max_skills = max(3, min(5, _scaled_count(4, scale, low=3, high=5)))
        snapshot["skill_programs"] = self.skills.report(
            observation=snapshot, top_k=max_skills, ollama=self.ollama
        )
        # Task B5: Few-shot example skills
        snapshot["example_skills"] = retrieve_few_shot_examples(
            snapshot, top_k=2, ollama=self.ollama
        )
        snapshot["previous_strategy_error"] = self.strategy_error
        list_fields = ("skill_programs", "example_skills", "nearby_blocks", "entities", "observed_lessons",
                       "recent_actions", "memory_events", "knowledge", "remembered_resources",
                       "my_placed_blocks", "craftable", "lookup", "waypoints", "my_notes")
        cap = _scaled_count(6, scale, low=2, high=20)
        for fname in list_fields:
            if isinstance(snapshot.get(fname), list) and len(snapshot[fname]) > cap:
                snapshot[fname] = snapshot[fname][:cap]
        # The worker receives immutable serialized data and never touches SQLite.
        system = PLAN_SYSTEM_COMPACT if self.prompt_tokens < self.compact_threshold_tokens else PLAN_SYSTEM
        plan_budget = self.role_prompt_bytes("strategist")
        while True:
            plan_messages = [{"role": "system", "content": system},
                             {"role": "user", "content": compact(snapshot)}]
            if len(compact(plan_messages).encode()) + len(compact(PLAN_SCHEMA).encode()) <= plan_budget:
                break
            fields = [key for key in list_fields if isinstance(snapshot.get(key), list) and snapshot[key]]
            if not fields:
                if "milestones" in snapshot and isinstance(snapshot["milestones"], dict):
                    snapshot["milestones"].pop("reminder", None)
                if "crafting" in snapshot and isinstance(snapshot["crafting"], dict):
                    snapshot["crafting"].pop("knowledge", None)
                if "diagnosis" in snapshot and isinstance(snapshot["diagnosis"], dict):
                    snapshot["diagnosis"].pop("risk_policy", None)
                if "terrain" in snapshot and isinstance(snapshot["terrain"], dict):
                    snapshot["terrain"].pop("exits", None)
                snapshot.pop("strategy", None)
                snapshot.pop("strategy_error", None)
                snapshot.pop("previous_strategy_error", None)
                snapshot.pop("my_placed_totals", None)
                if self.prompt_tokens < self.compact_threshold_tokens and system != PLAN_SYSTEM_COMPACT:
                    system = PLAN_SYSTEM_COMPACT
                    continue
                plan_messages = [{"role": "system", "content": system},
                                 {"role": "user", "content": compact(snapshot)}]
                if len(compact(plan_messages).encode()) + len(compact(PLAN_SCHEMA).encode()) <= plan_budget:
                    break
                LOG.error("Strategist skipped: essential snapshot exceeds context budget")
                self.planner_status = "strategist snapshot oversized"
                self.next_plan = now + self.next_plan_interval
                return
            key = max(fields, key=lambda name: len(compact(snapshot[name])))
            snapshot[key] = snapshot[key][:-1]
        self.plan_started = now
        self.plan_dimension = observation.get("dimension")
        self.plan_revision = self.chat_revision
        self.next_plan = now + self.next_plan_interval
        self.planner_status = "thinking"
        self.pending_plan = self.planner.submit(self.think_plan, plan_messages)

    def record_result(self, observation):
        with self._state_lock:
            return self._record_result_locked(observation)

    def _record_result_locked(self, observation):
        result = observation.get("last_result", {})
        if not isinstance(result, dict):
            raise ValueError("last_result must be an object")
        decision_id = result.get("id")
        if not decision_id:
            return
        row = self.db.execute(
            "SELECT action,target,hypothesis,result FROM decisions WHERE id=?", (decision_id,)
        ).fetchone()
        if not row or row[3] is not None:
            return
        delta = self.experience.learn(decision_id, row[0], row[1], observation)
        self.memory.record(result.get("action", row[0]), memory_target(row[0], row[1]), result, delta,
                           observation.get("dimension", "unknown"))
        encoded = compact(dict(result, inventory_change=delta or {}))
        self.db.execute("UPDATE decisions SET result=? WHERE id=?", (encoded, decision_id))
        status = result.get("status")
        self.skills.outcome(decision_id, None if status == "interrupted" else status == "success")
        skill_row = self.db.execute("SELECT skill_id FROM decision_skills WHERE decision_id=?", (decision_id,)).fetchone()
        if skill_row:
            if bool(delta):
                self.consecutive_skill_no_progress = 0
            else:
                self.consecutive_skill_no_progress += 1
        else:
            self.consecutive_skill_no_progress = 0
        # Store observed outcomes separately from model-authored hypotheses.
        if status == "interrupted":
            self.db.commit()
            return
        lesson = (f"{result.get('action', row[0])} {row[1]}: {result.get('status', '')} "
                  f"{str(result.get('detail', ''))[:160]}")
        if "damage" in result:
            lesson += f" damage={result['damage']} health_after={result.get('health_after')}"
        key = hashlib.sha256(lesson.encode()).hexdigest()
        self.db.execute("INSERT OR REPLACE INTO lessons VALUES (?,?,?)", (key, lesson, time.time()))
        self.db.execute(
            "DELETE FROM lessons WHERE id NOT IN (SELECT id FROM lessons ORDER BY used DESC LIMIT ?)",
            (self.lessons_cap,),
        )
        self.db.commit()

    def recent_actions(self, limit):
        return [
            {"action": row[0], "target": row[1], "result": json.loads(row[2])}
            for row in self.db.execute(
                "SELECT action,target,result FROM decisions WHERE result IS NOT NULL ORDER BY created DESC LIMIT ?",
                (int(limit),),
            )
        ]

    def perceive(self, observation):
        with self._state_lock:
            return self._perceive_locked(observation)

    def _perceive_locked(self, observation):
        self.record_result(observation)
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.ingest(observation)
        with self.db:
            self.memory.observe(observation)
        context = dict(observation)
        context["_gameplay_limits"] = self.gameplay_limits
        if hasattr(self, "pipeline") and self.pipeline:
            context = self.pipeline.enrich(context)
        context["lookup"], context["blocked_actions"] = self.experience.context(observation)
        context["inventory_change"] = self.experience.changes(observation)
        # Most-recent-first outcome history; validate_action uses it to break goto/explore loops.
        if "recent_actions" not in context:
            context["recent_actions"] = self.recent_actions(_scaled_count(20, self.trim_scale, low=6, high=32))
        counts = inventory(observation)
        pos = observation.get("position")
        sites = observation.get("crafting", {}).get("placement_sites", [])
        placeable = observation.get("placeable", [])
        can_pillar = bool((pos and pos in sites or "under_feet_placement_site" in observation.get("crafting", {}))
                          and any(item in placeable for item in counts))
        nearby_logs = [
            b for b in observation.get("nearby_blocks", [])
            if isinstance(b, dict) and isinstance(b.get("id"), str)
            and b["id"].endswith(("_log", "_stem"))
            and isinstance(b.get("at"), list) and len(b["at"]) == 3
            and b.get("reachable", True)
            and not blocked(observation, "mine", b["id"])
        ]
        logs_total = sum(v for k, v in counts.items() if k.endswith(("_log", "_stem")))
        planks_total = sum(v for k, v in counts.items() if k.endswith("_planks"))
        pickaxes_total = sum(v for k, v in counts.items() if k.endswith("_pickaxe"))
        if nearby_logs and logs_total < 8 and (pickaxes_total == 0 or (logs_total == 0 and planks_total < 4)):
            first_log = nearby_logs[0]
            context["visible_tree"] = {
                "id": first_log["id"],
                "nearest_at": first_log["at"],
                "count": len(nearby_logs),
                "recommended_action": {
                    "action": "mine",
                    "target": first_log["id"],
                    "x": int(first_log["at"][0]),
                    "y": int(first_log["at"][1]),
                    "z": int(first_log["at"][2]),
                },
            }
        enough_blocks = {
            item: count for item, count in counts.items() if item in BULK_ITEMS
            and any(drop == item and stockpile_full(context, block) for block, drop in BULK_DROPS.items())
        }
        if stockpile_full(context, "minecraft:oak_log"):
            for item, count in counts.items():
                if item.endswith(("_log", "_stem", "_planks")):
                    enough_blocks[item] = count
        context["milestones"] = {
            "owned_tools": {item: count for item, count in counts.items()
                            if item.endswith(("_pickaxe", "_axe", "_shovel", "_sword", "_hoe"))},
            "table_owned_or_reachable": bool(counts.get("minecraft:crafting_table")
                                             or observation.get("crafting", {}).get("table_reachable")),
            "stone_pickaxe_complete": counts.get("minecraft:stone_pickaxe", 0) > 0,
            "can_pillar_up": can_pillar,
            "inventory_free_slots": free_slots(observation),
            "enough_building_blocks": enough_blocks,
        }
        if (pickaxes_total > 0 and counts.get("minecraft:stone_pickaxe", 0) == 0
                and counts.get("minecraft:iron_pickaxe", 0) == 0 and counts.get("minecraft:cobblestone", 0) < 3):
            context["milestones"]["next_milestone"] = (
                "You already own a wooden_pickaxe! Stop chopping trees or crafting planks/wood; "
                "mine minecraft:stone (or explore/tunnel to find exposed stone) to gather 3 cobblestone for a stone_pickaxe."
            )
        if hasattr(self, "orchestrator") and self.orchestrator:
            pos_val = observation.get("position")
            if pos_val is None or (isinstance(pos_val, (list, tuple)) and len(pos_val) == 3):
                try:
                    context["v2_curriculum"] = self.orchestrator.observe_turn_transition(observation)
                except Exception as exc:
                    LOG.debug("V2 curriculum turn transition skipped: %s", exc)
        return context

    def messages(self, observation, correction="", role="actor"):
        if "milestones" not in observation:
            observation = self.perceive(observation)
        scale = self.trim_scale
        target_budget = self.role_prompt_bytes(role)
        lesson_limit = _scaled_count(12, scale, low=4, high=24)
        recent_limit = _scaled_count(20, scale, low=6, high=32)
        goal = self.db.execute("SELECT value FROM state WHERE key='goal'").fetchone()
        lessons = [r[0] for r in self.db.execute(
            "SELECT text FROM lessons ORDER BY used DESC LIMIT ?", (lesson_limit,)
        )]
        # Key order matters for local inference: Ollama reuses the KV cache only for the longest unchanged
        # prefix (and only once that prefix exceeds roughly 1k tokens), so slow-changing memory/inventory
        # data comes first and anything that changes every decision comes last.
        stable_keys = ("inventory", "placeable", "milestones", "craftable", "smeltable", "trades",
                       "lookup", "blocked_actions", "inventory_change", "dimension", "biome", "v2_curriculum")
        volatile_keys = ("crafting", "position", "health", "food", "air", "on_ground", "in_water", "time",
                         "weather", "terrain", "diagnosis", "nearby_blocks", "visible_tree", "entities", "seeing",
                         "listening", "doing", "sight", "hearing", "lfm_omni_synthesis", "active_tools",
                         "last_result", "chat")
        observed = {k: copy.deepcopy(observation[k]) for k in stable_keys + volatile_keys if k in observation}
        # Constant guidance lives in the system prompt; resending it per decision only costs tokens.
        if isinstance(observed.get("crafting"), dict):
            observed["crafting"].pop("knowledge", None)
        if isinstance(observed.get("diagnosis"), dict):
            observed["diagnosis"].pop("risk_policy", None)
        if isinstance(observed.get("milestones"), dict):
            observed["milestones"].pop("reminder", None)
        # The pipeline mirrors sight->seeing and hearing->listening; sending both wastes prompt tokens.
        if "seeing" in observed:
            observed.pop("sight", None)
        if "listening" in observed:
            observed.pop("hearing", None)
        for key in ("seeing", "sight"):
            if key in observed and isinstance(observed[key], dict):
                observed[key].pop("snapshot_png_base64", None)
        if "craftable" in observed:
            observed["craftable"] = craft_targets(observation)
        strategy = self.db.execute("SELECT value FROM state WHERE key='strategy'").fetchone()
        if isinstance(observation.get("recent_actions"), list):
            recent_actions = copy.deepcopy(observation["recent_actions"][:recent_limit])
        else:
            recent_actions = self.recent_actions(recent_limit)

        memory_context = self.memory.context(observation)
        context = {}
        if strategy:
            context["strategy"] = json.loads(strategy[0])
        context["my_notes"] = memory_context.pop("my_notes", [])
        for key in stable_keys:
            if key in observed:
                context[key] = observed[key]
        # Per-decision data from here on: the model's own goal text, status, distances and histories.
        context["saved_goal"] = goal[0] if goal else "Gather wood and establish a safe survival base."
        context["strategist_status"] = self.planner_status
        context.update(memory_context)
        context["observed_lessons"] = lessons
        context["recent_actions"] = recent_actions
        for key in volatile_keys:
            if key in observed:
                context[key] = observed[key]
        if correction:
            context["correction"] = str(correction)[:DECISION_LIMITS["memory"]]

        # Explicit active tools declaration for tool awareness
        seeing_info = context.get("seeing") or {}
        listening_info = context.get("listening") or {}
        active_tools = {
            "vision": {
                "status": "online",
                "looking_at": seeing_info.get("looking_at") or (observation.get("sight") or {}).get("looking_at"),
                "visible_entities": [
                    {"type": e.get("type"), "distance": e.get("distance"), "bearing": e.get("bearing")}
                    for e in (seeing_info.get("visible_entities") or (observation.get("sight") or {}).get("visible_entities", []))[:6]
                    if isinstance(e, dict)
                ],
                "visual_hazards": seeing_info.get("visual_hazards", []),
            },
            "hearing": {
                "status": "online",
                "alerts": listening_info.get("active_alerts") or (observation.get("hearing") or {}).get("active_alerts", []),
            },
            "actions": allowed_actions(observation),
        }
        vis_insight = seeing_info.get("vision_insight") or seeing_info.get("lfm_vision_insight")
        if vis_insight:
            active_tools["vision"]["insight"] = vis_insight
        aud_insight = listening_info.get("audio_insight") or listening_info.get("lfm_audio_insight")
        if aud_insight:
            active_tools["hearing"]["insight"] = aud_insight
        omni_syn = context.pop("lfm_omni_synthesis", None) or observation.get("omni_synthesis")
        if omni_syn:
            active_tools["omni_synthesis"] = omni_syn
        context["active_tools"] = active_tools

        schema_bytes = len(compact(action_schema(observation)).encode())
        s = lambda base, low=0, high=32: _scaled_count(base, scale, low=low, high=high)

        stages = [
            # Stage 0: Standard rich context scaled to real token budget
            {"nearby_blocks": s(4, 2, 16), "craftable": s(4, 2, 16), "entities": s(2, 1, 8), "lookup": s(4, 2, 12),
             "observed_lessons": s(4, 2, 12), "recent_actions": s(6, 2, 16), "chat": 1, "memory_events": 0,
             "knowledge": s(4, 2, 12), "remembered_resources": s(2, 1, 8), "my_notes": s(4, 2, 12),
             "my_placed_blocks": s(4, 2, 12), "waypoints": s(4, 2, 12)},
            # Stage 1: Moderate trimming
            {"nearby_blocks": s(3, 2, 10), "craftable": s(3, 2, 10), "entities": s(1, 1, 4), "lookup": s(2, 2, 6),
             "observed_lessons": s(2, 1, 6), "recent_actions": s(2, 2, 8), "chat": 1, "memory_events": 0,
             "knowledge": s(2, 1, 6), "remembered_resources": s(1, 1, 4), "my_notes": s(2, 1, 6),
             "my_placed_blocks": s(1, 1, 4), "waypoints": s(2, 1, 6)},
            # Stage 2: Aggressive trimming & compaction
            {"nearby_blocks": 2, "craftable": 2, "entities": 1, "lookup": 1,
             "observed_lessons": 1, "recent_actions": 2, "chat": 1, "memory_events": 0,
             "knowledge": 1, "remembered_resources": 0, "my_notes": 1, "my_placed_blocks": 0,
             "waypoints": 1},
            # Stage 3: Minimal survival floor
            {"nearby_blocks": 1, "craftable": 1, "entities": 0, "lookup": 0,
             "observed_lessons": 0, "recent_actions": 1, "chat": 1, "memory_events": 0,
             "knowledge": 0, "remembered_resources": 0, "my_notes": 0, "my_placed_blocks": 0,
             "waypoints": 0},
            # Stage 4: Absolute emergency floor
            {"nearby_blocks": 0, "craftable": 0, "entities": 0, "lookup": 0,
             "observed_lessons": 0, "recent_actions": 0, "chat": 1, "memory_events": 0,
             "knowledge": 0, "remembered_resources": 0, "my_notes": 0, "my_placed_blocks": 0,
             "waypoints": 0},
            # Stage 5: Ultra-compact fallback stripping verbose sensory/diagnostic dicts
            {"nearby_blocks": 0, "craftable": 0, "entities": 0, "lookup": 0,
             "observed_lessons": 0, "recent_actions": 0, "chat": 1, "memory_events": 0,
             "knowledge": 0, "remembered_resources": 0, "my_notes": 0, "my_placed_blocks": 0,
             "waypoints": 0},
        ]

        system_prompt = SYSTEM_COMPACT if self.prompt_tokens < self.compact_threshold_tokens else SYSTEM
        for stage_idx, minimums in enumerate(stages):
            if stage_idx == 1:
                if "crafting" in context and isinstance(context["crafting"], dict):
                    context["crafting"].pop("knowledge", None)
                if "diagnosis" in context and isinstance(context["diagnosis"], dict):
                    context["diagnosis"].pop("risk_policy", None)
                if "recent_actions" in context and isinstance(context["recent_actions"], list):
                    for act in context["recent_actions"]:
                        res = act.get("result")
                        if isinstance(res, dict):
                            res.pop("id", None)
                            if "detail" in res and isinstance(res["detail"], str) and len(res["detail"]) > 60:
                                res["detail"] = res["detail"][:60]
                if "last_result" in context and isinstance(context["last_result"], dict):
                    det = context["last_result"].get("detail")
                    if isinstance(det, str) and len(det) > 80:
                        context["last_result"]["detail"] = det[:80]
                if "strategy" in context and isinstance(context["strategy"], dict):
                    reason = context["strategy"].get("reason")
                    if isinstance(reason, str) and len(reason) > 60:
                        context["strategy"]["reason"] = reason[:60]
                if "terrain" in context and isinstance(context["terrain"], dict) and "exits" in context["terrain"]:
                    context["terrain"]["exits"] = [e for e in context["terrain"]["exits"] if e.get("standable")]
            elif stage_idx == 2:
                if "milestones" in context and isinstance(context["milestones"], dict):
                    context["milestones"].pop("reminder", None)
                if "terrain" in context and isinstance(context["terrain"], dict):
                    context["terrain"].pop("exits", None)
                if "active_tools" in context and isinstance(context["active_tools"], dict):
                    context["active_tools"].get("vision", {}).pop("insight", None)
                    context["active_tools"].get("hearing", {}).pop("insight", None)
                    context["active_tools"].pop("omni_synthesis", None)
                context.pop("strategy", None)
                context.pop("my_placed_totals", None)
            elif stage_idx == 3:
                if "active_tools" in context and isinstance(context["active_tools"], dict):
                    context["active_tools"] = {
                        "vision": "online",
                        "hearing": "online",
                        "actions": context["active_tools"].get("actions", []),
                    }
                context.pop("visible_tree", None)
                if self.prompt_tokens < self.compact_threshold_tokens:
                    system_prompt = SYSTEM_COMPACT
            elif stage_idx == 4:
                context.pop("active_tools", None)
                context.pop("visible_tree", None)
                system_prompt = SYSTEM_COMPACT
            elif stage_idx == 5:
                for non_essential in ("seeing", "sight", "listening", "hearing", "doing", "lfm_omni_synthesis",
                                      "terrain", "diagnosis", "milestones", "crafting", "placeable", "last_result",
                                      "lookup", "blocked_actions", "inventory_change", "strategist_status",
                                      "smeltable", "trades"):
                    context.pop(non_essential, None)
                system_prompt = SYSTEM_COMPACT

            while True:
                # Cache-friendly prompt ordering: static instructions/rules first, volatile observation last
                user_content = compact(context)
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}]
                current_size = len(compact(messages).encode()) + schema_bytes
                if current_size <= target_budget:
                    return messages
                overflow = current_size - target_budget
                candidates = [key for key, minimum in minimums.items()
                              if isinstance(context.get(key), list) and len(context[key]) > minimum]
                if not candidates:
                    break
                candidate_sizes = {name: len(compact(context[name])) for name in candidates}
                key = max(candidates, key=lambda name: candidate_sizes[name])
                key_len = len(context[key])
                min_len = minimums[key]
                max_removable = key_len - min_len
                avg_item_size = max(1, candidate_sizes[key] // max(1, key_len))
                items_to_drop = max(1, min(max_removable, (overflow + avg_item_size - 1) // avg_item_size))
                context[key] = context[key][:-items_to_drop]

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": compact(context)}]
        if len(compact(messages).encode()) + schema_bytes <= target_budget:
            return messages
        raise ValueError("Essential observation exceeds context budget")

    def _augment_tree_and_plan(self, decision, observation, from_actor=False):
        """Prevent circling/exploring when a tree is in sight and chain follow-up steps for continuous wood gathering."""
        if observation.get("chat") or observation.get("diagnosis", {}).get("stuck"):
            return decision
        counts = inventory(observation)
        logs = sum(v for k, v in counts.items() if k.endswith(("_log", "_stem")))
        planks = sum(v for k, v in counts.items() if k.endswith("_planks"))
        pickaxes = sum(v for k, v in counts.items() if k.endswith("_pickaxe"))
        # Need >= 12 plank-equivalents (3 logs) to craft a crafting table (4), sticks (2), and wooden pickaxe (3)
        needs_wood = (logs * 4 + planks < 12 and pickaxes == 0) or (logs == 0 and planks < 4)
        nearby_logs = [
            b for b in observation.get("nearby_blocks", [])
            if isinstance(b, dict) and isinstance(b.get("id"), str)
            and b["id"].endswith(("_log", "_stem"))
            and isinstance(b.get("at"), list) and len(b["at"]) == 3
            and b.get("reachable", True)
            and not blocked(observation, "mine", b["id"])
        ]
        here = observation.get("position")
        dest = [decision.get("x", 0), decision.get("y", 0), decision.get("z", 0)]
        goto_near_tree = (
            decision["action"] == "goto" and nearby_logs
            and isinstance(here, list) and len(here) == 3
            and (max(abs(nearby_logs[0]["at"][i] - here[i]) for i in range(3)) <= 6
                 or any(max(abs(b["at"][i] - dest[i]) for i in range(3)) <= 2 for b in nearby_logs))
        )
        # 1. If actor chose explore/wait/jump/sneak (or goto near a visible tree) while wood is needed, mine the log directly
        if (from_actor and (decision["action"] in ("explore", "wait", "jump", "sneak") or goto_near_tree)
                and needs_wood and nearby_logs and "mine" in allowed_actions(observation)):
            first = nearby_logs[0]
            candidate = dict(
                decision,
                action="mine",
                target=first["id"],
                x=int(first["at"][0]),
                y=int(first["at"][1]),
                z=int(first["at"][2]),
            )
            try:
                decision = validate_action(candidate, observation)
                LOG.info("Redirected %s to mine visible tree log %s at %s",
                         decision["action"], first["id"], first["at"])
            except (ValueError, TypeError, KeyError):
                pass
        # 2. If wooden pickaxe is owned and cobblestone < 3, redirect idle/goto near stone to mine stone directly
        stone_picks = counts.get("minecraft:stone_pickaxe", 0) + counts.get("minecraft:iron_pickaxe", 0)
        cobble = counts.get("minecraft:cobblestone", 0)
        nearby_stone = [
            b for b in observation.get("nearby_blocks", [])
            if isinstance(b, dict) and b.get("id") in ("minecraft:stone", "minecraft:cobblestone")
            and isinstance(b.get("at"), list) and len(b["at"]) == 3
            and b.get("reachable", True)
            and not blocked(observation, "mine", b["id"])
        ]
        goto_near_stone = (
            decision["action"] == "goto" and nearby_stone
            and isinstance(here, list) and len(here) == 3
            and (max(abs(nearby_stone[0]["at"][i] - here[i]) for i in range(3)) <= 6
                 or any(max(abs(b["at"][i] - dest[i]) for i in range(3)) <= 2 for b in nearby_stone))
        )
        if (from_actor and (decision["action"] in ("wait", "jump", "sneak") or goto_near_stone)
                and pickaxes > 0 and stone_picks == 0 and cobble < 3
                and nearby_stone and "mine" in allowed_actions(observation)):
            first_stone = nearby_stone[0]
            candidate = dict(
                decision,
                action="mine",
                target=first_stone["id"],
                x=int(first_stone["at"][0]),
                y=int(first_stone["at"][1]),
                z=int(first_stone["at"][2]),
            )
            try:
                decision = validate_action(candidate, observation)
                LOG.info("Redirected %s to mine visible %s at %s for stone progression",
                         decision["action"], first_stone["id"], first_stone["at"])
            except (ValueError, TypeError, KeyError):
                pass
        # 3. If mining a tree log without a follow-up plan, chain up to 6 follow-up trunk logs
        if (from_actor and decision["action"] == "mine" and decision["target"].endswith(("_log", "_stem"))
                and not decision.get("plan") and len(nearby_logs) > 1):
            cur_pos = [decision[k] for k in "xyz"] if all(k in decision and decision[k] is not None for k in "xyz") else None
            max_chain = min(6, DECISION_LIMITS["plan"])
            chained = []
            for blk in nearby_logs:
                if cur_pos is not None and blk.get("at") == cur_pos:
                    continue
                chained.append({
                    "action": "mine",
                    "target": blk["id"],
                    "x": int(blk["at"][0]),
                    "y": int(blk["at"][1]),
                    "z": int(blk["at"][2]),
                })
                if len(chained) >= max_chain:
                    break
            if chained:
                decision["plan"] = chained
        return decision

    def decide(self, observation):
        with self._state_lock:
            return self._decide_locked(observation)

    def _decide_locked(self, observation):
        observation = self.perceive(observation)
        reflex = self.pipeline.check_reflex(observation) if hasattr(self, "pipeline") and self.pipeline else None
        if reflex:
            reflex_id = str(uuid.uuid4())
            reflex["id"] = reflex_id
            reflex.setdefault("target", "")
            reflex.setdefault("x", 0)
            reflex.setdefault("y", 0)
            reflex.setdefault("z", 0)
            reflex.setdefault("say", "")
            reflex.setdefault("memory", "")
            reflex.setdefault("waypoint", "")
            try:
                reflex = validate_action(reflex, observation)
                LOG.warning("Immediate sensory emergency reflex triggered: %s", reflex.get("goal"))
                with self.db:
                    self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                                    (reflex_id, time.time(), reflex["action"], reflex["target"],
                                     reflex["goal"], reflex["memory"]))
                    self.experience.remember(reflex_id, reflex["action"], reflex["target"], observation)
                self._export_status(observation, reflex, "reflex", 0.0)
                return reflex
            except (ValueError, TypeError, KeyError) as exc:
                LOG.warning("Emergency reflex action %s was invalid (%s); falling back to cognitive cycle", reflex, exc)
        self.ensure_profile(timeout=self.actor_timeout)
        self.update_planner(observation)
        # A plan rejected on the previous request (HTTP 422) executed nothing, so last_result is unchanged;
        # remind the model why, otherwise a low-temperature model tends to repeat the same invalid answer.
        correction = ""
        retry_heat = 0.0
        if self.last_rejection and observation.get("last_result", {}).get("id") == self.last_rejection_result_id:
            correction = (f"Your previous plan was rejected and NOT executed: {self.last_rejection[:160]}. "
                          "Choose a different valid action from the CURRENT observation.")
            retry_heat = 0.3
        messages = self.messages(observation, correction=correction, role="actor")
        started = time.monotonic()
        decision = None
        skill_id = None
        source = "actor"
        limits = self.gameplay_limits
        danger_food = limits["danger_food"]
        danger_health = limits["danger_health"]
        danger_dist = limits["danger_hostile_distance"]
        if self.steps:
            last = observation.get("last_result", {})
            cur_food = observation.get("food", 20)
            cur_health = observation.get("health", 20)
            food_dropped = cur_food < min(danger_food, self.steps_food)
            health_dropped = cur_health <= 6 or cur_health < min(danger_health, self.steps_health)
            if (observation.get("chat") or observation.get("diagnosis", {}).get("stuck")
                    or food_dropped or health_dropped
                    or any(entity.get("hostile") and entity.get("distance", 99) < danger_dist
                           for entity in observation.get("entities", []))
                    or last.get("id") != self.steps_last_id or last.get("status") != "success"
                    or self.steps_dimension != observation.get("dimension")
                    or time.monotonic() > self.steps_deadline):
                LOG.info("Plan interrupted by outcome, changed situation, danger or player input")
                self.steps.clear()
            else:
                candidate = dict(self.steps.pop(0), goal=self.steps_goal, say="", memory="")
                try:
                    decision = validate_action(candidate, observation)
                    source = "plan"
                except (ValueError, TypeError, KeyError) as exc:
                    LOG.info("Plan precondition changed; replanning: %s", exc)
                    self.steps.clear()
        # Wire verified V2 skills / Q-fast-path execution via step-queue adapter over WillowBotActions
        if (decision is None and hasattr(self, "orchestrator") and self.orchestrator
                and not observation.get("chat") and not observation.get("diagnosis", {}).get("stuck")
                and observation.get("last_result", {}).get("status") != "failed"
                and observation.get("food", 20) >= danger_food and observation.get("health", 20) >= danger_health
                and not any(entity.get("hostile") and entity.get("distance", 99) < danger_dist
                            for entity in observation.get("entities", []))):
            try:
                snap = BotTelemetrySnapshot.from_observation(observation)
                ctx_hash = ContextStateHash.encode(snap)
                next_task = self.orchestrator.curriculum.get_next_objective(snap)
                verified_candidates = self.orchestrator.skill_store.get_verified_skills_for_task(next_task.id, snap)
                fast_skill_id, fast_q = self.orchestrator.q_table.select_fast_path_skill(
                    ctx_hash,
                    next_task.id,
                    [r.skill_id for r in verified_candidates],
                )
                if fast_skill_id is None and verified_candidates:
                    cand_rec = verified_candidates[0]
                    if cand_rec.q_value >= 0.0:
                        fast_skill_id = cand_rec.skill_id

                if fast_skill_id:
                    skill_code = self.orchestrator.skill_store.load_code(fast_skill_id)
                    from execution.bot_actions import WillowBotActions
                    bot_sim = WillowBotActions(snap, pool=self.orchestrator.memory_pool)
                    outcome = self.orchestrator.sandbox.execute_sync(
                        skill_code,
                        bot_sim,
                        skill_name=fast_skill_id,
                        timeout_ms=3000,
                    )
                    if outcome.action_result.Success and bot_sim.emitted_steps:
                        first_step = dict(bot_sim.emitted_steps[0], goal=next_task.description or fast_skill_id, say="", memory="")
                        valid_first = validate_action(first_step, observation)
                        self.steps = list(bot_sim.emitted_steps[1:])
                        self.steps_goal = next_task.description or fast_skill_id
                        self.steps_last_id = observation.get("last_result", {}).get("id")
                        self.steps_food = observation.get("food", 20)
                        self.steps_health = observation.get("health", 20)
                        self.steps_dimension = observation.get("dimension")
                        self.steps_deadline = time.monotonic() + 30.0
                        decision = valid_first
                        skill_id = fast_skill_id
                        source = f"v2_skill:{fast_skill_id}"
            except Exception as exc:
                LOG.debug("V2 fast-path / skill execution skipped: %s", exc)
        if (decision is None and not observation.get("chat") and not observation.get("diagnosis", {}).get("stuck")
                and observation.get("last_result", {}).get("status") != "failed"
                and observation.get("food", 20) >= danger_food and observation.get("health", 20) >= danger_health
                and not any(entity.get("hostile") and entity.get("distance", 99) < danger_dist
                            for entity in observation.get("entities", []))
                and time.monotonic() >= self.skill_hold_until
                and self.consecutive_skill_count < limits["skill_loop_consecutive"]
                and self.consecutive_skill_no_progress < limits["skill_loop_no_progress"]):
            for key, candidate in self.skills.candidates(observation):
                if blocked(observation, candidate["action"], candidate["target"]):
                    continue
                try:
                    validate_action(candidate, observation)
                except (ValueError, TypeError, KeyError) as exc:
                    self.skills.reject(key, str(exc))
                    continue
                if candidate["action"] in ("explore", "wait") and self.consecutive_skill_no_progress >= 2:
                    if self.consecutive_skill_id == key or self.consecutive_skill_action in ("explore", "wait"):
                        self.skills.reject(key, "Repeated exploration loop without resource progress")
                    continue
                skill_id, decision = key, candidate
                saved = self.db.execute("SELECT value FROM state WHERE key='goal'").fetchone()
                if saved:
                    decision["goal"] = saved[0]
                source = key
                break
        if decision is None:
            raw = None
            image = (observation.get("sight", {}).get("snapshot_png_base64")
                     or observation.get("seeing", {}).get("snapshot_png_base64"))
            images = [image] if image else None
            for attempt in range(2):
                try:
                    remaining = self.actor_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Actor correction deadline exceeded")
                    raw = self.inference(
                        messages, action_schema(observation), "actor", timeout=remaining, images=images,
                        temperature_boost=retry_heat + (0.3 if attempt else 0.0))
                    decision = validate_action(raw, observation)
                    decision = self._augment_tree_and_plan(decision, observation, from_actor=True)
                    break
                except (ValueError, TypeError, KeyError) as exc:
                    if attempt and isinstance(raw, dict) and raw.get("action") == "goto":
                        # Recover from repeated/adjacent goto rejection by promoting a valid plan step or exploring
                        recovered = None
                        for idx, step in enumerate(raw.get("plan") or []):
                            if isinstance(step, dict):
                                try:
                                    cand = dict(raw, action=step.get("action", ""), target=step.get("target", ""),
                                                x=step.get("x", 0), y=step.get("y", 0), z=step.get("z", 0),
                                                plan=list(raw.get("plan")[idx + 1:]))
                                    recovered = validate_action(cand, observation)
                                    break
                                except (ValueError, TypeError, KeyError):
                                    continue
                        if recovered is None and "explore" in allowed_actions(observation):
                            try:
                                recovered = validate_action(
                                    dict(raw, action="explore", target="", x=0, y=0, z=0, plan=[]), observation)
                            except (ValueError, TypeError, KeyError):
                                pass
                        if recovered is not None:
                            decision = self._augment_tree_and_plan(recovered, observation, from_actor=True)
                            LOG.info("Recovered rejected goto into %s %s", decision["action"], decision["target"])
                            break
                    if attempt:
                        LOG.warning("Model plan rejected; no action executed: %s (raw response: %s)", exc, raw)
                        self.last_rejection = str(exc)
                        self.last_rejection_result_id = observation.get("last_result", {}).get("id")
                        raise DecisionRejected(str(exc)) from exc
                    LOG.warning("Rejected actor response; requesting one correction: %s (raw response: %s)", exc, raw)
                    messages = self.messages(observation, correction=(
                        f"Previous response rejected: {str(exc)[:160]}. Choose a valid action from the CURRENT "
                        "observation. Use only unblocked targets. For placement use an owned placeable block "
                        "at one of crafting.placement_sites. No action was executed."), role="actor")
        if decision.get("plan"):
            self.steps = list(decision["plan"])
            self.steps_goal = decision["goal"]
            self.steps_dimension = observation.get("dimension")
            self.steps_deadline = time.monotonic() + self.steps_deadline_seconds
            self.steps_health = observation.get("health", 20)
            self.steps_food = observation.get("food", 20)
        self.last_rejection = ""
        self.last_rejection_result_id = None
        if skill_id:
            self.consecutive_skill_count += 1
            if self.consecutive_skill_id != skill_id:
                self.consecutive_skill_id = skill_id
                self.consecutive_skill_no_progress = 0
            self.consecutive_skill_action = decision["action"]
        else:
            self.consecutive_skill_id = None
            self.consecutive_skill_action = None
            self.consecutive_skill_count = 0
            self.consecutive_skill_no_progress = 0
        speech = decision["say"]
        if observation.get("chat"):
            LOG.info("Processing observation with chat: %s", observation["chat"])
            if not speech or not speech.strip():
                goal = decision.get("goal", "").strip()
                action = decision.get("action", "")
                target = decision.get("target", "")
                if goal:
                    speech = f"On it! {goal.rstrip('.')}."
                elif action in ("mine", "craft", "place", "pickup") and target:
                    clean_target = target.split(":")[-1].replace("_", " ")
                    speech = f"On it, going to {action} {clean_target}."
                else:
                    speech = f"I hear you! Going to {action}."
                decision["say"] = speech
            self.last_speech, self.last_speech_time = speech, time.monotonic()
            LOG.info("Willow chat reply to player: \"%s\"", speech)
        elif speech and speech == self.last_speech and time.monotonic() - self.last_speech_time < 60:
            decision["say"] = ""
        elif speech:
            self.last_speech, self.last_speech_time = speech, time.monotonic()
        elif not speech and time.monotonic() - self.last_speech_time > 25:
            alerts = observation.get("hearing", {}).get("active_alerts", [])
            hazards = observation.get("seeing", {}).get("visual_hazards", [])
            looking = observation.get("seeing", {}).get("looking_at", {}).get("id")
            if alerts:
                sound_name = alerts[0].get("sound", "").split(":")[-1]
                bearing = alerts[0].get("bearing", "nearby")
                speech = f"Heard {sound_name} {bearing}."
            elif hazards:
                speech = f"Watch out: {hazards[0]}."
            elif looking and looking not in ("minecraft:air", "air") and decision.get("action") == "mine":
                speech = f"Mining observed {looking.split(':')[-1]}."
            if speech:
                decision["say"] = speech
                self.last_speech, self.last_speech_time = speech, time.monotonic()
        decision_id = str(uuid.uuid4())
        with self.db:
            self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                            (decision_id, time.time(), decision["action"], stored_target(decision),
                             decision["goal"], decision["memory"]))
            self.db.execute("INSERT OR REPLACE INTO state VALUES ('goal',?)", (decision["goal"],))
            self.experience.remember(decision_id, observation)
            self.db.execute(
                "DELETE FROM decisions WHERE id NOT IN "
                "(SELECT id FROM decisions ORDER BY created DESC LIMIT ?)",
                (self.decisions_cap,),
            )
            if skill_id:
                self.db.execute("INSERT INTO decision_skills VALUES (?,?)", (decision_id, skill_id))
            self.db.execute("DELETE FROM decision_skills WHERE decision_id NOT IN (SELECT id FROM decisions)")
        with self.db:
            if decision["memory"]:
                self.memory.note(decision["memory"], observation)
            if decision["waypoint"] and self.memory.save_waypoint(decision["waypoint"], observation, decision["goal"][:DECISION_LIMITS["goal"]]):
                LOG.info("Saved waypoint %s at %s", decision["waypoint"], observation.get("position"))
        self.decisions_since_export += 1
        if self.decisions_since_export >= 10:
            self.decisions_since_export = 0
            try:
                self.memory.export()
            except OSError as exc:
                LOG.warning("Could not export memory.json: %s", exc)
        decision["id"] = decision_id
        decision["watch"] = self.memory.watch_list(observation)
        # Tells the mod whether rendering the PNG snapshot is worthwhile for this model/profile.
        decision["vision"] = bool((self.profile or {}).get("vision_enabled"))
        self.steps_last_id = decision_id
        elapsed = time.monotonic() - started
        LOG.info("decision=%s action=%s target=%s source=%s elapsed=%.2fs",
                 decision_id, decision["action"], decision["target"],
                 source, elapsed)
        self._export_status(observation, decision, source, elapsed)
        self.schedule_planner(observation, messages)
        return decision


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/decide":
            self.send_error(404)
            return
        if not hmac.compare_digest(self.headers.get("Authorization", ""),
                                   "Bearer " + self.server.token):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req_limit = getattr(self.server.brain, "request_limit_bytes", GAMEPLAY_LIMITS["request_limit_bytes"])
            if not 0 < length <= req_limit:
                self.send_error(413)
                return
            self.connection.settimeout(10.0)
            observation = json.loads(self.rfile.read(length))
            act_timeout = getattr(self.server.brain, "actor_timeout", ACTOR_TIMEOUT)
            self.connection.settimeout(act_timeout + 30)
            if not isinstance(observation, dict):
                raise ValueError("Expected an observation object")
            decision = self.server.brain.decide(observation)
            body = compact(decision).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except DecisionRejected as exc:
            LOG.warning("Model plan rejected; no action executed: %s", exc)
            body = compact({"error": "decision_rejected", "detail": str(exc)[:240]}).encode()
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
            LOG.exception("Planning request failed")
            self.send_error(503, "Local planner failed; consult brain.log")

    def log_message(self, fmt, *args):
        LOG.info(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--port-file", required=True, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--settings", default="{}", help="JSON per-model overrides from the settings screen")
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler(args.database.parent / "brain.log",
                                                      maxBytes=2_000_000, backupCount=2)])
    token = os.environ["LOCALAI_TOKEN"]
    if len(token) < 32:
        raise ValueError("Missing/weak local bridge token")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 5
    server.token = token
    server.brain = Brain(args.database, model=args.model, settings=json.loads(args.settings))
    # Publish port file immediately so BrainBridge.java (5s poll timeout) connects before cold model benchmark
    temporary = args.port_file.with_suffix(".tmp")
    temporary.write_text(str(server.server_port))
    temporary.replace(args.port_file)
    LOG.info("Local brain listening on 127.0.0.1:%d, model=%s", server.server_port, args.model)
    server.brain.autotune(run_benchmark=True)
    try:
        server.serve_forever()
    finally:
        server.brain.close()
        server.server_close()
        args.port_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
