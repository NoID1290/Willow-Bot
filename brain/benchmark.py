#!/usr/bin/env python3
"""Repeatable decision-quality/latency smoke benchmark, not a gameplay score."""
import argparse
import json
import logging
from pathlib import Path
import tempfile
import time
import urllib.request

from server import Brain, MODEL, CONTEXT, OUTPUT_TOKENS


def resource(name, x=3, y=65, z=1):
    return {"id": "minecraft:" + name, "at": [x, y, z]}


CASES = [
    ("start_without_supplies",
     {"nearby_blocks": [resource("oak_log"), resource("stone")], "inventory": []},
     "mine", "minecraft:oak_log"),
    ("craft_first_table",
     {"inventory": ["minecraft:oak_planks=4"], "craftable": ["minecraft:crafting_table"],
      "chat": ["Craft a crafting table now."]},
     "craft", "minecraft:crafting_table"),
    ("craft_pickaxe",
     {"inventory": ["minecraft:oak_planks=3", "minecraft:stick=2"],
      "nearby_blocks": [resource("crafting_table"), resource("stone")],
      "craftable": ["minecraft:wooden_pickaxe"], "chat": ["Make a wooden pickaxe."]},
     "craft", "minecraft:wooden_pickaxe"),
    ("eat_when_hungry",
     {"food": 6, "health": 12, "inventory": ["minecraft:bread=2"], "nearby_blocks": [resource("oak_log")]},
     "eat", None),
    ("reply_to_player",
     {"inventory": [], "nearby_blocks": [resource("oak_log")], "chat": ["Hi Willow, what are you doing?"]},
     None, None),
    ("escape_circular_tree_loop",
     {"position": [0, 80, 0], "inventory": [],
      "nearby_blocks": [resource("oak_leaves", 0, 79, 0)],
      "diagnosis": {"stuck": True, "reason": "Eight actions circling on a tree",
                    "recovery_attempts": 0}},
     "escape", None),
]


def benchmark(model):
    records = []
    for name, changes, expected_action, expected_target in CASES:
        observation = {"position": [0, 65, 0], "health": 20, "food": 20, "time": 1000,
                       "inventory": [], "nearby_blocks": [], "craftable": [], "entities": [],
                       "chat": [], "last_result": {}}
        observation.update(changes)
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "brain.sqlite3", model=model, planning=False)
            started = time.monotonic()
            try:
                result = brain.decide(observation)
                passed = ((expected_action is None or result["action"] == expected_action)
                          and (expected_target is None or result["target"] == expected_target)
                          and (name != "reply_to_player" or bool(result["say"])))
                records.append({"case": name, "passed": passed,
                                "seconds": round(time.monotonic() - started, 2), "decision": result})
            except (ValueError, KeyError, OSError) as exc:
                records.append({"case": name, "passed": False, "error": str(exc)})
            finally:
                brain.close()
    with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as response:
        residency = json.load(response)
    return {"model": model, "context": CONTEXT, "output_tokens": OUTPUT_TOKENS,
            "passed": sum(r["passed"] for r in records), "total": len(records),
            "cases": records, "ollama_residency": residency}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    results = benchmark(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
