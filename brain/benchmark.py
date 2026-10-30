#!/usr/bin/env python3
"""Repeatable decision-quality/latency benchmark, supporting full-loop mode (actor + strategist concurrent)."""
import argparse
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

from server import Brain, MODEL, CONTEXT, OUTPUT_TOKENS, DecisionRejected


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


def fetch_ollama_vram_and_residency(ollama="http://127.0.0.1:11434"):
    """Query /api/ps and optionally CLI ollama ps to measure live VRAM footprint."""
    residency = {"models": []}
    try:
        with urllib.request.urlopen(f"{ollama}/api/ps", timeout=4) as response:
            residency = json.load(response)
    except Exception:
        pass

    cli_ps = ""
    for cmd in ["ollama ps", "/run/host/usr/local/bin/ollama ps", "/usr/local/bin/ollama ps"]:
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                cli_ps = res.stdout.strip()
                break
        except Exception:
            continue

    total_vram_mib = 0
    for m in residency.get("models", []):
        size_v = m.get("size_vram") or m.get("size") or 0
        total_vram_mib += int(round(size_v / (1024 ** 2)))

    return residency, cli_ps, total_vram_mib


def benchmark(model, full_loop=False, settings=None, turns_per_case=2):
    records = []
    invalid_actions_count = 0
    rejected_actions_count = 0
    invalid_skills_count = 0

    with tempfile.TemporaryDirectory() as directory:
        brain = Brain(Path(directory) / "brain.sqlite3", model=model, planning=full_loop, settings=settings)
        try:
            for name, changes, expected_action, expected_target in CASES:
                for turn in range(turns_per_case):
                    observation = {
                        "position": [0, 65, 0], "health": 20, "food": 20, "time": 1000 + turn * 20,
                        "inventory": [], "nearby_blocks": [], "craftable": [], "entities": [],
                        "chat": [], "last_result": {}
                    }
                    observation.update(changes)
                    started = time.monotonic()
                    err = None
                    result = None
                    try:
                        result = brain.decide(observation)
                        passed = (
                            (expected_action is None or result["action"] == expected_action)
                            and (expected_target is None or result["target"] == expected_target)
                            and (name != "reply_to_player" or bool(result.get("say")))
                        )
                    except DecisionRejected as exc:
                        rejected_actions_count += 1
                        passed = False
                        err = f"DecisionRejected: {exc}"
                    except (ValueError, KeyError, TypeError) as exc:
                        invalid_actions_count += 1
                        passed = False
                        err = f"InvalidAction: {exc}"
                    except OSError as exc:
                        passed = False
                        err = f"OSError: {exc}"

                    elapsed = round(time.monotonic() - started, 3)

                    # Extract live metrics from profile/Ollama response if available
                    prof = brain.profile or {}
                    prompt_tok = prof.get("prompt_tokens")
                    gen_tok = prof.get("actor_tokens")
                    prompt_tps = prof.get("prompt_tps")
                    gen_tps = prof.get("gen_tps")

                    rec = {
                        "case": f"{name}_t{turn}",
                        "passed": passed,
                        "seconds": elapsed,
                        "prompt_tokens": prompt_tok,
                        "generated_tokens": gen_tok,
                        "prompt_tps": prompt_tps,
                        "gen_tps": gen_tps,
                        "decision": result,
                    }
                    if err:
                        rec["error"] = err
                    records.append(rec)

                    # In full_loop, give background strategist brief CPU slice to advance
                    if full_loop:
                        time.sleep(0.1)

            # Check invalid skills count in library
            try:
                invalid_skills_count = brain.db.execute(
                    "SELECT count(*) FROM skills WHERE disabled=1"
                ).fetchone()[0]
            except Exception:
                pass

        finally:
            brain.close()

    residency, cli_ps, vram_mib = fetch_ollama_vram_and_residency()

    passed_count = sum(1 for r in records if r["passed"])
    latencies = [r["seconds"] for r in records if "seconds" in r]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    warm_avg_latency = round(sum(warm_latencies) / len(warm_latencies), 3) if warm_latencies else 0.0

    return {
        "model": model,
        "full_loop": full_loop,
        "context": (settings or {}).get("contextLength", CONTEXT),
        "output_tokens": OUTPUT_TOKENS,
        "passed": passed_count,
        "total": len(records),
        "avg_latency_s": avg_latency,
        "warm_avg_latency_s": warm_avg_latency,
        "actor_under_3s_warm": warm_avg_latency <= 3.0,
        "invalid_actions_count": invalid_actions_count,
        "rejected_actions_count": rejected_actions_count,
        "invalid_skills_count": invalid_skills_count,
        "vram_used_mib": vram_mib,
        "ollama_ps_output": cli_ps,
        "ollama_residency": residency,
        "cases": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-loop", action="store_true", help="Run full loop with background strategist active")
    parser.add_argument("--context", type=int, help="Override context length")
    parser.add_argument("--actor-tokens", type=int, help="Override actor output tokens")
    parser.add_argument("--plan-tokens", type=int, help="Override strategist output tokens")
    parser.add_argument("--think", choices=["auto", "on", "off"], help="Override thinking")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    settings = {}
    if args.context:
        settings["contextLength"] = args.context
    if args.actor_tokens:
        settings["actorTokens"] = args.actor_tokens
    if args.plan_tokens:
        settings["planTokens"] = args.plan_tokens
    if args.think:
        settings["think"] = args.think

    results = benchmark(args.model, full_loop=args.full_loop, settings=settings or None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
