import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import monitor
import server


class MonitorConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log_path = self.dir / "brain.log"

    def tearDown(self):
        self.tmp.cleanup()

    def test_role_inference_parses_cached_tokens_and_strips_log_level(self):
        c = monitor.Console(self.log_path, color=False)
        line = (
            "2026-11-10 17:55:54,321 INFO role=actor prompt_tokens=12111 cached=6055 "
            "generated_tokens=109 elapsed=2.28s prompt_tps=8284.2 gen_tps=151.7"
        )
        out = c.format(line)
        self.assertIsNotNone(out)
        self.assertNotIn("INFO role=", out)
        self.assertIn("17:55:54", out)
        self.assertIn("ACTOR", out)
        self.assertIn("12,111→109 tok", out)
        self.assertIn("(50% KV)", out)
        self.assertIn("151.7 tok/s", out)
        self.assertEqual(len(c.calls), 1)
        self.assertAlmostEqual(c.calls[0][1], 151.7)
        self.assertEqual(c.calls[0][4], 12111)
        self.assertEqual(c.calls[0][5], 6055)

    def test_critic_and_rl_reward_fuse_into_single_clean_row(self):
        c = monitor.Console(self.log_path, color=False)
        emitted = []
        c.write = emitted.append

        c.emit("2026-11-10 17:56:27,050 INFO [CRITIC] FAIL task=t2_stone_tools_sub2 category=MISSING_PREREQUISITE (0ms GPU)")
        self.assertEqual(emitted, [])
        c.emit(
            "2026-11-10 17:56:27,052 INFO [RL-REWARD] task=t2_stone_tools_sub2 total=-0.08 "
            "(adv=+0.0 task=+0.0 shape=+0.0 intr=+0.0 | vital=-0.0 exec=-0.0 time=-0.08)"
        )
        self.assertEqual(len(emitted), 1)
        self.assertIn("EVAL", emitted[0])
        self.assertIn("t2_stone_tools_sub2", emitted[0])
        self.assertIn("prereq", emitted[0])
        self.assertIn("time -0.08", emitted[0])
        self.assertNotIn("adv=+0.0", emitted[0])
        self.assertEqual(c.critic_steps, 1)

        # Positive task completion fuses as REWARD
        c.emit("2026-11-10 17:56:30,010 INFO [CRITIC] PASS task=t2_stone_tools_sub2 category=NONE (0ms GPU)")
        c.emit(
            "2026-11-10 17:56:30,012 INFO [RL-REWARD] task=t2_stone_tools_sub2 total=+26.92 "
            "(adv=+0.0 task=+25.0 shape=+2.0 intr=+0.0 | vital=-0.0 exec=-0.0 time=-0.08)"
        )
        self.assertEqual(len(emitted), 2)
        self.assertIn("REWARD", emitted[1])
        self.assertIn("+26.92", emitted[1])
        self.assertIn("task +25", emitted[1])
        self.assertIn("shape +2", emitted[1])
        self.assertEqual(c.critic_passes, 1)
        self.assertAlmostEqual(c.rl_total, 26.84, places=2)

    def test_decision_and_curriculum_dedup_and_rejection_cleanup(self):
        c = monitor.Console(self.log_path, color=False)
        act_out = c.format("2026-11-10 17:56:29,210 INFO decision=abc action=pickup target=minecraft:cobblestone source=actor elapsed=2.17s")
        self.assertIn("ACTION", act_out)
        self.assertIn("pickup", act_out)
        self.assertIn("cobblestone", act_out)
        self.assertNotIn("minecraft:", act_out)

        # Consecutive identical curriculum selections are deduplicated
        cur_line = "2026-11-10 17:56:30,000 INFO [CURRICULUM] Selected objective 't2_stone_tools' (Tier 2, Reward +25, PrecondOK=True)"
        first = c.format(cur_line)
        second = c.format(cur_line)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

        # Raw JSON dump in actor rejection is stripped from live view
        rej = c.format('2026-11-10 17:56:31,000 WARNING Rejected actor response; requesting one correction: Invalid target (raw response: {"action":"mine"})')
        self.assertIn("REJECT", rej)
        self.assertIn("Invalid target", rej)
        self.assertNotIn('{"action"', rej)

        # HTTP /decide access log is suppressed
        self.assertIsNone(c.format('2026-11-10 17:56:31,100 INFO "POST /decide HTTP/1.1" 200 -'))

    def test_hud_exact_widths_and_column_alignment(self):
        (self.dir / "model-profile.json").write_text(json.dumps({
            "model": "lfm3-all:latest",
            "architecture": "gemma4",
            "parameter_size": "4.6B",
            "quantization": "Q4_K_M",
            "context": 16384,
            "trained_context": 131072,
            "thinking": False,
            "actor_tokens": 1024,
            "plan_tokens": 4096,
            "kv_cache_mib": 224,
            "vision_enabled": True,
        }))
        (self.dir / "status.json").write_text(json.dumps({
            "updated_at": 9999999999,
            "started_at": 9999999900,
            "health": 20.0,
            "max_health": 20.0,
            "food": 18,
            "armor": 6,
            "position": [97, 69, -113],
            "dimension": "minecraft:overworld",
            "time": 6000,
            "weather": "rain",
            "owned_tools": {"minecraft:stone_pickaxe": 1},
            "free_slots": 21,
            "curriculum": {"active_node_id": "t2_stone_tools", "tier": 2, "completed_count": 5},
            "goal": "Smelt raw_iron in furnace",
            "plan_queued": 3,
        }))
        for use_color in (False, True):
            c = monitor.Console(self.log_path, color=use_color)
            c._ollama_cache = {
                "ts": 1e18,
                "ok": True,
                "version": "0.35.0",
                "models": [{"name": "lfm3-all:latest", "size": 3_300_000_000, "size_vram": 3_300_000_000}],
            }
            c.format("2026-11-10 17:56:29,200 INFO role=actor prompt_tokens=11720 cached=9400 generated_tokens=112 elapsed=2.16s prompt_tps=8335.9 gen_tps=163.8")
            for width in (72, 80, 110, 140):
                hud = c.status(columns=width)
                self.assertEqual(len(hud), 7)
                for row in hud:
                    self.assertEqual(monitor.visible_len(row), width)
                plain_bottom = monitor.ANSI_ESCAPE_RE.sub("", hud[-1])
                self.assertEqual(plain_bottom[10], "┼")
                self.assertEqual(plain_bottom[21], "┬")

            plain_stream = monitor.ANSI_ESCAPE_RE.sub(
                "",
                c.format("2026-11-10 17:56:29,210 INFO decision=abc action=mine target=minecraft:stone source=actor elapsed=2.11s"),
            )
            self.assertEqual(plain_stream[10], "│")
            self.assertEqual(plain_stream[21], "│")

    def test_server_exports_status_json(self):
        db_path = self.dir / "brain.sqlite3"
        brain = server.Brain(db_path, planning=False)
        try:
            brain.profile = {
                "model": "qwen3.5:9b",
                "architecture": "qwen35",
                "parameter_size": "9B",
                "quantization": "Q4_K_M",
                "trained_context": 32768,
                "context": 8192,
                "kv_cache_mib": 512,
                "moe_experts": 0,
                "thinking": False,
                "actor_tokens": 512,
                "plan_tokens": 1024,
                "prompt_bytes": 18000,
                "options": {"num_ctx": 8192},
            }
            brain.inference = lambda *a, **kw: {
                "action": "mine",
                "target": "minecraft:oak_log",
                "x": 10,
                "y": 64,
                "z": 10,
                "goal": "Chop oak tree for planks",
                "say": "",
                "memory": "",
                "waypoint": "",
                "plan": [],
            }
            obs = {
                "position": [10, 64, 9],
                "dimension": "minecraft:overworld",
                "health": 19.0,
                "max_health": 20.0,
                "food": 17,
                "armor_points": 2,
                "time": 4000,
                "weather": "clear",
                "inventory": ["minecraft:wooden_pickaxe=1"],
                "nearby_blocks": [{"id": "minecraft:oak_log", "at": [10, 64, 10], "reachable": True}],
                "terrain": {"feet": "minecraft:air", "below": "minecraft:grass_block", "exits": []},
            }
            brain.decide(obs)
            status_file = self.dir / "status.json"
            self.assertTrue(status_file.is_file())
            snap = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(snap["health"], 19.0)
            self.assertEqual(snap["food"], 17)
            self.assertEqual(snap["armor"], 2)
            self.assertEqual(snap["position"], [10, 64, 9])
            self.assertEqual(snap["action"], "mine")
            self.assertIn("curriculum", snap)
        finally:
            brain.close()


if __name__ == "__main__":
    unittest.main()
