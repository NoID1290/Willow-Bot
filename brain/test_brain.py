import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import Brain, CONTEXT, OUTPUT_TOKENS, PROMPT_BYTES, SCHEMA, action_schema, action_targets, allowed_actions, compact, craft_targets, validate_action, validate_decision


def decision(**changes):
    value = dict(action="explore", target="", x=0, y=0, z=0,
                 goal="Find trees", say="", memory="")
    value.update(changes)
    return value


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = Brain(Path(self.tmp.name) / "brain.sqlite3", planning=False)

    def tearDown(self):
        self.brain.close()
        self.tmp.cleanup()

    def test_validation(self):
        self.assertEqual(validate_decision(decision())["action"], "explore")
        for value in (decision(action="shell"), decision(x=True), decision(x=30_000_001),
                      decision(say=5), decision(target="a\nb"), decision(target="a" * 97),
                      decision(extra="field"), [], None):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validate_decision(value)
        trimmed = validate_decision(decision(say="a" * 500, memory="b" * 500))
        self.assertEqual((len(trimmed["say"]), len(trimmed["memory"])), (240, 240),
                         "Long free text is trimmed instead of discarding a valid action")

    def test_target_sanitization_and_plan_resilience(self):
        # Target with coordinates or extra punctuation
        sanitized = validate_decision(decision(action="place", target="minecraft:spruce_log [-666, 85, -31]"))
        self.assertEqual(sanitized["target"], "minecraft:spruce_log")

        # Mixed-case and punctuation
        sanitized = validate_decision(decision(action="place", target="Minecraft:spruce_log."))
        self.assertEqual(sanitized["target"], "minecraft:spruce_log")

        # Missing namespace prefix
        sanitized = validate_decision(decision(action="mine", target="stone"))
        self.assertEqual(sanitized["target"], "minecraft:stone")

        # Attack target extraction
        sanitized = validate_decision(decision(action="attack", target="entity 1024"))
        self.assertEqual(sanitized["target"], "1024")

        # Follow-up steps in plan are sanitized
        step1 = dict(action="place", target="minecraft:spruce_log [-666, 85, -31]", x=-666, y=85, z=-31)
        step2 = dict(action="place", target="", x=0, y=0, z=0)
        res = validate_decision(decision(action="place", target="minecraft:spruce_log", plan=[step1, step2]))
        self.assertEqual(len(res["plan"]), 1)
        self.assertEqual(res["plan"][0]["target"], "minecraft:spruce_log")

    def test_context_is_byte_bounded_and_does_not_mutate_input(self):
        observation = {
            "position": [100, 65, -200], "health": 20, "food": 20,
            "inventory": ["minecraft:oak_log=64"] * 36,
            "nearby_blocks": ["minecraft:oak_log@101,65,-201"] * 200,
            "craftable": ["minecraft:wooden_pickaxe"] * 200,
            "entities": ["12:minecraft:pig@101,65,-201"] * 100,
            "chat": ["Hello, find some wood"], "last_result": {},
        }
        original = copy.deepcopy(observation)
        messages = self.brain.messages(observation)
        self.assertLessEqual(len(compact(messages).encode()) + len(compact(SCHEMA).encode()), PROMPT_BYTES)
        self.assertLessEqual(len(compact(messages).encode()) + len(compact(action_schema(observation)).encode()),
                             PROMPT_BYTES)
        self.assertEqual(original, observation)
        self.assertIn("Hello, find some wood", messages[-1]["content"])
        self.assertEqual(json.loads(messages[-1]["content"])["inventory"], original["inventory"])
        self.assertLessEqual(PROMPT_BYTES + OUTPUT_TOKENS + 256, CONTEXT)

    def test_recent_failed_actions_survive_large_observations(self):
        for i in range(6):
            self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                                 (str(i), i, "mine", "minecraft:spruce_log", "Get wood", "",
                                  compact({"status": "failed", "detail": "Obscured target"})))
        self.brain.db.commit()
        context = json.loads(self.brain.messages({
            "inventory": ["minecraft:oak_log=1"],
            "nearby_blocks": [{"id": "minecraft:stone", "at": [i, 64, 0]} for i in range(200)]
        })[-1]["content"])
        self.assertGreaterEqual(len(context["recent_actions"]), 2)
        self.assertEqual(context["recent_actions"][0]["result"]["status"], "failed")

    def test_oversized_essential_data_is_rejected(self):
        with self.assertRaises(ValueError):
            self.brain.messages({"position": "x" * (PROMPT_BYTES + 1)})

    def test_observed_learning_is_persistent_and_idempotent(self):
        self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                              ("abc", 0, "mine", "minecraft:stone", "Get stone", "I can fly"))
        self.brain.db.commit()
        self.brain.record_result({"last_result": {"id": "abc", "status": "failed", "detail": "No tool"}})
        self.brain.record_result({"last_result": {"id": "abc", "status": "success", "detail": "Invented"}})
        rows = self.brain.db.execute("SELECT text FROM lessons").fetchall()
        self.assertEqual(rows, [("mine minecraft:stone: failed No tool",)])
        other = Brain(Path(self.tmp.name) / "brain.sqlite3", planning=False)
        self.assertEqual(other.db.execute("SELECT count(*) FROM lessons").fetchone()[0], 1)
        other.close()

    def test_model_request_is_local_bounded_and_saves_decision(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, limit):
                return json.dumps({"message": {"content": json.dumps(decision())},
                                   "capabilities": ["completion", "thinking"],
                                   "model_info": {"general.architecture": "gemma4",
                                                  "gemma4.context_length": 262144},
                                   "done_reason": "stop", "prompt_eval_count": 500}).encode()

        with patch("urllib.request.urlopen", return_value=Response()) as send:
            value = self.brain.decide({"health": 20, "food": 20})
        req = send.call_args.args[0]
        payload = json.loads(req.data)
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["options"]["num_ctx"], 32768)
        self.assertEqual(payload["options"]["num_predict"], 8192)
        self.assertTrue(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertIsNotNone(self.brain.db.execute("SELECT id FROM decisions WHERE id=?", (value["id"],)).fetchone())

    def test_model_without_reasoning_is_usable_without_unsupported_think_option(self):
        import io
        metadata = io.BytesIO(json.dumps({"capabilities": ["completion"]}).encode())
        output = json.dumps({"message": {"content": json.dumps(decision())}}).encode()
        with patch("urllib.request.urlopen", side_effect=[metadata, io.BytesIO(output), io.BytesIO(output)]) as send:
            self.brain.decide({})
            self.brain.decide({})
        self.assertEqual(send.call_count, 3, "Model capabilities should be queried only once per brain")
        payload = json.loads(send.call_args.args[0].data)
        self.assertNotIn("think", payload)
        self.assertEqual(payload["options"]["num_ctx"], 8192, "Unknown metadata uses a safe trained context")
        self.assertEqual(payload["options"]["num_predict"], 1024, "Non-reasoning actor output is tightly capped")

    def test_lesson_retention(self):
        for index in range(2050):
            self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                                  (str(index), index, "mine", str(index), "", ""))
            self.brain.record_result({"last_result": {"id": str(index), "status": "failed", "detail": "No tool"}})
        self.assertEqual(self.brain.db.execute("SELECT count(*) FROM lessons").fetchone()[0], 2000)

    def test_observation_disables_impossible_actions_without_mutating_schema(self):
        original = copy.deepcopy(SCHEMA)
        allowed = action_schema({"inventory": [], "craftable": [], "entities": [],
                                 "nearby_blocks": []})["properties"]["action"]["enum"]
        self.assertEqual(set(allowed), {"explore", "goto", "wait", "escape", "jump", "sneak", "run", "sprint"})
        self.assertEqual(SCHEMA, original)
        allowed = allowed_actions({"entities": [{"type": "minecraft:item", "id": 1}]})
        self.assertIn("pickup", allowed)
        self.assertNotIn("attack", allowed)

    def test_stuck_detection_forces_a_different_experiment_and_retains_diagnosis(self):
        diagnosis = {"stuck": True, "reason": "Circling on a tree", "recovery_attempts": 0}
        observation = {"diagnosis": diagnosis, "inventory": [], "nearby_blocks": []}
        self.assertEqual(action_schema(observation)["properties"]["action"]["enum"], ["escape"])
        context = json.loads(self.brain.messages(observation)[-1]["content"])
        self.assertEqual(context["diagnosis"], diagnosis)

    def test_recovery_damage_is_persisted_as_an_observed_lesson(self):
        self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                              ("drop", 1, "escape", "", "Get off tree", "This will be safe"))
        self.brain.db.commit()
        self.brain.record_result({"last_result": {"id": "drop", "action": "escape",
                                 "status": "success", "detail": "Landed", "damage": 4,
                                 "health_after": 16}})
        lesson = self.brain.db.execute("SELECT text FROM lessons").fetchone()[0]
        self.assertIn("damage=4", lesson)
        self.assertIn("health_after=16", lesson)
        self.assertNotIn("This will be safe", lesson)

    def test_craft_schema_constrains_targets_without_restricting_other_actions(self):
        observation = {"craftable": ["minecraft:spruce_planks"], "inventory": ["minecraft:spruce_log=11"]}
        crafting, other = action_schema(observation)["oneOf"]
        self.assertEqual(crafting["properties"]["action"]["enum"], ["craft"])
        self.assertEqual(crafting["properties"]["target"]["enum"], ["minecraft:spruce_planks"])
        self.assertNotIn("craft", other["properties"]["action"]["enum"])
        self.assertEqual(other["properties"]["target"]["enum"], [""])

    def test_recipe_schema_is_bounded_and_prioritizes_requested_and_next_recipes(self):
        observation = {"craftable": [f"minecraft:recipe_{i}" for i in range(60)],
                       "crafting": {"next_step": {"target": "minecraft:recipe_58"}},
                       "chat": ["Craft minecraft:recipe_59 please."]}
        targets = action_schema(observation)["oneOf"][0]["properties"]["target"]["enum"]
        self.assertEqual(len(targets), 32)
        self.assertEqual(targets[0], "minecraft:recipe_58")
        self.assertIn("minecraft:recipe_59", targets)
        self.assertEqual(len(observation["craftable"]), 60)

    def test_unavailable_craft_is_corrected_once_and_not_persisted_as_executed(self):
        observation = {"craftable": ["minecraft:spruce_planks"], "inventory": ["minecraft:spruce_log=11"]}
        invalid = decision(action="craft", target="minecraft:wooden_pickaxe")
        valid = decision(action="craft", target="minecraft:spruce_planks")
        with self.assertLogs("localai", level="WARNING"), patch.object(
                self.brain, "inference", side_effect=[invalid, valid]) as model:
            response = self.brain.decide(observation)
        self.assertEqual(response["target"], "minecraft:spruce_planks")
        self.assertEqual(model.call_count, 2)
        retry = model.call_args.args
        context = json.loads(retry[0][-1]["content"])
        self.assertIn("not an available recipe", context["correction"])
        self.assertLessEqual(len(compact(retry[0]).encode()) + len(compact(retry[1]).encode()), PROMPT_BYTES)
        self.assertEqual(self.brain.db.execute("SELECT action,target FROM decisions").fetchall(),
                         [("craft", "minecraft:spruce_planks")])

    def test_low_context_budget_adapts_without_rejection(self):
        for i in range(10):
            self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                                  (str(i), i, "mine", "minecraft:spruce_log", "Get wood", "",
                                   compact({"status": "success", "detail": f"Mined block {i} at standard site"})))
            self.brain.db.execute("INSERT INTO lessons VALUES (?,?,?)",
                                  (f"lesson_{i}", f"mine minecraft:spruce_log success detail {i}", i))
        self.brain.db.commit()
        obs = {
            "position": [-624, 103, -48], "dimension": "minecraft:overworld",
            "health": 20, "food": 20, "time": 1000,
            "inventory": ["minecraft:spruce_planks=2", "minecraft:stick=4"],
            "nearby_blocks": [{"id": "minecraft:spruce_log", "at": [-625, 105, -49]}] * 20,
            "craftable": ["minecraft:wooden_pickaxe", "minecraft:stick", "minecraft:spruce_planks"],
            "crafting": {"grid": "3x3", "table_reachable": True,
                         "placement_sites": [[-624, 103, -47], [-623, 103, -48]]},
            "last_result": {"id": "9", "status": "success", "detail": "Mined block"},
        }
        for budget in (20608, 16000, 12000, 10000, 8000):
            self.brain.profile = {"prompt_bytes": budget, "thinking": True}
            messages = self.brain.messages(obs)
            total = len(compact(messages).encode()) + len(compact(action_schema(obs)).encode())
            self.assertLessEqual(total, budget)

    def test_oversized_unicode_strategist_reason_is_safely_bounded(self):
        long_unicode_reason = "Finding wood — spruce trees 🌲 are visible nearby. " * 10
        with patch.object(self.brain, "inference", return_value={
            "goal": "Collect wood", "reason": long_unicode_reason, "code": ""
        }):
            plan = self.brain.think_plan([{"role": "user", "content": "{}"}])
        self.assertLessEqual(len(plan["reason"].encode("utf-8")), 240)

    def test_tool_deduplication_excludes_owned_tools_from_craft_targets(self):
        obs = {
            "inventory": ["minecraft:stone_pickaxe=1", "minecraft:stone_sword=1", "minecraft:cobblestone=20", "minecraft:stick=10"],
            "craftable": ["minecraft:stone_pickaxe", "minecraft:wooden_pickaxe", "minecraft:stone_sword",
                          "minecraft:wooden_sword", "minecraft:stone_axe", "minecraft:lever"],
            "crafting": {"table_reachable": True, "placement_sites": []},
            "chat": [],
        }
        targets = craft_targets(obs)
        self.assertNotIn("minecraft:stone_pickaxe", targets)
        self.assertNotIn("minecraft:wooden_pickaxe", targets)
        self.assertNotIn("minecraft:stone_sword", targets)
        self.assertNotIn("minecraft:wooden_sword", targets)
        self.assertNotIn("minecraft:lever", targets)
        self.assertIn("minecraft:stone_axe", targets)

    def test_tool_deduplication_allows_upgrades(self):
        obs = {
            "inventory": ["minecraft:wooden_pickaxe=1", "minecraft:cobblestone=20", "minecraft:stick=4"],
            "craftable": ["minecraft:stone_pickaxe", "minecraft:wooden_pickaxe"],
            "crafting": {"table_reachable": True, "placement_sites": []},
            "chat": [],
        }
        targets = craft_targets(obs)
        self.assertNotIn("minecraft:wooden_pickaxe", targets)
        self.assertIn("minecraft:stone_pickaxe", targets)

    def test_chat_requested_tools_override_deduplication(self):
        obs = {
            "inventory": ["minecraft:stone_pickaxe=2"],
            "craftable": ["minecraft:stone_pickaxe", "minecraft:wooden_pickaxe"],
            "crafting": {"table_reachable": True, "placement_sites": []},
            "chat": ["Willow craft a stone pickaxe for me"],
        }
        targets = craft_targets(obs)
        self.assertIn("minecraft:stone_pickaxe", targets)
        self.assertEqual(targets[0], "minecraft:stone_pickaxe")

    def test_duplicate_craft_rejected_by_validate_action(self):
        obs = {
            "inventory": ["minecraft:stone_pickaxe=1"],
            "craftable": ["minecraft:stone_pickaxe", "minecraft:stone_axe"],
            "crafting": {"table_reachable": True, "placement_sites": []},
            "chat": [],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_action(decision(action="craft", target="minecraft:stone_pickaxe"), obs)


    def test_movement_and_control_actions_are_validated(self):
        obs = {
            "position": [10, 64, 20],
            "inventory": ["minecraft:dirt=10", "minecraft:cobblestone=20"],
            "placeable": ["minecraft:dirt", "minecraft:cobblestone"],
            "crafting": {"placement_sites": [[10, 64, 20], [11, 64, 20]]},
        }
        # Jump
        val = validate_action(decision(action="jump"), obs)
        self.assertEqual(val["action"], "jump")
        # Run / Sprint
        val = validate_action(decision(action="run", x=15, y=64, z=20), obs)
        self.assertEqual(val["action"], "run")
        val = validate_action(decision(action="sprint", x=15, y=64, z=20), obs)
        self.assertEqual(val["action"], "sprint")
        # Sneak
        val = validate_action(decision(action="sneak"), obs)
        self.assertEqual(val["action"], "sneak")
        # Pillar with auto-select or target
        val = validate_action(decision(action="pillar"), obs)
        self.assertEqual(val["action"], "pillar")
        val = validate_action(decision(action="pillar", target="minecraft:cobblestone"), obs)
        self.assertEqual(val["target"], "minecraft:cobblestone")
        with self.assertRaisesRegex(ValueError, "owned placeable"):
            validate_action(decision(action="pillar", target="minecraft:diamond"), obs)

    def test_chat_requested_mine_targets_are_prioritized(self):
        obs = {
            "nearby_blocks": [
                {"id": "minecraft:stone", "at": [1, 64, 1]},
                {"id": "minecraft:oak_log", "at": [2, 64, 2]},
                {"id": "minecraft:dirt", "at": [3, 64, 3]},
            ],
            "chat": ["Willow please gather some oak logs for me"],
        }
        targets = action_targets(obs, "mine")
        self.assertIn("minecraft:oak_log", targets)
        self.assertEqual(targets[0], "minecraft:oak_log")

    def test_chat_ensures_speech_even_if_model_say_is_empty(self):
        with patch.object(self.brain, "inference", return_value=decision(action="mine", target="minecraft:oak_log", goal="Gather wood", say="")):
            dec = self.brain.decide({
                "position": [0, 64, 0],
                "nearby_blocks": [{"id": "minecraft:oak_log", "at": [1, 64, 0]}],
                "chat": ["Hello Willow!"],
            })
            self.assertTrue(dec["say"])
            self.assertIn("Gather wood", dec["say"])

    def test_active_tools_awareness_in_messages(self):
        obs = {
            "position": [10, 64, 20],
            "sight": {
                "facing": "North",
                "looking_at": {"id": "minecraft:oak_log", "at": [10, 65, 21], "distance": 1.2},
                "visible_entities": [{"type": "minecraft:pig", "distance": 3.0, "bearing": "ahead"}],
                "visual_hazards": [],
            },
            "hearing": {
                "active_alerts": [],
                "recent_sounds": [{"sound": "minecraft:entity.pig.ambient", "distance": 3.0}],
            },
            "inventory": ["minecraft:wooden_pickaxe=1"],
            "nearby_blocks": [{"id": "minecraft:oak_log", "at": [10, 65, 21]}],
            "craftable": [],
        }
        messages = self.brain.messages(obs)
        user_content = json.loads(messages[-1]["content"])
        self.assertIn("active_tools", user_content)
        tools = user_content["active_tools"]
        self.assertIn("vision", tools)
        self.assertIn("hearing", tools)
        self.assertIn("actions", tools)
        self.assertEqual(tools["vision"]["looking_at"]["id"], "minecraft:oak_log")
        self.assertEqual(tools["vision"]["visible_entities"][0]["type"], "minecraft:pig")
        self.assertIn("mine", tools["actions"])

    def test_schema_and_validator_limits_are_unified(self):
        from server import DECISION_LIMITS, PLAN_SCHEMA
        for key in ("target", "goal", "say", "memory", "waypoint"):
            self.assertEqual(SCHEMA["properties"][key]["maxLength"], DECISION_LIMITS[key])
        self.assertEqual(SCHEMA["properties"]["plan"]["maxItems"], DECISION_LIMITS["plan"])
        self.assertEqual(PLAN_SCHEMA["properties"]["goal"]["maxLength"], DECISION_LIMITS["goal"])
        self.assertEqual(PLAN_SCHEMA["properties"]["reason"]["maxLength"], DECISION_LIMITS["memory"])
        self.assertEqual(PLAN_SCHEMA["properties"]["code"]["maxLength"], DECISION_LIMITS["code"])

    def test_simulated_observations_across_trimming_stages_and_longest_prompt(self):
        from server import SYSTEM, SYSTEM_COMPACT
        for i in range(25):
            self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                                  (f"d_{i}", i, "mine", "minecraft:oak_log", "Gather wood", "",
                                   compact({"status": "success", "detail": f"Mined log {i} at forest clearing"})))
            self.brain.db.execute("INSERT INTO lessons VALUES (?,?,?)",
                                  (f"l_{i}", f"mine minecraft:oak_log: success Mined log {i}", i))
        self.brain.db.commit()
        longest_obs = {
            "position": [-1200, 68, 450],
            "dimension": "minecraft:overworld",
            "health": 19.5,
            "food": 18,
            "time": 6000,
            "inventory": [f"minecraft:item_{i}=16" for i in range(36)],
            "nearby_blocks": [{"id": "minecraft:oak_log", "at": [-1200 + (i % 10), 68 + (i // 10), 450]} for i in range(64)],
            "craftable": [f"minecraft:recipe_{i}" for i in range(40)],
            "placeable": ["minecraft:dirt", "minecraft:cobblestone", "minecraft:crafting_table"],
            "crafting": {
                "grid": "3x3",
                "table_reachable": True,
                "placement_sites": [[-1199, 68, 450], [-1200, 68, 451], [-1201, 68, 450], [-1200, 68, 449]],
                "knowledge": ["Planks craft table; sticks + planks craft pickaxe." * 4],
            },
            "entities": [{"id": i, "type": "minecraft:zombie", "hostile": True, "distance": 12.0} for i in range(1, 20)],
            "terrain": {"exits": [{"at": [-1198, 68, 450], "standable": True, "feet": "minecraft:air", "head": "minecraft:air"} for _ in range(12)]},
            "diagnosis": {"stuck": False, "risk_policy": "Avoid deep shafts without blocks" * 3},
            "sight": {"facing": "East", "looking_at": {"id": "minecraft:oak_log", "at": [-1199, 68, 450]}},
            "hearing": {"active_alerts": [{"sound": "minecraft:entity.zombie.ambient", "distance": 12.0, "bearing": "ahead"}]},
            "chat": ["Keep gathering logs for our shelter"],
        }
        schema_len = len(compact(action_schema(longest_obs)).encode())
        # Stage 0: large 16k-token budget (~60 KB) retains scaled lists (>4 nearby_blocks, >6 recent_actions)
        self.brain.profile = {"prompt_bytes": 60000, "prompt_tokens": 14800, "trim_scale": 2.6, "compact_prompt_threshold_tokens": 4096}
        msgs_large = self.brain.messages(longest_obs)
        self.assertEqual(msgs_large[0]["content"], SYSTEM)
        ctx_large = json.loads(msgs_large[-1]["content"])
        self.assertGreaterEqual(len(ctx_large["nearby_blocks"]), 10)
        self.assertGreaterEqual(len(ctx_large["recent_actions"]), 12)
        self.assertLessEqual(len(compact(msgs_large).encode()) + schema_len, 60000)

        # Stages 1-4: progressively smaller budgets down to tight compact emergency budget
        for budget, tok in ((18000, 5100), (14000, 4000), (11000, 3100), (9200, 2600)):
            self.brain.profile = {"prompt_bytes": budget, "prompt_tokens": tok, "compact_prompt_threshold_tokens": 3000}
            msgs = self.brain.messages(longest_obs)
            self.assertLessEqual(len(compact(msgs).encode()) + schema_len, budget)
            if tok < 3000:
                self.assertEqual(msgs[0]["content"], SYSTEM_COMPACT)

    def test_live_metrics_and_done_reason_length_recovery(self):
        import io
        show_resp = io.BytesIO(json.dumps({
            "capabilities": ["completion", "thinking"],
            "model_info": {"general.architecture": "gemma4", "gemma4.context_length": 32768},
        }).encode())
        len_resp = io.BytesIO(json.dumps({
            "message": {"content": ""},
            "done_reason": "length",
        }).encode())
        ok_resp = io.BytesIO(json.dumps({
            "message": {"content": json.dumps(decision(action="explore"))},
            "done_reason": "stop",
            "prompt_eval_count": 800,
            "prompt_eval_duration": 800_000_000,  # 1000 tok/s
            "eval_count": 100,
            "eval_duration": 2_000_000_000,        # 50 tok/s
        }).encode())
        with patch("urllib.request.urlopen", side_effect=[show_resp, len_resp, ok_resp]) as send:
            res = self.brain.decide({"health": 20, "food": 20})
        self.assertEqual(res["action"], "explore")
        self.assertEqual(send.call_count, 3, "Should retry once with think=False after done_reason=='length'")
        retry_payload = json.loads(send.call_args.args[0].data)
        self.assertIs(retry_payload["think"], False)
        self.assertGreater(self.brain.profile["prompt_tps"], 100)
        self.assertGreater(self.brain.profile["gen_tps"], 20)
        self.assertLess(self.brain.actor_timeout, 240)

    def test_tree_in_sight_prevents_circling_and_chains_plan(self):
        obs = {
            "position": [10, 64, 10],
            "inventory": ["minecraft:oak_planks=4"],  # 1 log worth (< 12 plank-equiv needed for table+pickaxe)
            "nearby_blocks": [
                {"id": "minecraft:oak_log", "at": [12, 64, 10]},
                {"id": "minecraft:oak_log", "at": [12, 65, 10]},
                {"id": "minecraft:oak_log", "at": [12, 66, 10]},
                {"id": "minecraft:oak_log", "at": [12, 67, 10]},
                {"id": "minecraft:oak_log", "at": [12, 68, 10]},
            ],
            "entities": [],
            "craftable": [],
            "health": 20,
            "food": 20,
        }
        msgs = self.brain.messages(obs)
        user_ctx = json.loads(msgs[-1]["content"])
        self.assertIn("visible_tree", user_ctx)
        self.assertEqual(user_ctx["visible_tree"]["nearest_at"], [12, 64, 10])
        with patch.object(self.brain, "inference", return_value=decision(action="jump", goal="Find wood")):
            dec = self.brain.decide(obs)
        self.assertEqual(dec["action"], "mine")
        self.assertEqual(dec["target"], "minecraft:oak_log")
        self.assertEqual([dec["x"], dec["y"], dec["z"]], [12, 64, 10])
        self.assertEqual(len(dec["plan"]), 4, "Remaining 4 tree trunk logs should be chained into follow-up plan")
        self.assertEqual(dec["plan"][0]["action"], "mine")
        self.assertEqual([dec["plan"][0]["x"], dec["plan"][0]["y"], dec["plan"][0]["z"]], [12, 65, 10])

    def test_goto_loop_guards_and_phantom_resource_pruning(self):
        obs = {"position": [-625, 78, -1014], "nearby_blocks": [], "inventory": []}
        with self.assertRaisesRegex(ValueError, "Already at or adjacent"):
            validate_action(decision(action="goto", x=-625, y=78, z=-1013), obs)

        # Partial zero-axis hallucination (x=0 while z is local)
        partial_zero_obs = {"position": [-1732, 63, -672], "nearby_blocks": [], "inventory": []}
        fixed = validate_action(decision(action="goto", x=0, y=63, z=-676), partial_zero_obs)
        self.assertEqual((fixed["action"], fixed["x"], fixed["y"], fixed["z"]), ("explore", 0, 0, 0))

        # Full or near-zero coordinate hallucinations (e.g. [0, 99, 0] or [0, 102, -8] when far from origin)
        for bad_goto in (decision(action="goto", x=0, y=99, z=0), decision(action="goto", x=0, y=102, z=-8)):
            cleared = validate_action(bad_goto, obs)
            self.assertEqual((cleared["action"], cleared["x"], cleared["y"], cleared["z"]), ("explore", 0, 0, 0))

        # Repeated goto to identical coordinates in recent_actions is rejected
        repeat_obs = {
            "position": [-625, 78, -1014],
            "nearby_blocks": [],
            "inventory": [],
            "recent_actions": [{"action": "goto", "target": "", "result": {"status": "success", "detail": "Arrived at -615 78 -1010"}}],
        }
        with self.assertRaisesRegex(ValueError, "Repeated goto"):
            validate_action(decision(action="goto", x=-615, y=78, z=-1010), repeat_obs)

        # Verify memory prunes adjacent absent seen_blocks and never reports distance<=3 in remembered_resources
        self.brain.perceive({
            "position": [-625, 78, -1014],
            "dimension": "minecraft:overworld",
            "nearby_blocks": [{"id": "minecraft:oak_log", "at": [-625, 78, -1013]}],
        })
        after_mined = self.brain.perceive({
            "position": [-625, 78, -1014],
            "dimension": "minecraft:overworld",
            "nearby_blocks": [{"id": "minecraft:dirt", "at": [-625, 77, -1014]}],
        })
        mem_ctx = self.brain.memory.context(after_mined)
        self.assertEqual(mem_ctx["remembered_resources"], [])

    def test_planner_mode_idle_only_defers_under_contention(self):
        planned = Brain(Path(self.tmp.name) / "planned.sqlite3", planning=True)
        try:
            planned.profile = {
                "planner_mode": "idle_only",
                "prompt_bytes": 24000,
                "prompt_tokens": 6000,
                "timeouts": {"next_plan_interval": 30},
            }
            obs = {"position": [0, 64, 0], "health": 20, "food": 20, "inventory": [], "nearby_blocks": []}
            msgs = planned.messages(obs)
            planned._active_actor_inferences = 1
            planned.schedule_planner(obs, msgs)
            self.assertIsNone(planned.pending_plan)
            self.assertIn("deferred", planned.planner_status)

            # Danger also defers idle_only planner
            planned._active_actor_inferences = 0
            danger_obs = {"position": [0, 64, 0], "health": 6, "food": 20, "inventory": [], "nearby_blocks": []}
            planned.schedule_planner(danger_obs, msgs)
            self.assertIsNone(planned.pending_plan)
            self.assertIn("deferred", planned.planner_status)
        finally:
            planned.close()

    def test_long_unattended_session_memory_bounds(self):
        self.brain.profile = {"gameplay_limits": {"lessons_cap": 50, "decisions_cap": 80}}
        for i in range(120):
            did = f"dec_{i}"
            self.brain.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                                  (did, float(i), "mine", f"minecraft:block_{i}", "Survive", ""))
            self.brain.record_result({"last_result": {"id": did, "status": "success", "detail": f"Step {i}"}})
        self.brain.db.execute(
            "DELETE FROM decisions WHERE id NOT IN (SELECT id FROM decisions ORDER BY created DESC LIMIT ?)",
            (self.brain.decisions_cap,),
        )
        self.brain.db.commit()
        self.assertLessEqual(self.brain.db.execute("SELECT count(*) FROM lessons").fetchone()[0], 50)
        self.assertLessEqual(self.brain.db.execute("SELECT count(*) FROM decisions").fetchone()[0], 80)

    def test_new_survival_actions_and_anti_circling_optimizations(self):
        # 1. Consecutive goto loop guard rejects a 3rd consecutive goto
        loop_obs = {
            "position": [180, 64, 53],
            "nearby_blocks": [{"id": "minecraft:birch_log", "at": [180, 65, 53]}],
            "inventory": [],
            "recent_actions": [
                {"action": "goto", "target": "", "result": {"status": "success", "detail": "Reached 179, 64, 53"}},
                {"action": "goto", "target": "", "result": {"status": "success", "detail": "Reached 180, 64, 53"}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "2 consecutive goto"):
            validate_action(decision(action="goto", x=185, y=64, z=55), loop_obs)

        # 2. Leaf litter and decorative ground cover are filtered out of mine targets when real resources exist
        foliage_obs = {
            "position": [180, 64, 53],
            "nearby_blocks": [
                {"id": "minecraft:leaf_litter", "at": [180, 64, 54]},
                {"id": "minecraft:short_grass", "at": [181, 64, 53]},
                {"id": "minecraft:birch_log", "at": [180, 65, 53]},
            ],
            "inventory": [],
        }
        mine_targets = action_targets(foliage_obs, "mine")
        self.assertEqual(mine_targets, ["minecraft:birch_log"])

        # 3. Smelt, chest, sleep, trade, boat, tunnel, bridge, shelter, portal are enabled when conditions hold
        rich_obs = {
            "position": [10, 50, 10],
            "time": 14000,
            "in_water": True,
            "inventory": ["minecraft:raw_iron=2", "minecraft:coal=2", "minecraft:cobblestone=32",
                          "minecraft:oak_boat=1", "minecraft:red_bed=1", "minecraft:chest=1"],
            "placeable": ["minecraft:cobblestone", "minecraft:chest", "minecraft:red_bed"],
            "smeltable": ["minecraft:raw_iron"],
            "trades": ["minecraft:emerald"],
            "nearby_blocks": [{"id": "minecraft:obsidian", "at": [11, 50, 10]}, {"id": "minecraft:water", "at": [10, 50, 11]}],
            "crafting": {"furnace_reachable": True, "chest_reachable": True, "bed_reachable": True,
                         "placement_sites": [[11, 50, 11]]},
            "entities": [{"id": 5, "type": "minecraft:villager", "distance": 2.0, "hostile": False}],
        }
        acts = allowed_actions(rich_obs)
        for expected in ("smelt", "chest", "sleep", "trade", "boat", "tunnel", "bridge", "shelter", "portal"):
            self.assertIn(expected, acts)

    def test_oak_leaves_bark_wood_and_log_hoarding_are_blocked(self):
        obs = {
            "position": [-143, 64, -1105],
            "dimension": "minecraft:overworld",
            "inventory": ["minecraft:oak_log=12", "minecraft:oak_planks=16", "minecraft:wooden_pickaxe=1"],
            "craftable": ["minecraft:oak_planks", "minecraft:oak_wood", "minecraft:birch_wood", "minecraft:stick"],
            "nearby_blocks": [
                {"id": "minecraft:oak_leaves", "at": [-143, 65, -1103]},
                {"id": "minecraft:oak_log", "at": [-143, 65, -1104]},
                {"id": "minecraft:stone", "at": [-140, 63, -1105]},
            ],
        }
        # 1. Decorative 4-sided bark blocks (*_wood) and excess planks (when pickaxe owned) are excluded
        self.assertNotIn("minecraft:oak_wood", craft_targets(obs))
        self.assertNotIn("minecraft:birch_wood", craft_targets(obs))
        self.assertNotIn("minecraft:oak_planks", craft_targets(obs))
        # 2. Both oak_leaves and excess oak_log are excluded from mine targets; stone remains available
        self.assertEqual(action_targets(obs, "mine"), ["minecraft:stone"])
        with self.assertRaisesRegex(ValueError, "enough wood"):
            validate_action(decision(action="mine", target="minecraft:oak_log", x=-143, y=65, z=-1104), obs)
        with self.assertRaisesRegex(ValueError, "Do not mine tree leaves"):
            validate_action(decision(action="mine", target="minecraft:oak_leaves", x=-143, y=65, z=-1103),
                            dict(obs, inventory=[]))


if __name__ == "__main__":
    unittest.main()

