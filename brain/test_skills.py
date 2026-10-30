from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from server import Brain, PLAN_MAX_AGE, PROMPT_BYTES, allowed_actions, compact, validate_action
from skills import interpret, normalize, parse
from test_brain import decision


CODE = '''def skill():
    if can_craft("minecraft:spruce_planks") and count("#planks") < 4:
        return craft("minecraft:spruce_planks")
    return None
'''


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "brain.sqlite3"
        self.brain = Brain(self.path, planning=False)

    def tearDown(self):
        self.brain.close()
        self.tmp.cleanup()

    def test_restricted_language_and_current_observation(self):
        body = parse(CODE)
        observation = {"craftable": ["minecraft:spruce_planks"], "inventory": [],
                       "crafting": {"planks": 0}}
        self.assertEqual(interpret(body, observation)["target"], "minecraft:spruce_planks")
        observation["crafting"]["planks"] = 4
        self.assertIsNone(interpret(body, observation))

    def test_host_python_and_unbounded_programs_are_rejected(self):
        bad = [
            "import os\n",
            "def skill():\n    return __import__('os')\n",
            "def skill():\n    return open('/tmp/x')\n",
            "def skill():\n    while True:\n        return None\n",
            "def skill():\n    return (1).__class__\n",
            "def skill():\n    return [x for x in []]\n",
            "def skill():\n    return 'x' * 10000000\n",
            "@explore()\ndef skill():\n    return None\n",
            "def skill(x=explore()):\n    return None\n",
            "def skill():\n    return foundation()\n" + "# large\n" * 600,
        ]
        for code in bad:
            with self.subTest(code=code[:60]), self.assertRaises(ValueError):
                parse(code)

    def test_inventory_table_fast_path_uses_live_site(self):
        observation = {"inventory": ["minecraft:crafting_table=1"], "craftable": [],
                       "placeable": ["minecraft:crafting_table"],
                       "crafting": {"placement_sites": [[2, 64, 3]], "next_step": {
                           "action": "place", "target": "minecraft:crafting_table", "x": 2, "y": 64, "z": 3}}}
        with patch.object(self.brain, "inference", side_effect=AssertionError("No model needed")):
            value = self.brain.decide(observation)
        self.assertEqual((value["action"], value["x"]), ("place", 2))
        with self.assertRaisesRegex(ValueError, "usable site"):
            validate_action(decision(action="place", target="minecraft:crafting_table"), observation)

    def test_unavailable_recipe_is_not_accepted_when_other_recipes_exist(self):
        with self.assertRaisesRegex(ValueError, "available recipe"):
            validate_action(decision(action="craft", target="minecraft:wooden_pickaxe"),
                            {"craftable": ["minecraft:spruce_planks"]})

    def test_failed_escape_does_not_force_an_endless_escape_loop(self):
        observation = {"diagnosis": {"stuck": True},
                       "last_result": {"action": "escape", "status": "failed"},
                       "craftable": ["minecraft:spruce_planks"]}
        self.assertIn("craft", allowed_actions(observation))

    def test_foundation_collects_materials_before_generated_mining(self):
        self.brain.skills.install('def skill():\n    return mine("minecraft:oak_log")\n')
        with patch.object(self.brain, "inference", side_effect=AssertionError("Use current wood drops")):
            value = self.brain.decide({"crafting": {"pickaxes": 0}, "entities": [
                {"type": "minecraft:item", "item": "minecraft:oak_log"}],
                "nearby_blocks": [{"id": "minecraft:oak_log"}]})
        self.assertEqual(value["action"], "pickup")

    def test_saved_generated_skill_executes_and_failures_disable_without_reset(self):
        key = self.brain.skills.install(CODE)
        self.assertEqual((Path(self.tmp.name) / "skills" / (key + ".py")).read_text(), CODE)
        observation = {"craftable": ["minecraft:spruce_planks"], "inventory": ["minecraft:spruce_log=12"]}
        for attempt in range(3):
            observation["position"] = [attempt * 16, 64, 0]
            with patch.object(self.brain, "inference", side_effect=AssertionError("Use generated code")):
                value = self.brain.decide(observation)
            result = {"last_result": {"id": value["id"], "status": "failed", "detail": "Resource changed"}}
            self.brain.record_result(result)
            self.brain.record_result(result)
        self.brain.skills.install(CODE)
        self.assertEqual(self.brain.db.execute(
            "SELECT failures,disabled FROM skills WHERE id=?", (key,)).fetchone(), (3, 1))
        other = Brain(self.path, planning=False)
        self.assertEqual(other.db.execute("SELECT disabled FROM skills WHERE id=?", (key,)).fetchone(), (1,))
        other.close()

    def test_invalid_generated_action_disables_version_and_uses_actor(self):
        key = self.brain.skills.install('def skill():\n    return craft("minecraft:diamond_pickaxe")\n')
        with self.assertLogs("localai", level="ERROR"), patch.object(self.brain, "inference", return_value=decision()):
            value = self.brain.decide({"craftable": ["minecraft:spruce_planks"]})
        self.assertEqual(value["action"], "explore")
        self.assertEqual(self.brain.db.execute("SELECT disabled FROM skills WHERE id=?", (key,)).fetchone(), (1,))

    def test_chat_and_emergency_preempt_skills(self):
        self.brain.skills.install(CODE)
        for state in ({"chat": ["Hello"]}, {"food": 10}, {"diagnosis": {"stuck": True}}):
            expected = decision(action="escape" if "diagnosis" in state else "explore")
            with patch.object(self.brain, "inference", return_value=expected) as actor:
                self.brain.decide(dict(craftable=["minecraft:spruce_planks"], **state))
            actor.assert_called_once()

    def test_two_roles_overlap_without_sharing_sqlite_with_worker(self):
        brain = Brain(Path(self.tmp.name) / "parallel.sqlite3")
        entered = threading.Event()
        release = threading.Event()
        actor_called = threading.Event()

        def inference(messages, schema, role, **kwargs):
            self.assertLessEqual(len(compact(messages).encode()) + len(compact(schema).encode()), PROMPT_BYTES)
            if role == "strategist":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("Actor did not run while strategist was thinking")
                return {"goal": "Make planks", "reason": "Logs are available", "code": CODE}
            actor_called.set()
            return decision()

        try:
            with patch.object(brain, "inference", side_effect=inference):
                brain.decide({"inventory": []})
                self.assertTrue(actor_called.is_set())
                self.assertTrue(entered.wait(3))
                self.assertFalse(brain.pending_plan.done())
                brain.decide({"inventory": []})
                self.assertFalse(brain.pending_plan.done())
                release.set()
                brain.pending_plan.result(timeout=3)
                brain.decide({"inventory": [], "craftable": ["minecraft:spruce_planks"]})
            self.assertIsNotNone(brain.db.execute("SELECT value FROM state WHERE key='strategy'").fetchone())
            self.assertEqual(brain.db.execute("SELECT count(*) FROM skills").fetchone()[0], 2)
        finally:
            release.set()
            brain.close()

    def test_stale_strategy_is_discarded_and_background_errors_reported(self):
        brain = Brain(Path(self.tmp.name) / "stale.sqlite3")
        try:
            brain.next_plan = time.monotonic() + 100
            for stale in ("age", "dimension", "chat"):
                future = Future()
                future.set_result({"goal": "Old plan", "reason": "", "code": CODE})
                brain.pending_plan = future
                brain.plan_started = time.monotonic() - (PLAN_MAX_AGE + 1 if stale == "age" else 0)
                brain.plan_dimension = "overworld" if stale == "dimension" else None
                brain.plan_revision = brain.chat_revision
                observation = {"chat": ["New task"]} if stale == "chat" else {}
                with self.assertLogs("localai", level="WARNING"), patch.object(brain, "inference", return_value=decision()):
                    brain.decide(observation)
                self.assertIsNone(brain.db.execute("SELECT value FROM state WHERE key='strategy'").fetchone())
            future = Future()
            future.set_exception(ValueError("Bad generated program"))
            brain.pending_plan = future
            with self.assertLogs("localai", level="ERROR"), patch.object(brain, "inference", return_value=decision()):
                brain.decide({})
            self.assertIn("failed", brain.planner_status)
        finally:
            brain.close()

    def test_skill_retention_bounds_database_and_files(self):
        for i in range(20):
            self.brain.skills.install(f'def skill():\n    if count("minecraft:stick") == {i}:\n        return None\n')
        self.assertEqual(self.brain.db.execute("SELECT count(*) FROM skills").fetchone()[0], 16)
        self.assertEqual(len(list((Path(self.tmp.name) / "skills").glob("*.py"))), 16)

    def test_repeated_explore_loop_without_progress_is_disabled_and_yields_to_actor(self):
        code = 'def skill():\n    if count("#planks") == 0:\n        return explore()\n    return None\n'
        key = self.brain.skills.install(code)
        observation = {"inventory": [], "craftable": []}

        # Turn 1: runs explore via skill
        with patch.object(self.brain, "inference", side_effect=AssertionError("Skill should run")):
            val1 = self.brain.decide(observation)
        self.assertEqual(val1["action"], "explore")
        self.brain.record_result({"last_result": {"id": val1["id"], "action": "explore", "status": "success"}})

        # Turn 2: runs explore via skill
        with patch.object(self.brain, "inference", side_effect=AssertionError("Skill should run")):
            val2 = self.brain.decide(observation)
        self.assertEqual(val2["action"], "explore")
        self.brain.record_result({"last_result": {"id": val2["id"], "action": "explore", "status": "success"}})

        # Turn 3: 3rd attempt without progress -> rejected and disabled, actor called
        actor_decision = decision(action="explore", goal="Actor taking over")
        with patch.object(self.brain, "inference", return_value=actor_decision) as actor_mock:
            val3 = self.brain.decide(observation)
        actor_mock.assert_called_once()
        self.assertEqual(val3["goal"], "Actor taking over")
        self.assertEqual(self.brain.db.execute("SELECT disabled FROM skills WHERE id=?", (key,)).fetchone(), (1,))

    def test_continuous_skill_burst_yields_to_actor_after_five_turns(self):
        code = 'def skill():\n    if can_craft("minecraft:spruce_planks"):\n        return craft("minecraft:spruce_planks")\n    return None\n'
        self.brain.skills.install(code)
        observation = {"inventory": ["minecraft:spruce_log=64"], "craftable": ["minecraft:spruce_planks"]}

        for i in range(5):
            with patch.object(self.brain, "inference", side_effect=AssertionError(f"Turn {i} should use skill")):
                val = self.brain.decide(observation)
            self.assertEqual(val["action"], "craft")
            self.brain.record_result({"last_result": {"id": val["id"], "action": "craft", "status": "success"},
                                      "inventory": [f"minecraft:spruce_planks={(i+1)*4}"]})
            observation["inventory"] = [f"minecraft:spruce_planks={(i+1)*4}"]

        actor_decision = decision(action="explore", goal="Situational check")
        with patch.object(self.brain, "inference", return_value=actor_decision) as actor_mock:
            val6 = self.brain.decide(observation)
        actor_mock.assert_called_once()

    def test_three_consecutive_no_progress_skill_actions_yield_to_actor(self):
        code = 'def skill():\n    return mine("minecraft:stone")\n'
        self.brain.skills.install(code)
        observation = {"inventory": [], "craftable": [], "nearby_blocks": [{"id": "minecraft:stone"}]}

        for i in range(3):
            with patch.object(self.brain, "inference", side_effect=AssertionError(f"Turn {i} should use skill")):
                val = self.brain.decide(observation)
            self.assertEqual(val["action"], "mine")
            self.brain.record_result({"last_result": {"id": val["id"], "action": "mine", "status": "success"}})

        actor_decision = decision(action="explore", goal="Break stall")
        with patch.object(self.brain, "inference", return_value=actor_decision) as actor_mock:
            val4 = self.brain.decide(observation)
        actor_mock.assert_called_once()

    def test_strategist_invalid_code_is_discarded_preserving_goal_and_reason(self):
        bad_response = {
            "goal": "Build a wooden shelter",
            "reason": "Need safe place for the night",
            "code": "def skill():\n    for x in range(10):\n        pass\n"
        }
        with patch.object(self.brain, "inference", return_value=bad_response):
            plan = self.brain.think_plan([{"role": "user", "content": "{}"}])
        self.assertEqual(plan["goal"], "Build a wooden shelter")
        self.assertEqual(plan["reason"], "Need safe place for the night")
        self.assertEqual(plan["code"], "")

    def test_pillar_places_under_feet_when_placement_site_available(self):
        code = 'def skill():\n    return pillar("minecraft:dirt")\n'
        body = parse(code)
        observation = {
            "position": [10, 64, 20],
            "inventory": ["minecraft:dirt=5"],
            "placeable": ["minecraft:dirt"],
            "crafting": {"placement_sites": [[11, 64, 20], [10, 64, 20]]},
        }
        res = interpret(body, observation)
        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "place")
        self.assertEqual(res["target"], "minecraft:dirt")
        self.assertEqual((res["x"], res["y"], res["z"]), (10, 64, 20))

        # Returns None if position is not in placement_sites or block not owned
        bad_obs = dict(observation, crafting={"placement_sites": [[11, 64, 20]]})
        self.assertIsNone(interpret(body, bad_obs))

    def test_normalize_unescapes_newlines_and_tabs(self):
        escaped = "def skill():\\n\\tif stuck():\\n\\t\\treturn explore()\\n"
        norm = normalize(escaped)
        self.assertIn("\n", norm)
        self.assertNotIn("\\n", norm)
        body = parse(norm)
        self.assertEqual(len(body), 1)

    def test_can_pillar_and_pillar_empty_target(self):
        code = 'def skill():\n    if can_pillar_up():\n        return pillar("")\n    return None\n'
        body = parse(code)
        observation = {
            "position": [10, 64, 20],
            "inventory": ["minecraft:cobblestone=10"],
            "placeable": ["minecraft:cobblestone"],
            "crafting": {"placement_sites": [[10, 64, 20]]},
        }
        res = interpret(body, observation)
        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "place")
        self.assertEqual(res["target"], "minecraft:cobblestone")

    def test_extended_survival_skills_smelt_chest_sleep_trade_boat_tunnel_bridge_shelter_portal(self):
        # 1. Smelt skill + furnace_reachable + can_smelt + place_furnace
        smelt_code = (
            'def skill():\n'
            '    if count("#raw_ores") > 0 and not furnace_reachable():\n'
            '        return place_furnace()\n'
            '    if can_smelt("minecraft:raw_iron"):\n'
            '        return smelt("minecraft:raw_iron")\n'
            '    return None\n'
        )
        smelt_body = parse(smelt_code)
        obs_place_furnace = {
            "inventory": ["minecraft:raw_iron=3", "minecraft:furnace=1", "minecraft:coal=2"],
            "crafting": {"furnace_reachable": False, "placement_sites": [[5, 64, 5]]},
        }
        res_place = interpret(smelt_body, obs_place_furnace)
        self.assertEqual((res_place["action"], res_place["target"]), ("place", "minecraft:furnace"))

        obs_smelt = {
            "inventory": ["minecraft:raw_iron=3", "minecraft:coal=2"],
            "smeltable": ["minecraft:raw_iron"],
            "crafting": {"furnace_reachable": True, "placement_sites": []},
        }
        res_smelt = interpret(smelt_body, obs_smelt)
        self.assertEqual((res_smelt["action"], res_smelt["target"]), ("smelt", "minecraft:raw_iron"))

        # 2. Sleep skill + is_night + can_sleep
        sleep_code = 'def skill():\n    if is_night() and can_sleep():\n        return sleep()\n    return None\n'
        sleep_body = parse(sleep_code)
        obs_night = {
            "time": 14000,
            "inventory": ["minecraft:red_bed=1"],
            "crafting": {"bed_reachable": True, "placement_sites": [[1, 64, 1]]},
        }
        res_sleep = interpret(sleep_body, obs_night)
        self.assertEqual(res_sleep["action"], "sleep")

        # 3. Chest, trade, boat, tunnel, bridge, shelter, portal
        for call_expr, expected_action, expected_target in (
            ('chest("minecraft:cobblestone")', "chest", "minecraft:cobblestone"),
            ('trade("minecraft:emerald")', "trade", "minecraft:emerald"),
            ('boat()', "boat", ""),
            ('tunnel()', "tunnel", ""),
            ('bridge()', "bridge", ""),
            ('shelter()', "shelter", ""),
            ('portal()', "portal", ""),
        ):
            b = parse(f"def skill():\n    return {call_expr}\n")
            out = interpret(b, {"inventory": ["minecraft:cobblestone=32"], "trades": ["minecraft:emerald"]})
            self.assertEqual((out["action"], out["target"]), (expected_action, expected_target))


if __name__ == "__main__":
    unittest.main()
