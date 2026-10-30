import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experience import situation
from server import Brain, PROMPT_BYTES, action_schema, allowed_actions, compact, validate_action, validate_decision
from skills import interpret, parse
from test_brain import decision


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "brain.sqlite3"
        self.brain = Brain(self.path, planning=False)
        self.observation = {"position": [10, 64, 10], "dimension": "minecraft:overworld",
                            "inventory": ["minecraft:wooden_pickaxe=1"],
                            "nearby_blocks": [{"id": "minecraft:stone", "at": [11, 64, 10]}],
                            "entities": [], "craftable": [], "health": 20, "food": 20}

    def tearDown(self):
        self.brain.close()
        self.tmp.cleanup()

    def outcome(self, observation, response, status="success"):
        return dict(observation, last_result={"id": response["id"], "action": response["action"],
                                             "status": status, "detail": "Measured result", "damage": 0})

    def test_two_failures_block_same_situation_but_not_new_positions_or_tools(self):
        for _ in range(2):
            with patch.object(self.brain, "inference", return_value=decision(action="mine", target="minecraft:stone")):
                value = self.brain.decide(self.observation)
            self.brain.record_result(self.outcome(self.observation, value, "failed"))
        state = self.brain.perceive(self.observation)
        self.assertEqual(state["lookup"][0]["failures"], 2)
        self.assertNotIn("mine", allowed_actions(state))
        moved = dict(self.observation, position=[30, 64, 10])
        self.assertIn("mine", allowed_actions(self.brain.perceive(moved)))
        changed = dict(self.observation, inventory=["minecraft:stone_pickaxe=1"])
        self.assertNotEqual(situation(changed), situation(self.observation))
        other = Brain(self.path, planning=False)
        self.assertEqual(other.perceive(self.observation)["lookup"][0]["failures"], 2)
        other.close()

    def test_lookup_records_actual_inventory_delta_once(self):
        with patch.object(self.brain, "inference", return_value=decision(action="mine", target="minecraft:stone")):
            value = self.brain.decide(self.observation)
        after = dict(self.observation, inventory=["minecraft:wooden_pickaxe=1", "minecraft:cobblestone=1"])
        observed = self.outcome(after, value)
        self.brain.record_result(observed)
        self.brain.record_result(observed)
        row = self.brain.perceive(after)["lookup"][0]
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["inventory_change"], {"minecraft:cobblestone": 1})

    def test_queued_plan_uses_fresh_preconditions_without_another_prompt(self):
        first = decision(action="mine", target="minecraft:stone", goal="Collect one cobblestone",
                         plan=[dict(action="pickup", target="minecraft:cobblestone", x=0, y=0, z=0)])
        with patch.object(self.brain, "inference", return_value=first) as model:
            value = self.brain.decide(self.observation)
            after = dict(self.observation, entities=[{"type": "minecraft:item", "item": "minecraft:cobblestone"}])
            pickup = self.brain.decide(self.outcome(after, value))
        self.assertEqual(model.call_count, 1)
        self.assertEqual(pickup["action"], "pickup")
        self.assertEqual(pickup["goal"], first["goal"])
        self.assertFalse(self.brain.steps)

    def test_plan_stops_on_failure_chat_danger_or_changed_preconditions(self):
        for change in ("failure", "chat", "danger", "missing_recipe"):
            with self.subTest(change=change):
                first = decision(plan=[dict(action="craft", target="minecraft:stick", x=0, y=0, z=0)])
                with patch.object(self.brain, "inference", side_effect=[first, decision(action="explore")]) as model:
                    value = self.brain.decide(self.observation)
                    after = self.outcome(self.observation, value, "failed" if change == "failure" else "success")
                    if change == "chat":
                        after["chat"] = ["Change plans"]
                    if change == "danger":
                        after["health"] = 4
                    result = self.brain.decide(after)
                self.assertEqual(model.call_count, 2)
                self.assertEqual(result["action"], "explore")
                self.assertFalse(self.brain.steps)

    def test_plans_are_bounded_and_cannot_embed_another_plan(self):
        step = dict(action="wait", target="", x=0, y=0, z=0)
        with self.assertRaises(ValueError):
            validate_decision(decision(plan=[step] * 9))
        with self.assertRaises(ValueError):
            validate_decision(decision(plan=[dict(step, plan=[])]))

    def test_a_destroyed_block_coordinate_is_not_replayed_as_another_block(self):
        far = validate_action(decision(action="goto", x=0, y=0, z=0), dict(self.observation, position=[300, 64, 10]))
        self.assertEqual(far["action"], "explore", "Unreachable goto becomes exploration")
        near = validate_action(decision(action="goto", x=305, y=64, z=12), dict(self.observation, position=[300, 64, 10]))
        self.assertEqual(near["action"], "goto")
        stale = validate_action(decision(action="mine", target="minecraft:stone", x=12, y=64, z=10), self.observation)
        self.assertEqual([stale[k] for k in "xyz"], [0, 0, 0], "Stale coordinates fall back to automatic selection")
        current = validate_action(decision(action="mine", target="minecraft:stone", x=11, y=64, z=10), self.observation)
        self.assertEqual([current[k] for k in "xyz"], [11, 64, 10])

    def test_generic_pickup_cannot_bypass_a_failed_specific_target(self):
        state = dict(self.observation, entities=[
            {"type": "minecraft:item", "item": "minecraft:cobblestone"},
            {"type": "minecraft:item", "item": "minecraft:spruce_log"}],
            blocked_actions=[{"action": "pickup", "target": "minecraft:spruce_log"}])
        with self.assertRaisesRegex(ValueError, "specific unblocked"):
            validate_action(decision(action="pickup"), state)
        validate_action(decision(action="pickup", target="minecraft:cobblestone"), state)

    def test_code_can_use_variables_docstrings_and_its_learned_lookup(self):
        code = '''def skill():
    """Gather stone only if this approach has not repeatedly failed."""
    target = "minecraft:stone"
    missing = 16 - count("minecraft:cobblestone")
    attempts = failures("mine", target)
    if missing > 0 and attempts < 2 and nearby(target):
        return mine(target)
    else:
        pass
    return None
'''
        state = dict(self.observation, lookup=[dict(action="mine", target="minecraft:stone", failures=1, successes=0)])
        body = parse(code)
        self.assertEqual(interpret(body, state)["action"], "mine")
        state["lookup"][0]["failures"] = 2
        self.assertIsNone(interpret(body, state))
        with self.assertRaises(ValueError):
            interpret(parse("def skill():\n    return craft(count('x'))\n"), state)

    def test_context_contains_milestones_and_preserves_inventory(self):
        observation = dict(self.observation, inventory=["minecraft:stone_pickaxe=2", "minecraft:cobblestone=16"])
        original = copy.deepcopy(observation)
        messages = self.brain.messages(observation)
        context = json.loads(messages[-1]["content"])
        self.assertTrue(context["milestones"]["stone_pickaxe_complete"])
        self.assertEqual(context["milestones"]["owned_tools"]["minecraft:stone_pickaxe"], 2)
        self.assertEqual(context["inventory"], observation["inventory"])
        self.assertIn("lookup", context)
        self.assertLessEqual(len(compact(messages).encode()) + len(compact(action_schema(observation)).encode()), PROMPT_BYTES)
        self.assertEqual(original, observation)

    def test_repeated_speech_is_suppressed_but_direct_chat_is_not(self):
        with patch.object(self.brain, "inference", side_effect=[
                decision(say="Mining stone."), decision(say="Mining stone."), decision(say="Mining stone.")]):
            first = self.brain.decide(self.observation)
            second = self.brain.decide(self.observation)
            third = self.brain.decide(dict(self.observation, chat=["What are you doing?"]))
        self.assertTrue(first["say"])
        self.assertFalse(second["say"])
        self.assertTrue(third["say"])


if __name__ == "__main__":
    unittest.main()
