"""Regression tests from the 'AI TEST 1' log: endless cobblestone mining and rejected skill code."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import Brain, action_targets, stockpile_full, validate_action
from skills import interpret, normalize, parse
from test_brain import decision

# Willow's real inventory: 36/36 slots, 704 cobblestone.
FULL = ["minecraft:cobblestone=704", "minecraft:lever=298", "minecraft:polished_granite=199",
        "minecraft:stone_pickaxe=1", "minecraft:diorite=6", "minecraft:dirt=17"]
# The looping skill from the log: minecraft:logs always counts 0, so it mined stone forever.
LOOPING_SKILL = '''def skill():
    if count("minecraft:logs") > 0:
        return None
    if nearby("minecraft:stone") and count("#pickaxes") > 0:
        return mine("minecraft:stone")
    return None
'''


def observation(inventory, **extra):
    return dict({"inventory": inventory, "free_slots": 0,
                 "nearby_blocks": [{"id": "minecraft:stone", "at": [1, 64, 0]},
                                   {"id": "minecraft:coal_ore", "at": [2, 64, 0]}]}, **extra)


class StockpileTests(unittest.TestCase):
    def test_plentiful_cobblestone_is_not_a_mine_target(self):
        obs = observation(FULL)
        self.assertIn("enough", stockpile_full(obs, "minecraft:stone"))
        self.assertEqual(action_targets(obs, "mine"), ["minecraft:coal_ore"])
        with self.assertRaisesRegex(ValueError, "Already carrying 704 minecraft:cobblestone"):
            validate_action(decision(action="mine", target="minecraft:stone"), obs)
        validate_action(decision(action="mine", target="minecraft:coal_ore"), obs)

    def test_full_inventory_blocks_drops_that_cannot_be_stored(self):
        obs = observation(["minecraft:cobblestone=64"], free_slots=0)
        self.assertIn("Inventory is full", stockpile_full(obs, "minecraft:stone"))
        self.assertEqual(stockpile_full(dict(obs, free_slots=3), "minecraft:stone"), "")

    def test_small_stock_still_mines_and_exit_can_be_dug(self):
        self.assertEqual(stockpile_full(observation(["minecraft:cobblestone=10"], free_slots=20),
                                        "minecraft:stone"), "")
        obs = observation(FULL, terrain={"exits": [{"feet": "minecraft:stone", "head": "minecraft:air",
                                                    "at": [1, 64, 0], "standable": False}]})
        self.assertIn("minecraft:stone", action_targets(obs, "mine"))
        validate_action(decision(action="mine", target="minecraft:stone", x=1, y=64, z=0), obs)

    def test_cave_wall_is_not_a_stockpile_loophole_when_she_can_walk_out(self):
        # Cave log: 195 building blocks, stone walls on two sides, an open passage on the third.
        obs = observation(FULL, position=[5, 60, -37], terrain={"exits": [
            {"feet": "minecraft:stone", "head": "minecraft:stone", "at": [5, 60, -38], "standable": False},
            {"feet": "minecraft:stone", "head": "minecraft:stone", "at": [6, 60, -37], "standable": False},
            {"feet": "minecraft:air", "head": "minecraft:air", "at": [4, 60, -37], "standable": True}]})
        self.assertEqual(action_targets(obs, "mine"), ["minecraft:coal_ore"])
        for at in ((5, 59, -38), (5, 60, -38), (0, 0, 0)):
            with self.subTest(at=at), self.assertRaisesRegex(ValueError, "that is enough"):
                validate_action(decision(action="mine", target="minecraft:stone", x=at[0], y=at[1], z=at[2]), obs)

    def test_enclosed_mine_is_redirected_to_the_wall_instead_of_rejected(self):
        obs = observation(FULL, position=[5, 60, -37], terrain={"exits": [
            {"feet": "minecraft:stone", "head": "minecraft:stone", "at": [5, 60, -38], "standable": False},
            {"feet": "minecraft:air", "head": "minecraft:stone", "at": [6, 60, -37], "standable": False}]})
        self.assertIn("minecraft:stone", action_targets(obs, "mine"))
        # The floor block under the crosshair becomes the wall in the same direction.
        floor = validate_action(decision(action="mine", target="minecraft:stone", x=5, y=59, z=-38), obs)
        self.assertEqual([floor["x"], floor["y"], floor["z"]], [5, 60, -38])
        # An overhanging head block is dug one above the exit cell.
        head = validate_action(decision(action="mine", target="minecraft:stone", x=6, y=59, z=-37), obs)
        self.assertEqual([head["x"], head["y"], head["z"]], [6, 61, -37])
        # Automatic selection would pick the floor; it is pinned to the first wall instead.
        auto = validate_action(decision(action="mine", target="minecraft:stone"), obs)
        self.assertEqual([auto["x"], auto["y"], auto["z"]], [5, 60, -38])
        # Coal is never redirected.
        coal = validate_action(decision(action="mine", target="minecraft:coal_ore", x=2, y=64, z=0), obs)
        self.assertEqual([coal["x"], coal["y"], coal["z"]], [2, 64, 0])

    def test_newlines_in_free_text_do_not_discard_a_valid_action(self):
        obs = observation(["minecraft:cobblestone=10"], free_slots=20)
        value = validate_action(decision(action="mine", target="minecraft:stone", goal="Mine.\nPlan:\n1. pickup",
                                         say="a\tb", memory="x\r\ny", waypoint="w\n"), obs)
        self.assertEqual(value["goal"], "Mine. Plan: 1. pickup")
        self.assertEqual((value["say"], value["memory"], value["waypoint"]), ("a b", "x y", "w"))
        with self.assertRaisesRegex(ValueError, "Control character in target"):
            validate_action(decision(action="mine", target="minecraft:st\none"), obs)


class SkillLanguageTests(unittest.TestCase):
    def test_fake_item_ids_are_rejected_with_fix(self):
        with self.assertRaisesRegex(ValueError, "line 2: 'minecraft:logs' is not a real item ID.*#logs"):
            parse(LOOPING_SKILL)

    def test_errors_name_the_line_and_construct(self):
        cases = {
            "def skill():\n    for b in range(3):\n        pass\n": "line 2: 'For'.*loops",
            "def skill():\n    return len(\"x\")\n": "line 2: unknown function 'len'",
            "def skill():\n    return inv[0]\n": "line 2: .*indexing like x\\[0\\]",
            "def skill():\n    return inv\n": "line 2: unknown name 'inv'",
            "def skill():\n    mine(\"minecraft:stone\")\n": "line 2: a bare call",
            "def skill(:\n": "Invalid skill syntax \\(line 1",
        }
        for code, message in cases.items():
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, message):
                parse(code)

    def test_markdown_fences_and_prose_are_stripped(self):
        raw = "Here is the skill:\n```python\ndef skill():\n\treturn explore()\n```\nDone."
        self.assertEqual(normalize(raw), "def skill():\n    return explore()\n")
        parse(normalize(raw))

    def test_common_harmless_python_is_supported(self):
        code = '''def skill():
    need = 3
    need -= 1
    step = foundation()
    if step is not None:
        return step
    return mine("minecraft:stone") if count("minecraft:cobblestone") < need else (foundation() or explore())
'''
        body = parse(code)
        obs = {"inventory": ["minecraft:cobblestone=1"]}
        self.assertEqual(interpret(body, obs)["action"], "mine")
        obs = {"inventory": ["minecraft:cobblestone=5"]}
        self.assertEqual(interpret(body, obs)["action"], "explore")


class StrategistRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = Brain(Path(self.tmp.name) / "brain.sqlite3", planning=False)

    def tearDown(self):
        self.brain.close()
        self.tmp.cleanup()

    def test_invalid_code_gets_one_repair_with_the_error(self):
        bad = {"goal": "Mine", "reason": "r", "code": "def skill():\n    for i in range(3):\n        pass\n"}
        good = {"goal": "Mine", "reason": "r", "code": "```python\ndef skill():\n    return explore()\n```"}
        with self.assertLogs("localai", level="WARNING") as logs, \
                patch.object(self.brain, "inference", side_effect=[bad, good]) as model:
            plan = self.brain.think_plan([{"role": "user", "content": "{}"}])
        self.assertEqual(model.call_count, 2)
        self.assertIn("REJECTED by the validator: line 2", model.call_args.args[0][-1]["content"])
        self.assertIn("for i in range(3)", "\n".join(logs.output))
        self.assertEqual(plan["code"], "def skill():\n    return explore()\n")
        self.assertNotIn("code_error", plan)

    def test_unrepaired_error_is_reported_to_next_strategist(self):
        bad = {"goal": "Mine", "reason": "r", "code": "def skill():\n    return len(1)\n"}
        with self.assertLogs("localai", level="WARNING"), \
                patch.object(self.brain, "inference", side_effect=[bad, dict(bad)]):
            plan = self.brain.think_plan([{"role": "user", "content": "{}"}])
        self.assertEqual(plan["code"], "")
        self.assertIn("unknown function 'len'", plan["code_error"])

    def test_existing_looping_skill_is_disabled(self):
        key = self.brain.skills.install('def skill():\n    return mine("minecraft:stone")\n')
        with self.assertLogs("localai", level="ERROR"), \
                patch.object(self.brain, "inference", return_value=decision()):
            value = self.brain.decide(observation(FULL))
        self.assertEqual(value["action"], "explore")
        self.assertEqual(self.brain.db.execute("SELECT disabled FROM skills WHERE id=?", (key,)).fetchone(), (1,))


if __name__ == "__main__":
    unittest.main()
