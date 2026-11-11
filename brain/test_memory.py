import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from server import Brain, validate_action
from test_brain import decision

HERE = [10, 64, 10]


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "brain.sqlite3"
        self.brain = Brain(self.path, planning=False)
        self.observation = {"position": HERE, "dimension": "minecraft:overworld",
                            "inventory": ["minecraft:crafting_table=1", "minecraft:oak_planks=4"],
                            "placeable": ["minecraft:crafting_table", "minecraft:oak_planks"],
                            "nearby_blocks": [{"id": "minecraft:iron_ore", "at": [12, 60, 10]},
                                              {"id": "minecraft:stone", "at": [11, 63, 10]}],
                            "crafting": {"placement_sites": [[11, 64, 10]]},
                            "entities": [], "craftable": [], "health": 20, "food": 20}

    def tearDown(self):
        self.brain.close()
        self.tmp.cleanup()

    def act(self, observation, **changes):
        with patch.object(self.brain, "inference", return_value=decision(**changes)):
            return self.brain.decide(observation)

    def outcome(self, response, status="success", observation=None, **result):
        return dict(observation or self.observation, last_result=dict(
            id=response["id"], action=response["action"], target=response["target"], status=status,
            detail="Measured", damage=0, **result))

    def place_table(self):
        placed = self.act(self.observation, action="place", target="minecraft:crafting_table", x=11, y=64, z=10)
        after = dict(self.observation, inventory=["minecraft:oak_planks=4"])
        return placed, self.outcome(placed, observation=after, block_position="11, 64, 10")

    def test_placed_blocks_are_remembered_watched_and_survive_restart(self):
        _, after = self.place_table()
        state = self.brain.messages(after)
        context = json.loads(state[-1]["content"])
        self.assertEqual(context["my_placed_blocks"],
                         [{"id": "minecraft:crafting_table", "at": [11, 64, 10], "distance": 1}])
        self.assertIn("Placed minecraft:crafting_table at [11, 64, 10]", context["memory_events"])
        response = self.act(after)
        self.assertEqual(response["watch"], [[11, 64, 10]], "The game must re-check remembered placements")
        self.brain.close()
        self.brain = Brain(self.path, planning=False)
        far = dict(self.observation, position=[200, 70, 200], nearby_blocks=[])
        context = json.loads(self.brain.messages(far)[-1]["content"])
        self.assertEqual(context["my_placed_blocks"][0]["at"], [11, 64, 10], "Placements persist forever")
        self.assertEqual(context["my_placed_totals"], {"minecraft:crafting_table": 1})

    def test_world_check_and_mining_update_placed_block_state(self):
        _, after = self.place_table()
        self.brain.perceive(after)
        broken = dict(after, remembered_blocks=[{"at": [11, 64, 10], "id": "minecraft:air"}])
        context = json.loads(self.brain.messages(broken)[-1]["content"])
        self.assertEqual(context["my_placed_blocks"], [], "A block that is gone is not reported as present")
        self.assertTrue(any("is now gone" in event for event in context["memory_events"]))
        restored = dict(after, remembered_blocks=[{"at": [11, 64, 10], "id": "minecraft:crafting_table"}])
        context = json.loads(self.brain.messages(restored)[-1]["content"])
        self.assertEqual(len(context["my_placed_blocks"]), 1)
        mine = self.act(restored, action="mine", target="minecraft:stone")
        self.brain.perceive(self.outcome(mine, block_position="11, 64, 10"))
        state = self.brain.db.execute("SELECT state FROM placed_blocks").fetchone()[0]
        self.assertEqual(state, "gone")

    def test_waypoints_notes_and_knowledge_are_saved_and_shown(self):
        response = self.act(self.observation, waypoint="home", memory="Iron ore is below the starting hill")
        self.act(self.observation, memory="Iron ore is below the starting hill")
        self.brain.perceive(self.outcome(response))
        context = json.loads(self.brain.messages(dict(self.observation, position=[40, 64, 10], nearby_blocks=[]))[-1]["content"])
        names = {w["name"]: w for w in context["waypoints"]}
        self.assertEqual(names["home"]["at"], HERE)
        self.assertEqual(names["home"]["distance"], 30)
        self.assertIn("first_spawn", names)
        self.assertEqual(context["my_notes"], ["Iron ore is below the starting hill"], "Notes are deduplicated")
        self.assertTrue(any(entry.startswith("explore: 1/1 ok") for entry in context["knowledge"]))
        self.assertIn({"id": "minecraft:iron_ore", "at": [12, 60, 10], "distance": 28},
                      context["remembered_resources"], "Useful blocks seen earlier are remembered")
        self.assertNotIn("minecraft:stone", [r["id"] for r in context["remembered_resources"]])
        self.assertEqual(self.brain.db.execute("SELECT note FROM waypoints WHERE name='home'").fetchone()[0],
                         "Find trees")

    def test_repeated_waypoint_is_saved_once_and_moves_when_resaved(self):
        self.act(self.observation, waypoint="home")
        self.act(self.observation, waypoint="home")
        moved = dict(self.observation, position=[20, 65, 10])
        self.act(moved, waypoint="home")
        events = [r[0] for r in self.brain.db.execute("SELECT text FROM events WHERE kind='waypoint'")]
        self.assertEqual(events, ["Saved waypoint home at [10, 64, 10]"],
                         "Re-saving a name seconds later must not let the place drift every decision")
        self.brain.db.execute("UPDATE waypoints SET updated=updated-600 WHERE name='home'")
        self.act(moved, waypoint="home")
        events = [r[0] for r in self.brain.db.execute("SELECT text FROM events WHERE kind='waypoint'")]
        self.assertEqual(events, ["Saved waypoint home at [10, 64, 10]", "Saved waypoint home at [20, 65, 10]"])
        # A player request moves it immediately.
        self.act(dict(self.observation, chat=["save this as home"]), waypoint="home")
        self.assertEqual(self.brain.db.execute("SELECT x,y,z FROM waypoints WHERE name='home'").fetchone(), (10, 64, 10))

    def test_invalid_waypoint_name_is_ignored(self):
        self.act(self.observation, waypoint="last_position")
        self.act(self.observation, waypoint="bad/name")
        names = [r[0] for r in self.brain.db.execute("SELECT name FROM waypoints")]
        self.assertEqual(sorted(names), ["first_spawn", "last_position"])

    def test_discoveries_and_export(self):
        self.brain.perceive(self.observation)
        self.brain.memory.export()
        data = json.loads((self.path.parent / "memory.json").read_text())
        self.assertEqual(data["discovered_items"], ["minecraft:crafting_table", "minecraft:oak_planks"])
        self.assertEqual({w["name"] for w in data["waypoints"]}, {"first_spawn", "last_position"})

    def test_far_goto_heads_toward_destination(self):
        value = validate_action(decision(action="goto", x=300, y=70, z=-40), self.observation)
        self.assertEqual((value["action"], value["x"], value["y"], value["z"]), ("explore", 300, 70, -40))

    def test_batch_seen_blocks_pruning(self):
        now = time.time()
        dim = "minecraft:overworld"
        self.brain.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)", (dim, "minecraft:stone", 10, 64, 11, now))
        self.brain.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)", (dim, "minecraft:dirt", 11, 64, 10, now))
        self.brain.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)", (dim, "minecraft:iron_ore", 12, 64, 10, now))
        self.brain.db.execute("INSERT OR REPLACE INTO placed_blocks VALUES (?,?,?,?,?,?,?,'present')",
                              (dim, 10, 64, 11, "minecraft:stone", now, now))

        # Observation where [12, 64, 10] disappeared from nearby_blocks
        obs = dict(self.observation,
                   position=[10, 64, 10],
                   dimension=dim,
                   nearby_blocks=[{"id": "minecraft:dirt", "at": [11, 64, 10]}])
        self.brain.memory.observe(obs)

        remaining = set(self.brain.db.execute("SELECT x,y,z FROM seen_blocks WHERE dimension=?", (dim,)).fetchall())
        self.assertNotIn((12, 64, 10), remaining, "Disappeared unplaced block must be pruned by batch delete")
        self.assertIn((10, 64, 11), remaining, "Placed block must be preserved even if absent from nearby_blocks")
        self.assertIn((11, 64, 10), remaining, "Present block must remain in seen_blocks")


if __name__ == "__main__":
    unittest.main()
