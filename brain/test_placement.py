import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from server import (Brain, DecisionRejected, Handler, HTTPServer, PROMPT_BYTES,
                    action_schema, allowed_actions, compact, validate_action)
from test_brain import decision


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.observation = {
            "position": [-591, 85, -22], "health": 20, "food": 20,
            "inventory": ["minecraft:cobblestone_slab=18", "minecraft:dirt=12", "minecraft:stone_pickaxe=2"],
            "placeable": ["minecraft:cobblestone_slab", "minecraft:dirt"],
            "crafting": {"placement_sites": [[-590, 85, -22], [-591, 85, -23]]},
            "blocked_actions": [{"action": "place", "target": "minecraft:cobblestone_slab"}],
        }

    def test_blocked_slab_is_excluded_from_all_placement_schema_branches(self):
        before = copy.deepcopy(self.observation)
        branches = action_schema(self.observation)["oneOf"]
        placements = [branch for branch in branches if "place" in branch["properties"]["action"]["enum"]]
        self.assertEqual(len(placements), 2)
        for branch, position in zip(placements, self.observation["crafting"]["placement_sites"]):
            self.assertEqual(branch["properties"]["action"]["enum"], ["place"])
            self.assertEqual(branch["properties"]["target"]["enum"], ["minecraft:dirt"])
            for key, value in zip(("x", "y", "z"), position):
                self.assertEqual(branch["properties"][key]["enum"], [value])
        self.assertEqual(before, self.observation)

    def test_placement_is_disabled_without_sites_or_unblocked_inventory_blocks(self):
        for changes in (
                {"crafting": {"placement_sites": []}},
                {"placeable": []},
                {"blocked_actions": [{"action": "place", "target": ""}]}):
            self.assertNotIn("place", allowed_actions(dict(self.observation, **changes)))

    def test_all_block_placement_checks_inventory_and_current_site(self):
        validate_action(decision(action="place", target="minecraft:dirt", x=-590, y=85, z=-22), self.observation)
        with self.assertRaisesRegex(ValueError, "usable site"):
            validate_action(decision(action="place", target="minecraft:dirt", x=-591, y=85, z=-22), self.observation)
        with self.assertRaisesRegex(ValueError, "owned unblocked"):
            validate_action(decision(action="place", target="minecraft:stone_pickaxe", x=-590, y=85, z=-22),
                            self.observation)

    def test_expanded_schema_and_placement_information_stay_within_context_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "brain.sqlite3", planning=False)
            try:
                observation = dict(self.observation, nearby_blocks=[
                    {"id": "minecraft:stone", "at": [n, 85, 0]} for n in range(150)],
                    craftable=["minecraft:stick", "minecraft:cobblestone_slab"],
                    entities=[{"id": 1, "type": "minecraft:zombie"}, {"id": 2, "type": "minecraft:item", "item": "minecraft:stone"}])
                messages = brain.messages(observation)
                self.assertLessEqual(len(compact(messages).encode()) + len(compact(action_schema(observation)).encode()),
                                     PROMPT_BYTES)
                self.assertEqual(json.loads(messages[-1]["content"])["placeable"], observation["placeable"])
            finally:
                brain.close()

    def test_two_invalid_plans_are_a_rejection_not_a_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "brain.sqlite3", planning=False)
            try:
                invalid = decision(action="place", target="minecraft:cobblestone_slab", x=-590, y=85, z=-22)
                with (self.assertLogs("localai", level="WARNING"),
                      patch.object(brain.experience, "context", return_value=([], self.observation["blocked_actions"])),
                      patch.object(brain, "inference", return_value=invalid) as model):
                    with self.assertRaisesRegex(DecisionRejected, "failed repeatedly"):
                        brain.decide(self.observation)
                self.assertEqual(model.call_count, 2)
                self.assertEqual(brain.db.execute("SELECT count(*) FROM decisions").fetchone()[0], 0)
                with patch.object(brain, "inference", side_effect=OSError("Ollama offline")):
                    with self.assertRaisesRegex(OSError, "offline"):
                        brain.decide(self.observation)
            finally:
                brain.close()

    def test_http_rejection_is_422_and_service_still_handles_next_request(self):
        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            server.token = "test-token"
            server.brain = SimpleNamespace(decide=Mock(side_effect=[
                DecisionRejected("Placement blocked"), decision(), OSError("Ollama offline")]))
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                def request():
                    return urllib.request.urlopen(urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/decide", data=b"{}",
                        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"}), timeout=3)
                with self.assertLogs("localai", level="WARNING"), self.assertRaises(urllib.error.HTTPError) as rejection:
                    request()
                self.assertEqual(rejection.exception.code, 422)
                self.assertEqual(json.load(rejection.exception), {"error": "decision_rejected", "detail": "Placement blocked"})
                rejection.exception.close()
                with request() as response:
                    self.assertEqual(json.load(response)["action"], "explore")
                with self.assertLogs("localai", level="ERROR"), self.assertRaises(urllib.error.HTTPError) as outage:
                    request()
                self.assertEqual(outage.exception.code, 503)
                outage.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=3)

    def test_placement_under_feet_allowed_when_in_sites(self):
        obs = dict(self.observation, crafting={"placement_sites": [[-590, 85, -22], [-591, 85, -22]]})
        decision_val = decision(action="place", target="minecraft:dirt", x=-591, y=85, z=-22)
        validated = validate_action(decision_val, obs)
        self.assertEqual((validated["action"], validated["x"], validated["y"], validated["z"]),
                         ("place", -591, 85, -22))


if __name__ == "__main__":
    unittest.main()
