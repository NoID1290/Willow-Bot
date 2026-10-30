import time
import unittest

from multimodal import ConcurrentPipeline, MultimodalState


class MultimodalTests(unittest.TestCase):
    def test_state_snapshot_and_threadsafe_updates(self):
        state = MultimodalState()
        state.update_sight({"facing": "North", "looking_at": {"id": "minecraft:oak_log"}})
        state.update_hearing({"acoustic_summary": "Footsteps ahead"})
        state.update_doing({"action": "mine", "target": "minecraft:oak_log"})

        snap = state.snapshot()
        self.assertEqual(snap["seeing"]["facing"], "North")
        self.assertEqual(snap["listening"]["acoustic_summary"], "Footsteps ahead")
        self.assertEqual(snap["doing"]["action"], "mine")

    def test_pipeline_concurrent_enrichment(self):
        pipeline = ConcurrentPipeline(model="lfm3:1.7b")
        try:
            obs = {
                "position": [0, 64, 0],
                "sight": {
                    "facing": "East",
                    "looking_at": {"id": "minecraft:stone", "distance": 2.5},
                    "retina": ["[ . . + . . ]"],
                    "visible_entities": [{"id": 42, "type": "minecraft:cow", "distance": 3.0}],
                },
                "hearing": {
                    "recent_sounds": [{"sound": "minecraft:block.stone.break", "distance": 2.5, "urgency": "NOTICE"}],
                    "active_alerts": [],
                    "acoustic_summary": "1 recent ambient/footstep sounds heard.",
                },
                "doing": {
                    "action": "mine",
                    "target": "minecraft:stone",
                    "progress": "Mining stone",
                    "ticks_active": 15,
                },
            }
            pipeline.ingest(obs)
            # Give background workers a slice of time
            time.sleep(0.05)

            enriched = pipeline.enrich(obs)
            self.assertIn("seeing", enriched)
            self.assertIn("listening", enriched)
            self.assertIn("doing", enriched)
            self.assertEqual(enriched["seeing"]["facing"], "East")
            self.assertEqual(enriched["doing"]["action"], "mine")
        finally:
            pipeline.close()

    def test_emergency_audio_reflex_for_creeper(self):
        pipeline = ConcurrentPipeline(model="lfm3:1.7b")
        try:
            creeper_obs = {
                "hearing": {
                    "active_alerts": [
                        {
                            "sound": "minecraft:entity.creeper.primed",
                            "distance": 3.2,
                            "bearing": "behind",
                            "urgency": "CRITICAL",
                        }
                    ],
                    "recent_sounds": [],
                }
            }
            reflex = pipeline.check_reflex(creeper_obs)
            self.assertIsNotNone(reflex, "Creeper primed sound within 5m must trigger emergency reflex")
            self.assertEqual(reflex["action"], "escape")
            self.assertTrue(reflex.get("reflex"))
            self.assertIn("Creeper", reflex["goal"])
        finally:
            pipeline.close()

    def test_omni_synthesis_integration(self):
        pipeline = ConcurrentPipeline(model="lfm3:1.7b")
        try:
            pipeline.state.update_omni("Vision confirms clear terrain; sounds indicate safe ambient cave.")
            obs = {"position": [0, 64, 0]}
            enriched = pipeline.enrich(obs)
            self.assertIn("lfm_omni_synthesis", enriched)
            self.assertEqual(enriched["lfm_omni_synthesis"], "Vision confirms clear terrain; sounds indicate safe ambient cave.")
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
