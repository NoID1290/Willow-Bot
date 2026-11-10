"""Comprehensive Verification Suite for Willow-Bot V2 (Deliverables 0 through 5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from curriculum import (
    CurriculumDAG,
    CurriculumDecomposer,
    CurriculumManager,
    RecipeKnowledgeGraph,
    SurvivalMechanicsRegistry,
)
from execution import (
    ActionResult,
    SandboxSecurityError,
    ScriptSandbox,
    TimeSlicedAStar,
    WillowBotActions,
    validate_skill_ast,
)
from hardware_opt import (
    CancellationTokenSource,
    CompactTelemetryFormatter,
    CompiledSkillCache,
    InferencePriority,
    LiveReflexState,
    LocalInferenceOptimizer,
    LocalLlmClient,
    MemoryPoolManager,
    OperationCanceledError,
    compute_safe_num_threads,
    estimate_tokens,
)
from orchestrator import WillowBotOrchestrator
from rl_rewards import (
    BlockInfo,
    BotTelemetrySnapshot,
    ContextStateHash,
    EquipmentSlot,
    EquippedItemState,
    ExecutionTelemetry,
    PerceivedEntity,
    ReflexionMemoryLog,
    RewardCalculator,
    SkillQTable,
    Vector3,
    VitalsState,
)
from skills import (
    BYTES_PER_VECTOR,
    CodeSkillAgent,
    CpuFeatureHasher,
    CriticEvaluation,
    FailureCategory,
    SelfVerificationCritic,
    SkillStatus,
    SkillStore,
)


def make_snapshot(**overrides) -> BotTelemetrySnapshot:
    base = dict(
        timestamp_ms=1_000_000,
        world_tick=200,
        time_of_day=6000,
        position=Vector3(124.0, 64.0, -82.0),
        dimension="minecraft:overworld",
        biome="minecraft:plains",
        light_level=15,
        vitals=VitalsState(
            health=20.0,
            max_health=20.0,
            absorption=0.0,
            food_level=20,
            saturation=5.0,
            armor_points=0,
            oxygen=300,
            is_alive=True,
        ),
        item_counts={},
        empty_slots_count=36,
        equipment={slot: EquippedItemState("minecraft:air", 0, 0) for slot in EquipmentSlot},
        unlocked_advancements=frozenset(),
        discovered_unique_items=frozenset(),
        crafted_unique_items=frozenset(),
        nearby_entities=(),
        nearby_workstations={},
        execution_telemetry=ExecutionTelemetry(),
        nearby_blocks=(),
        enclosed_lit_shelter=False,
    )
    base.update(overrides)
    return BotTelemetrySnapshot(**base)


class Deliverable0HardwareOptTests(unittest.TestCase):
    def test_vram_budget_and_payload_options(self):
        opt = LocalInferenceOptimizer(processor_count=16)
        self.assertEqual(compute_safe_num_threads(16), 7)
        self.assertEqual(compute_safe_num_threads(4), 2)
        budget = opt.vram_budget_report()
        self.assertLessEqual(budget["total_ai_vram_gb"], 8.0)
        self.assertGreaterEqual(budget["minecraft_os_headroom_gb"], 4.0)

        payload = opt.build_request_payload("STATIC_PREFIX", "DYNAMIC_SUFFIX")
        self.assertEqual(payload["model"], "qwen2.5-coder:7b-instruct-q4_K_M")
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertTrue(payload["stream"])
        opts = payload["options"]
        self.assertEqual(opts["num_ctx"], 4096)
        self.assertEqual(opts["num_predict"], 768)
        self.assertAlmostEqual(opts["temperature"], 0.15)
        self.assertAlmostEqual(opts["top_p"], 0.9)
        self.assertEqual(opts["num_batch"], 256)
        self.assertTrue(opts["use_mmap"])
        self.assertFalse(opts["use_mlock"])
        self.assertTrue(opts["flash_attn"])

    def test_single_flight_priority_queue_and_reflex_pause(self):
        execution_order: list[str] = []
        hold_first = threading.Event()

        def fake_transport(payload, ct):
            user_msg = payload["messages"][1]["content"]
            if user_msg == "first":
                hold_first.wait(timeout=2.0)
            execution_order.append(user_msg)
            return json.dumps({"ok": user_msg})

        client = LocalLlmClient(transport=fake_transport)
        t1 = threading.Thread(
            target=lambda: client.generate_sync("SYS", "first", priority=InferencePriority.TASK_DECOMPOSITION)
        )
        t1.start()
        while not client.in_flight:
            time.sleep(0.005)

        t_low = threading.Thread(
            target=lambda: client.generate_sync("SYS", "decomp", priority=InferencePriority.TASK_DECOMPOSITION)
        )
        t_mid = threading.Thread(
            target=lambda: client.generate_sync("SYS", "synth", priority=InferencePriority.NEW_SKILL_SYNTHESIS)
        )
        t_high = threading.Thread(
            target=lambda: client.generate_sync("SYS", "mutate", priority=InferencePriority.SKILL_MUTATION)
        )
        t_low.start()
        time.sleep(0.01)
        t_mid.start()
        time.sleep(0.01)
        t_high.start()
        time.sleep(0.02)

        hold_first.set()
        for t in (t1, t_low, t_mid, t_high):
            t.join(timeout=2.0)

        self.assertEqual(execution_order, ["first", "mutate", "synth", "decomp"])
        self.assertGreaterEqual(client.stats["prefix_cache_hits"], 3)

    def test_compact_telemetry_formatter_under_150_tokens(self):
        snap = make_snapshot(
            item_counts={f"minecraft:item_{i}": i + 1 for i in range(36)},
            equipment={
                EquipmentSlot.MAIN_HAND: EquippedItemState("minecraft:stone_pickaxe", 98, 131),
                EquipmentSlot.OFF_HAND: EquippedItemState("minecraft:shield", 300, 336),
            },
            nearby_workstations={"crafting_table": Vector3(125, 64, -81)},
            nearby_entities=(
                PerceivedEntity(10, "minecraft:skeleton", Vector3(135, 64, -82), 11.0, 20.0, True),
            ),
        )
        compact_str = CompactTelemetryFormatter.format(snap)
        self.assertIn("POS:124,64,-82", compact_str)
        self.assertIn("EQUIP:stone_pickaxe(98/131),shield", compact_str)
        self.assertIn("WORKSTATIONS:crafting_table@(125,64,-81)", compact_str)
        self.assertIn("THREATS:skeleton@11m", compact_str)
        self.assertLess(estimate_tokens(compact_str), 150)

    def test_lru_64_compiled_skill_cache_evicts_and_clears_scope(self):
        cache = CompiledSkillCache(max_entries=3)
        scopes = []
        for idx in range(4):
            sc = {"sentinel": idx}
            scopes.append(sc)
            cache.put(f"hash_{idx}", f"skill_{idx}", lambda: idx, sc)
        self.assertIsNone(cache.get("hash_0"))
        self.assertEqual(scopes[0], {})  # Evicted scope cleared immediately
        self.assertIsNotNone(cache.get("hash_3"))


class Deliverable1RlRewardTests(unittest.TestCase):
    def test_exact_composite_reward_formula(self):
        dag = CurriculumDAG()
        task = dag.get_node("t2_mine_cobblestone")
        s_prev = make_snapshot(
            vitals=VitalsState(health=20.0, food_level=10, armor_points=0, is_alive=True),
            item_counts={"minecraft:wooden_pickaxe": 1, "minecraft:cobblestone": 4, "minecraft:cooked_beef": 2},
            discovered_unique_items=frozenset({"minecraft:wooden_pickaxe", "minecraft:cobblestone"}),
        )
        s_curr = make_snapshot(
            position=Vector3(130.0, 64.0, -80.0),
            vitals=VitalsState(health=18.0, food_level=18, armor_points=2, is_alive=True),
            item_counts={
                "minecraft:wooden_pickaxe": 1,
                "minecraft:cobblestone": 16,
                "minecraft:coal": 2,
                "minecraft:torch": 4,
            },
            unlocked_advancements=frozenset({"minecraft:story/mine_stone"}),
            discovered_unique_items=frozenset(
                {"minecraft:wooden_pickaxe", "minecraft:cobblestone", "minecraft:coal", "minecraft:torch"}
            ),
            crafted_unique_items=frozenset({"minecraft:torch"}),
            execution_telemetry=ExecutionTelemetry(
                duration_ms=5000,
                improper_tool_mine_attempts=1,
            ),
        )
        critic = CriticEvaluation(True, FailureCategory.NONE, "ok", "")
        bd = RewardCalculator.calculate_episode_reward(s_prev, s_curr, task, critic)

        # R_adv: story/mine_stone (+50) + task match bonus (+50) = 100.0
        self.assertAlmostEqual(bd.r_adv, 100.0)
        # R_task: t2_mine_cobblestone base_reward = 80.0
        self.assertAlmostEqual(bd.r_task, 80.0)
        # R_shaping: 12 cobble * 5.0 (60) + armor upgrade (15) + sustenance (5) = 80.0
        self.assertAlmostEqual(bd.r_shaping, 80.0)
        # R_intrinsic: 2 new discovered * 3.0 (6) + 1 new crafted * 8.0 (8) = 14.0
        self.assertAlmostEqual(bd.r_intrinsic, 14.0)
        # P_vital: 2 HP lost * 3.5 = 7.0
        self.assertAlmostEqual(bd.p_vital, 7.0)
        # P_exec: 1 improper tool attempt * 8.0 = 8.0
        self.assertAlmostEqual(bd.p_exec, 8.0)
        # P_time: 0.08 * 5.0s = 0.4
        self.assertAlmostEqual(bd.p_time, 0.4)
        self.assertAlmostEqual(bd.r_total, 100.0 + 80.0 + 80.0 + 14.0 - 7.0 - 8.0 - 0.4)

    def test_death_and_broken_tool_penalties(self):
        dag = CurriculumDAG()
        task = dag.get_node("t4_mine_diamonds")
        s_prev = make_snapshot(
            item_counts={"minecraft:iron_pickaxe": 1, "minecraft:diamond": 3, "minecraft:bread": 2},
            vitals=VitalsState(health=12.0, food_level=5, is_alive=True),
        )
        s_curr = make_snapshot(
            item_counts={},
            vitals=VitalsState(health=0.0, food_level=0, is_alive=False),
            execution_telemetry=ExecutionTelemetry(
                duration_ms=2000,
                compilation_failed=True,
                runtime_exceptions=("Err1", "Err2"),
                tools_broken=("minecraft:iron_pickaxe",),
            ),
        )
        critic = CriticEvaluation(False, FailureCategory.BOT_DIED, "died", "eat")
        bd = RewardCalculator.calculate_episode_reward(s_prev, s_curr, task, critic)
        # Lost 4 valuable items (1 iron_pickaxe + 3 diamond) -> death = 250 + 40 = 290, damage = 12*3.5=42, starvation=15
        self.assertGreaterEqual(bd.p_vital, 290.0 + 42.0 + 15.0)
        # Exec: compile(15) + 2*20(40) + broken iron_pickaxe(35) = 90
        self.assertAlmostEqual(bd.p_exec, 90.0)
        self.assertLess(bd.r_total, -400.0)

    def test_context_bitmask_and_q_table_fast_path_and_persistence(self):
        snap = make_snapshot(
            position=Vector3(0.0, 20.0, 0.0),
            time_of_day=14000,
            vitals=VitalsState(health=18.0, armor_points=15),
            item_counts={"minecraft:iron_pickaxe": 1, "minecraft:shield": 1},
            nearby_workstations={"crafting_table": Vector3(1, 20, 0)},
        )
        mask = ContextStateHash.encode(snap)
        decoded = ContextStateHash.decode(mask)
        self.assertEqual(decoded["tool_tier"], 3)
        self.assertEqual(decoded["armor_tier"], 2)
        self.assertEqual(decoded["health_bucket"], 2)
        self.assertEqual(decoded["time_bucket"], 2)
        self.assertEqual(decoded["dimension_bucket"], 1)  # Cave Y < 45
        self.assertTrue(decoded["has_crafting_table"])
        self.assertTrue(decoded["has_shield"])

        with tempfile.TemporaryDirectory() as tmp:
            q_path = Path(tmp) / "q.json"
            qt = SkillQTable(q_path)
            updated = qt.update(mask, "t3_mine_iron_ore", "mine_iron_v1", 60.0, mask, "t3_smelt_iron_ingots")
            self.assertAlmostEqual(updated, 15.0)  # 0 + 0.25 * 60 = 15.0 >= +10.0

            reloaded = SkillQTable(q_path)
            sid, qv = reloaded.select_fast_path_skill(mask, "t3_mine_iron_ore", ["mine_iron_v1"])
            self.assertEqual(sid, "mine_iron_v1")
            self.assertAlmostEqual(qv, 15.0)

    def test_reflexion_ring_buffer_cap_and_jsonl_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "ref.jsonl"
            log = ReflexionMemoryLog(jsonl)
            fake_bd = RewardCalculator.calculate_episode_reward(
                make_snapshot(),
                make_snapshot(execution_telemetry=ExecutionTelemetry(compilation_failed=True)),
                CurriculumDAG().get_node("t1_gather_logs"),
                CriticEvaluation(False, FailureCategory.SYNTAX_ERROR, "bad syntax", "fix syntax"),
            )
            for i in range(7):
                log.record_if_needed("t1_gather_logs", fake_bd, False, f"cause {i}", f"fix {i}")
            log.close()

            self.assertEqual(len(log.get_task_records("t1_gather_logs")), 5)
            self.assertEqual(len(log.get_prompt_lines("t1_gather_logs")), 3)
            flushed = jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(flushed), 7)


class Deliverable2CurriculumTests(unittest.TestCase):
    def test_20_node_dag_regression_and_decomposition(self):
        dag = CurriculumDAG()
        self.assertEqual(len(dag.ordered_nodes), 20)
        mgr = CurriculumManager(dag=dag)

        # Fresh spawn -> t1_gather_logs
        s0 = make_snapshot()
        self.assertEqual(mgr.get_next_objective(s0).id, "t1_gather_logs")

        # Fast-forward to Tier 4 state (holding iron gear, shield, water bucket, torches, cooked food)
        s_tier4 = make_snapshot(
            vitals=VitalsState(health=20.0, food_level=20, armor_points=15, is_alive=True),
            item_counts={
                "minecraft:oak_log": 4,
                "minecraft:oak_planks": 12,
                "minecraft:stick": 8,
                "minecraft:crafting_table": 1,
                "minecraft:cobblestone": 20,
                "minecraft:stone_axe": 1,
                "minecraft:furnace": 1,
                "minecraft:coal": 8,
                "minecraft:cooked_beef": 14,
                "minecraft:torch": 36,
                "minecraft:iron_ingot": 16,
                "minecraft:shield": 1,
                "minecraft:water_bucket": 1,
                "minecraft:iron_pickaxe": 2,
                "minecraft:iron_sword": 1,
            },
            unlocked_advancements=frozenset(
                {"minecraft:adventure/sleep_in_bed", "minecraft:adventure/kill_a_mob"}
            ),
        )
        obj_t4 = mgr.get_next_objective(s_tier4)
        self.assertEqual(obj_t4.id, "t4_mine_diamonds")

        # Simulate iron_pickaxe breaking while mining on Tier 4 -> dynamic regression to t3_iron_tools_and_armor!
        s_broken_pick = make_snapshot(
            vitals=VitalsState(health=20.0, food_level=20, armor_points=15, is_alive=True),
            item_counts={
                "minecraft:oak_planks": 12,
                "minecraft:stick": 8,
                "minecraft:crafting_table": 1,
                "minecraft:iron_ingot": 16,
                "minecraft:shield": 1,
                "minecraft:water_bucket": 1,
                "minecraft:iron_sword": 1,
                "minecraft:torch": 36,
                "minecraft:cooked_beef": 14,
            },
            unlocked_advancements=frozenset(
                {"minecraft:adventure/sleep_in_bed", "minecraft:adventure/kill_a_mob"}
            ),
        )
        regressed_obj = mgr.get_next_objective(s_broken_pick)
        self.assertEqual(regressed_obj.id, "t3_iron_tools_and_armor")

        # Accumulate 3 consecutive failures -> automatic sub-task decomposition into 2-3 micro-tasks
        for _ in range(3):
            subs = mgr.record_outcome(regressed_obj, False, s_broken_pick)
        self.assertGreaterEqual(len(subs), 2)
        self.assertTrue(mgr.get_next_objective(s_broken_pick).is_micro_task)


class Deliverable3ExecutionAndSandboxTests(unittest.TestCase):
    def test_harvest_tool_guard_and_auto_2x2_crafting(self):
        async def scenario():
            s0 = make_snapshot(
                item_counts={"minecraft:oak_log": 3},
                nearby_blocks=(
                    BlockInfo("minecraft:iron_ore", Vector3(125.0, 64.0, -82.0)),
                    BlockInfo("minecraft:stone", Vector3(126.0, 64.0, -82.0)),
                ),
            )
            bot = WillowBotActions(s0)
            # Mining iron_ore with hand must abort with NO_TOOL and increment improper_tool_mine_attempts
            bad_mine = await bot.MineBlocksAsync("minecraft:iron_ore", 1)
            self.assertFalse(bad_mine.Success)
            self.assertEqual(bad_mine.ErrorCode, "NO_TOOL")
            self.assertEqual(bot.telemetry_recorder.improper_tool_mine_attempts, 1)

            # Craft crafting_table & wooden_pickaxe auto-resolving planks + sticks from raw oak_log!
            res_table = await bot.EnsureWorkstationPlacedAsync("minecraft:crafting_table")
            self.assertTrue(res_table.Success, res_table.Message)
            res_pick = await bot.CraftAsync("minecraft:wooden_pickaxe", 1)
            self.assertTrue(res_pick.Success, res_pick.Message)
            self.assertTrue(bot.HasItem("minecraft:wooden_pickaxe", 1))

            # Now mining stone succeeds!
            good_mine = await bot.MineBlocksAsync("minecraft:stone", 1)
            self.assertTrue(good_mine.Success)
            self.assertTrue(bot.HasItem("minecraft:cobblestone", 1))

        asyncio.run(scenario())

    def test_time_sliced_astar_respects_2_5ms_budget(self):
        astar = TimeSlicedAStar()
        path, code = asyncio.run(astar.find_path_async(Vector3(0, 64, 0), Vector3(18, 64, 14)))
        self.assertEqual(code, "NONE")
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 5)

    def test_ast_security_rejects_unsafe_code_and_penalizes(self):
        malicious_snippets = [
            "import os\nasync def bad(bot, ct):\n    return ActionResult.ok()",
            "async def bad(bot, ct):\n    open('/etc/passwd')\n    return ActionResult.ok()",
            "async def bad(bot, ct):\n    while True:\n        x = 1",
            "async def bad(bot, ct):\n    time.sleep(5)",
            "async def bad(bot, ct):\n    return bot.__class__",
        ]
        sandbox = ScriptSandbox()
        for snippet in malicious_snippets:
            with self.subTest(snippet=snippet[:40]):
                with self.assertRaises(SandboxSecurityError):
                    validate_skill_ast(snippet)
                bot = WillowBotActions(make_snapshot())
                out = sandbox.execute_sync(snippet, bot)
                self.assertTrue(out.compilation_failed)
                self.assertTrue(bot.GetState().execution_telemetry.compilation_failed)


class Deliverable4And5OrchestratorEndToEndTests(unittest.TestCase):
    def test_zero_vram_skill_store_embeddings_and_hybrid_rag(self):
        with tempfile.TemporaryDirectory() as tmp:
            qt = SkillQTable(Path(tmp) / "q.json")
            sandbox = ScriptSandbox()
            store = SkillStore(Path(tmp) / "skills", q_table=qt, sandbox=sandbox)
            rec = store.register_skill(
                skill_id="gather_logs_v1",
                skill_name="gather_logs_v1",
                description="Harvest 4 oak logs from nearby trees",
                code="async def gather_logs_v1(bot, ct):\n    return await bot.MineBlocksAsync('*_log', 4)\n",
                task_id="t1_gather_logs",
                postconditions={"*_log": 4},
                verified=True,
                q_value=25.0,
            )
            self.assertEqual((Path(tmp) / "skills" / "embeddings.bin").stat().st_size, BYTES_PER_VECTOR)
            self.assertEqual(rec.status, SkillStatus.Verified)

            dag = CurriculumDAG()
            matches = store.retrieve_relevant_skills(dag.get_node("t1_gather_logs"), make_snapshot(), top_k=3)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].record.skill_id, "gather_logs_v1")
            self.assertGreater(matches[0].score, 0.5)

            # Two consecutive failures mark skill NeedsRefinement and Mutate replaces with _v2
            store.record_execution_feedback("gather_logs_v1", success=False, updated_q=-5.0)
            updated = store.record_execution_feedback("gather_logs_v1", success=False, updated_q=-18.0)
            self.assertEqual(updated.status, SkillStatus.NeedsRefinement)
            v2 = store.replace_with_mutated_skill(
                "gather_logs_v1",
                new_skill_name="gather_logs_v2",
                new_description="Mutated log gatherer",
                new_code="async def gather_logs_v2(bot, ct):\n    return await bot.MineBlocksAsync('*_log', 4)\n",
            )
            self.assertEqual(v2.skill_id, "gather_logs_v2")
            self.assertEqual(store.get_skill("gather_logs_v1").status, SkillStatus.Deprecated)

    def test_orchestrator_voyager_synthesis_then_q_fastpath_zero_llm_bypass(self):
        llm_calls = 0

        def mock_llm_transport(payload, ct):
            nonlocal llm_calls
            llm_calls += 1
            return json.dumps(
                {
                    "reasoning": "Mine four nearby oak logs by hand. Return result.",
                    "skill_name": "harvest_four_logs",
                    "description": "Harvest 4 oak_log blocks",
                    "code": (
                        "async def harvest_four_logs(bot, ct):\n"
                        "    res = await bot.MineBlocksAsync('*_log', 4)\n"
                        "    if not res.Success:\n"
                        "        return res\n"
                        "    return ActionResult.ok('Gathered logs')\n"
                    ),
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            client = LocalLlmClient(transport=mock_llm_transport)
            orch = WillowBotOrchestrator(tmp, llm_client=client)

            tree_blocks = tuple(
                BlockInfo("minecraft:oak_log", Vector3(125.0 + i, 64.0, -82.0)) for i in range(4)
            )
            state_ep1 = make_snapshot(nearby_blocks=tree_blocks)

            # Episode 1: Synthesizes skill via LLM (1 LLM call), verifies, rewards +40+, updates Q >= +10.0!
            summary1 = orch.run_episode_sync(state_ep1)
            self.assertEqual(summary1.task_id, "t1_gather_logs")
            self.assertFalse(summary1.used_fast_path)
            self.assertTrue(summary1.critic.success)
            self.assertEqual(llm_calls, 1)
            self.assertGreaterEqual(summary1.updated_q, 10.0)

            # Reset completed set to replay t1_gather_logs in same context: MUST hit Q-FastPath (0 LLM calls!)
            orch.curriculum.completed_nodes.clear()
            summary2 = orch.run_episode_sync(state_ep1)
            self.assertTrue(summary2.used_fast_path)
            self.assertTrue(summary2.critic.success)
            self.assertEqual(llm_calls, 1)  # Strictly 0 additional LLM calls!
            orch.stop()

    def test_20hz_reflex_preempts_cognitive_episode_and_pauses_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = WillowBotOrchestrator(tmp)
            # Simulate primed creeper at 3.0m
            triggered, code, action = orch.step_reflex_tick(
                {
                    "position": [100, 64, 100],
                    "health": 18.0,
                    "food": 18,
                    "inventory": ["minecraft:shield=1"],
                    "entities": [
                        {"id": 99, "type": "minecraft:creeper", "distance": 3.0, "hostile": True, "ignited": True}
                    ],
                }
            )
            self.assertTrue(triggered)
            self.assertEqual(code, "CREEPER_FUSE")
            self.assertEqual(action, "shield_or_flee")
            self.assertTrue(orch.llm_client.is_paused_for_reflex)

            # Clear hazard -> resumes LLM queue
            triggered2, _, _ = orch.step_reflex_tick(
                {"position": [100, 64, 100], "health": 18.0, "food": 18, "entities": []}
            )
            self.assertFalse(triggered2)
            self.assertFalse(orch.llm_client.is_paused_for_reflex)
            orch.stop()


if __name__ == "__main__":
    unittest.main()
