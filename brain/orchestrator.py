"""Module 5: Dual-Loop Master Orchestrator (20Hz Zero-LLM Survival Reflex + Slow RL/Voyager Cognitive Loop)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from curriculum.curriculum_dag import CurriculumDAG, CurriculumNode
from curriculum.curriculum_manager import CurriculumManager
from execution.bot_actions import ActionCommandExecutor, ActionResult, WillowBotActions
from execution.sandbox import SandboxExecutionOutcome, ScriptSandbox
from hardware_opt.llm_client import (
    DEFAULT_RESIDENT_MODEL,
    CancellationTokenSource,
    LocalInferenceOptimizer,
    LocalLlmClient,
    OperationCanceledError,
)
from hardware_opt.memory_pool import LiveReflexState, MemoryPoolManager
from rl_rewards.q_table import ContextStateHash, SkillQTable
from rl_rewards.reflexion_log import ReflexionMemoryLog
from rl_rewards.reward_calculator import RewardBreakdown, RewardCalculator
from rl_rewards.telemetry_snapshot import (
    BotTelemetrySnapshot,
    ExecutionTelemetryRecorder,
    Vector3,
    normalize_id,
)
from skills.code_agent import CodeSkillAgent
from skills.critic import CriticEvaluation, SelfVerificationCritic
from skills.skill_store import SkillRecord, SkillStatus, SkillStore

LOG = logging.getLogger("localai.orchestrator")

REFLEX_TICK_INTERVAL_SEC = 0.050  # Strict 20 Hz / 50ms tick


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode_index: int
    task_id: str
    skill_id: str
    used_fast_path: bool
    mutated_skill: bool
    critic: CriticEvaluation
    reward: RewardBreakdown
    updated_q: float
    preempted_by_reflex: bool = False


class WillowBotOrchestrator:
    """Coordinates the 20Hz Zero-Allocation Survival Reflex Loop and Background RL + Voyager Loop."""

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        resident_model: str = DEFAULT_RESIDENT_MODEL,
        ollama_url: str = "http://127.0.0.1:11434",
        llm_client: LocalLlmClient | None = None,
        observation_provider: Callable[[], dict[str, Any]] | None = None,
        reflex_executor: Callable[[str, str, LiveReflexState], None] | None = None,
        command_executor: ActionCommandExecutor | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.memory_pool = MemoryPoolManager()
        self.optimizer = LocalInferenceOptimizer(
            resident_model=resident_model,
            ollama_url=ollama_url,
        )
        self.llm_client = llm_client or LocalLlmClient(optimizer=self.optimizer)

        self.q_table = SkillQTable(storage_path=self.storage_dir / "rl_q_table.json")
        self.reflexion_log = ReflexionMemoryLog(jsonl_path=self.storage_dir / "reflexion_history.jsonl")
        self.dag = CurriculumDAG()
        self.curriculum = CurriculumManager(dag=self.dag, q_table=self.q_table)

        self.sandbox = ScriptSandbox(cache=self.memory_pool.skill_cache)
        self.skill_store = SkillStore(
            storage_dir=self.storage_dir / "v2_skills",
            q_table=self.q_table,
            sandbox=self.sandbox,
        )
        self.critic = SelfVerificationCritic(llm_client=self.llm_client)
        self.code_agent = CodeSkillAgent(
            llm_client=self.llm_client,
            skill_store=self.skill_store,
            reflexion_log=self.reflexion_log,
        )

        self._observation_provider = observation_provider
        self._reflex_executor = reflex_executor
        self._command_executor = command_executor

        self._lock = threading.Lock()
        self._master_shutdown_cts = CancellationTokenSource()
        self._episode_cts = CancellationTokenSource()
        self._reflex_thread: threading.Thread | None = None
        self._cognitive_thread: threading.Thread | None = None
        self._running = False
        self._episode_counter = 0

        # Cumulative discovery sets across episodes for intrinsic curiosity bonus
        self._discovered_items: set[str] = set()
        self._crafted_items: set[str] = set()
        self._prev_turn_snapshot: BotTelemetrySnapshot | None = None
        self._prev_turn_task: CurriculumNode | None = None
        self._prev_turn_skill_id: str = "actor_turn"

    @property
    def reflex_state(self) -> LiveReflexState:
        return self.memory_pool.reflex_state

    # ---- 20 Hz Fast Reactive Reflex Loop (0% GPU / 0 GC Allocations) ---------------------

    def step_reflex_tick(self, raw_observation: dict[str, Any] | None = None) -> tuple[bool, str, str]:
        """Executes one 50ms deterministic survival check in-place and preempts cognitive loop on danger."""
        obs = raw_observation
        if obs is None and self._observation_provider is not None:
            obs = self._observation_provider()
        if obs is None:
            return False, "NONE", ""

        rstate = self.memory_pool.reflex_state
        rstate.update_in_place(obs)
        triggered, hazard_code, reflex_action = rstate.evaluate_hazard()

        if triggered:
            was_active = rstate.combat_or_flee_active
            rstate.combat_or_flee_active = True
            rstate.active_reflex_reason = f"{hazard_code}:{reflex_action}"
            # Pause new LLM inference requests immediately while fighting/fleeing
            self.llm_client.pause_for_reflex()
            # Preempt running cognitive episode via master CancellationTokenSource
            with self._lock:
                if not self._episode_cts.is_cancellation_requested:
                    self._episode_cts.cancel()
            if not was_active:
                LOG.warning(
                    "[REFLEX-INTERRUPT] Preempted Slow Cognitive Loop! Hazard=%s Action=%s HP=%.1f Food=%d HostileDist=%.1fm",
                    hazard_code,
                    reflex_action,
                    rstate.health,
                    rstate.food_level,
                    rstate.nearest_hostile_distance,
                )
            if self._reflex_executor is not None:
                self._reflex_executor(hazard_code, reflex_action, rstate)
            return True, hazard_code, reflex_action

        if rstate.combat_or_flee_active:
            rstate.combat_or_flee_active = False
            rstate.active_reflex_reason = ""
            with self._lock:
                if self._episode_cts.is_cancellation_requested:
                    self._episode_cts = self._master_shutdown_cts.create_linked_source()
            self.llm_client.resume_after_reflex()
            LOG.info("[REFLEX-INTERRUPT] Survival hazard cleared; resumed Slow Cognitive Loop.")

        return False, "NONE", ""

    def _reflex_loop_worker(self) -> None:
        next_deadline = time.perf_counter()
        while self._running and not self._master_shutdown_cts.is_cancellation_requested:
            try:
                self.step_reflex_tick()
            except Exception as exc:
                LOG.error("[REFLEX-INTERRUPT] Reflex tick error: %s", exc)
            next_deadline += REFLEX_TICK_INTERVAL_SEC
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0:
                self._master_shutdown_cts._event.wait(timeout=sleep_sec)
            else:
                next_deadline = time.perf_counter()

    # ---- Slow Cognitive RL + Voyager Episode Execution -----------------------------------

    async def run_episode_async(
        self,
        initial_state: BotTelemetrySnapshot,
        *,
        bot_factory: Callable[[BotTelemetrySnapshot, CancellationTokenSource], WillowBotActions] | None = None,
    ) -> EpisodeSummary:
        with self._lock:
            self._episode_counter += 1
            ep_num = self._episode_counter
            if self._episode_cts.is_cancellation_requested and not self._master_shutdown_cts.is_cancellation_requested:
                self._episode_cts = self._master_shutdown_cts.create_linked_source()
            ep_cts = self._episode_cts.create_linked_source()

        ct = ep_cts.token
        self._discovered_items.update(initial_state.discovered_unique_items)
        self._crafted_items.update(initial_state.crafted_unique_items)

        s_prev = initial_state
        ctx_prev = ContextStateHash.encode(s_prev)
        task = self.curriculum.get_next_objective(s_prev)

        recorder = ExecutionTelemetryRecorder()
        if bot_factory is not None:
            bot = bot_factory(s_prev, ep_cts)
        else:
            bot = WillowBotActions(
                s_prev,
                ct=ct,
                pool=self.memory_pool,
                telemetry_recorder=recorder,
                command_executor=self._command_executor,
            )

        used_fast_path = False
        mutated_skill = False
        selected_record: SkillRecord | None = None
        skill_code = ""
        skill_id = ""
        skill_name = ""

        try:
            # 1. Check Tabular Q-Table Fast-Path (Q(c, n, k) >= +10.0 with satisfied preconditions -> 0 LLM calls!)
            verified_candidates = self.skill_store.get_verified_skills_for_task(task.id, s_prev)
            fast_skill_id, _fast_q = self.q_table.select_fast_path_skill(
                ctx_prev,
                task.id,
                [r.skill_id for r in verified_candidates],
            )
            if fast_skill_id is not None:
                selected_record = self.skill_store.get_skill(fast_skill_id)
                if selected_record is not None:
                    skill_code = self.skill_store.load_code(fast_skill_id)
                    skill_id = selected_record.skill_id
                    skill_name = selected_record.skill_name
                    used_fast_path = True

            # 2. Check if an existing task skill needs mutation (Q < -15.0 or >= 2 consecutive failures)
            if not skill_code:
                refinement_candidate = self._find_refinement_candidate(task.id)
                if refinement_candidate is not None:
                    trace = self.reflexion_log.format_for_prompt(task.id, max_lines=2)
                    mutated_rec = await self.code_agent.mutate_skill_async(
                        refinement_candidate,
                        task,
                        s_prev,
                        failure_trace=trace,
                        ct=ct,
                    )
                    selected_record = mutated_rec
                    skill_id = mutated_rec.skill_id
                    skill_name = mutated_rec.skill_name
                    skill_code = self.skill_store.load_code(skill_id)
                    mutated_skill = True
                else:
                    # 3. Synthesize new Voyager skill via single-flight 12GB-VRAM-bounded LLM
                    bundle = await self.code_agent.synthesize_skill_async(task, s_prev, ct=ct)
                    skill_id = bundle.skill_id
                    skill_name = bundle.skill_name
                    skill_code = bundle.code
                    if not skill_id.startswith("failed_"):
                        preconds = dict(getattr(task, "preconditions", {}) or {})
                        selected_record = self.skill_store.register_skill(
                            skill_id=skill_id,
                            skill_name=skill_name,
                            description=bundle.description,
                            code=skill_code,
                            task_id=task.id,
                            preconditions=preconds,
                            postconditions=dict(task.target_items),
                            verified=False,
                            q_value=0.0,
                        )
                    else:
                        selected_record = None

            # 4. Execute in AST-verified LRU-64 ScriptSandbox
            outcome: SandboxExecutionOutcome = await self.sandbox.execute_async(
                skill_code,
                bot,
                ct=ct,
                skill_name=skill_name,
            )

        except OperationCanceledError:
            LOG.warning("[REFLEX-INTERRUPT] Episode #%d (%s) aborted cleanly due to survival reflex preemption.", ep_num, task.id)
            s_interrupted = bot.GetState()
            neutral_critic = CriticEvaluation(
                success=False,
                failure_category=self.critic_category_interrupted(),
                cause="Preempted by 20Hz Survival Reflex Loop",
                suggested_fix="Resume objective after immediate hazard is cleared.",
            )
            zero_reward = RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            _ = s_interrupted
            return EpisodeSummary(
                episode_index=ep_num,
                task_id=task.id,
                skill_id=skill_id or "preempted",
                used_fast_path=used_fast_path,
                mutated_skill=mutated_skill,
                critic=neutral_critic,
                reward=zero_reward,
                updated_q=0.0,
                preempted_by_reflex=True,
            )

        # 5. Capture S_{t+1} and run Deterministic-First SelfVerificationCritic
        s_curr = bot.GetState()
        self._discovered_items.update(s_curr.discovered_unique_items)
        self._crafted_items.update(s_curr.crafted_unique_items)
        self.memory_pool.record_snapshot(s_curr.world_tick, s_curr)

        critic_eval = await self.critic.evaluate_async(
            s_prev,
            s_curr,
            task,
            outcome.action_result,
            ct=ct,
        )

        # 6. Compute exact mathematical RL reward R_total & update Tabular Q-Table
        reward = RewardCalculator.calculate_episode_reward(s_prev, s_curr, task, critic_eval)
        ctx_next = ContextStateHash.encode(s_curr)
        next_task_peek = self.curriculum.get_next_objective(s_curr)
        updated_q = self.q_table.update(
            context_prev=ctx_prev,
            task_prev=task.id,
            skill_id=skill_id,
            reward_total=reward.r_total,
            context_next=ctx_next,
            task_next=next_task_peek.id,
        )

        # 7. Update SkillStore status (promote Verified or mark NeedsRefinement) & Reflexion Ring Buffer
        if selected_record is not None:
            self.skill_store.record_execution_feedback(
                skill_id,
                success=critic_eval.success,
                updated_q=updated_q,
            )
            if critic_eval.success:
                self.skill_store._bind_into_sandbox(selected_record)

        self.reflexion_log.record_if_needed(
            task_id=task.id,
            reward=reward,
            critic_success=critic_eval.success,
            cause=critic_eval.cause,
            fix=critic_eval.suggested_fix,
        )

        # 8. Update CurriculumManager completion & automatic sub-task decomposition (>= 3 failures)
        self.curriculum.record_outcome(
            task,
            critic_eval.success,
            s_curr,
            reflexion_lines=self.reflexion_log.get_prompt_lines(task.id),
            llm_client=self.llm_client,
            ct=ct,
        )

        return EpisodeSummary(
            episode_index=ep_num,
            task_id=task.id,
            skill_id=skill_id,
            used_fast_path=used_fast_path,
            mutated_skill=mutated_skill,
            critic=critic_eval,
            reward=reward,
            updated_q=updated_q,
            preempted_by_reflex=False,
        )

    def run_episode_sync(
        self,
        initial_state: BotTelemetrySnapshot,
        *,
        bot_factory: Callable[[BotTelemetrySnapshot, CancellationTokenSource], WillowBotActions] | None = None,
    ) -> EpisodeSummary:
        return asyncio.run(self.run_episode_async(initial_state, bot_factory=bot_factory))

    def observe_turn_transition(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        """Lightweight 0-LLM per-turn hook called by server.py's HTTP /decide loop."""
        with self._lock:
            self.memory_pool.reflex_state.update_in_place(raw_observation)
            last_res = raw_observation.get("last_result") if isinstance(raw_observation.get("last_result"), dict) else {}
            status = str(last_res.get("status") or "")

            if status == "success" and last_res.get("action") == "craft" and last_res.get("target"):
                self._crafted_items.add(normalize_id(str(last_res["target"])))

            snap = BotTelemetrySnapshot.from_observation(
                raw_observation,
                discovered_items=self._discovered_items,
                crafted_items=self._crafted_items,
            )
            self._discovered_items.update(snap.discovered_unique_items)
            self._crafted_items.update(snap.crafted_unique_items)

            last_reward: float | None = None
            if self._prev_turn_snapshot is not None and self._prev_turn_task is not None and status in ("success", "failed"):
                action_ok = status == "success"
                act_res = (
                    ActionResult.ok(str(last_res.get("detail") or ""))
                    if action_ok
                    else ActionResult.fail("MISSING_ITEM", str(last_res.get("detail") or "failed"))
                )
                # Purely deterministic critic evaluation (llm_client=None -> 0ms overhead on live turn)
                det_critic = SelfVerificationCritic(llm_client=None).evaluate_sync(
                    self._prev_turn_snapshot,
                    snap,
                    self._prev_turn_task,
                    act_res,
                )
                breakdown = RewardCalculator.calculate_episode_reward(
                    self._prev_turn_snapshot,
                    snap,
                    self._prev_turn_task,
                    det_critic,
                )
                last_reward = breakdown.r_total
                c_prev = ContextStateHash.encode(self._prev_turn_snapshot)
                c_curr = ContextStateHash.encode(snap)
                next_obj = self.curriculum.get_next_objective(snap)
                self.q_table.update(
                    context_prev=c_prev,
                    task_prev=self._prev_turn_task.id,
                    skill_id=self._prev_turn_skill_id,
                    reward_total=breakdown.r_total,
                    context_next=c_curr,
                    task_next=next_obj.id,
                )
                self.reflexion_log.record_if_needed(
                    self._prev_turn_task.id,
                    breakdown,
                    det_critic.success,
                    det_critic.cause,
                    det_critic.suggested_fix,
                )
                self.curriculum.record_outcome(self._prev_turn_task, det_critic.success, snap)
            else:
                next_obj = self.curriculum.get_next_objective(snap)

            self._prev_turn_snapshot = snap
            self._prev_turn_task = next_obj
            self._prev_turn_skill_id = f"turn_{next_obj.id}"

            return {
                "active_node_id": next_obj.id,
                "tier": next_obj.tier,
                "objective": next_obj.description,
                "postcondition": next_obj.verification_expr,
                "completed_count": len(self.curriculum.completed_nodes),
                "context_hash": ContextStateHash.encode(snap),
                "last_reward": round(last_reward, 2) if last_reward is not None else None,
            }

    def start(self) -> None:
        """Launches both the 20Hz Reflex Loop thread and Low-Priority Slow Cognitive Loop worker."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._master_shutdown_cts = CancellationTokenSource()
            self._episode_cts = self._master_shutdown_cts.create_linked_source()

        self._reflex_thread = threading.Thread(
            target=self._reflex_loop_worker,
            name="Willow-20Hz-ReflexLoop",
            daemon=True,
        )
        self._cognitive_thread = threading.Thread(
            target=self._cognitive_loop_worker,
            name="Willow-BelowNormal-CognitiveLoop",
            daemon=True,
        )
        self._reflex_thread.start()
        self._cognitive_thread.start()
        LOG.info("[VRAM-OPT] Started WillowBotOrchestrator Dual-Loop Runtime (20Hz Reflex + BelowNormal Cognitive Worker).")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                self.reflexion_log.close()
                return
            self._running = False
            self._master_shutdown_cts.cancel()

        if self._reflex_thread is not None:
            self._reflex_thread.join(timeout=2.0)
        if self._cognitive_thread is not None:
            self._cognitive_thread.join(timeout=2.0)
        self.reflexion_log.close()
        self.q_table.save()

    def _cognitive_loop_worker(self) -> None:
        # Best-effort ThreadPriority.BelowNormal equivalent on POSIX without requiring root privileges
        try:
            if hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_BATCH"):
                pass
            os.nice(5)
        except OSError:
            pass

        while self._running and not self._master_shutdown_cts.is_cancellation_requested:
            if self._observation_provider is None:
                self._master_shutdown_cts._event.wait(timeout=0.1)
                continue
            if self.llm_client.is_paused_for_reflex:
                self._master_shutdown_cts._event.wait(timeout=0.05)
                continue
            try:
                raw_obs = self._observation_provider()
                snap = BotTelemetrySnapshot.from_observation(
                    raw_obs,
                    discovered_items=self._discovered_items,
                    crafted_items=self._crafted_items,
                )
                self.run_episode_sync(snap)
            except OperationCanceledError:
                continue
            except Exception as exc:
                LOG.error("Slow cognitive loop episode error: %s", exc)
                self._master_shutdown_cts._event.wait(timeout=0.25)

    def _find_refinement_candidate(self, task_id: str) -> SkillRecord | None:
        with self.skill_store._lock:
            for rec in self.skill_store._records.values():
                if rec.task_id == task_id and rec.status == SkillStatus.NeedsRefinement:
                    return rec
        return None

    @staticmethod
    def critic_category_interrupted():
        from skills.critic import FailureCategory

        return FailureCategory.PATHFINDING_STUCK
