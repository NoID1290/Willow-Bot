"""Module 4: Deterministic-First SelfVerificationCritic (0ms GPU Fast Path + 150-Token LLM Fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

from curriculum.curriculum_dag import CurriculumNode
from execution.bot_actions import ActionResult
from hardware_opt.llm_client import CancellationToken, InferencePriority, LocalLlmClient
from hardware_opt.telemetry_formatter import CompactTelemetryFormatter
from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot, TelemetryStateDiff

LOG = logging.getLogger("localai.critic")

FALLBACK_CRITIC_STATIC_PREFIX = (
    "You are Willow-Bot's Diagnostic Critic. Analyze in <=150 tokens why the skill finished cleanly "
    "without satisfying the postcondition. "
    'Output strictly valid JSON: {"cause": "max 15 words", "fix": "max 15 words"}'
)


class FailureCategory(str, Enum):
    NONE = "NONE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    API_MISUSE = "API_MISUSE"
    WRONG_HARVEST_TOOL = "WRONG_HARVEST_TOOL"
    BOT_DIED = "BOT_DIED"
    PATHFINDING_STUCK = "PATHFINDING_STUCK"
    MISSING_PREREQUISITE = "MISSING_PREREQUISITE"
    AMBIGUOUS_FAILURE = "AMBIGUOUS_FAILURE"


@dataclass(frozen=True, slots=True)
class CriticEvaluation:
    success: bool
    failure_category: FailureCategory
    cause: str
    suggested_fix: str
    used_llm_fallback: bool = False

    @property
    def Success(self) -> bool:
        return self.success

    @property
    def summary(self) -> str:
        return f"{self.failure_category.value}: {self.cause}"


class SelfVerificationCritic:
    """Executes 100% deterministic verification first; invokes resident LLM only on ambiguous failures."""

    def __init__(self, llm_client: LocalLlmClient | None = None) -> None:
        self.llm_client = llm_client

    def evaluate_sync(
        self,
        s_prev: BotTelemetrySnapshot,
        s_curr: BotTelemetrySnapshot,
        task: CurriculumNode,
        action_result: ActionResult | None = None,
        *,
        ct: CancellationToken | None = None,
    ) -> CriticEvaluation:
        tel = s_curr.execution_telemetry

        # 1. Native deterministic postcondition check
        if not tel.compilation_failed and s_curr.vitals.is_alive and task.verify(s_prev, s_curr):
            verdict = CriticEvaluation(
                success=True,
                failure_category=FailureCategory.NONE,
                cause="Postcondition verified deterministically.",
                suggested_fix="",
                used_llm_fallback=False,
            )
            LOG.info("[CRITIC] PASS task=%s category=NONE (0ms GPU)", task.id)
            return verdict

        # 2. Deterministic failure classifiers (0 LLM calls)
        if tel.compilation_failed:
            msg = action_result.Message if action_result is not None else "Static AST or syntax error"
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.SYNTAX_ERROR,
                cause=f"Compilation/AST rejected: {msg[:90]}",
                suggested_fix="Use only async def(bot, ct), await bot.* methods, no imports or blocking loops.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        if tel.runtime_exceptions:
            exc_text = tel.runtime_exceptions[0]
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.API_MISUSE,
                cause=f"Unhandled runtime exception: {exc_text[:90]}",
                suggested_fix="Call only valid IWillowBotActions methods and check ActionResult.Success.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        if tel.improper_tool_mine_attempts > 0 or (
            action_result is not None and action_result.ErrorCode == "NO_TOOL"
        ):
            detail = action_result.Message if action_result is not None else "Mined block with insufficient pickaxe tier"
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.WRONG_HARVEST_TOOL,
                cause=detail[:110],
                suggested_fix="Craft/equip required pickaxe tier before calling MineBlocksAsync.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        if not s_curr.vitals.is_alive or s_curr.vitals.health <= 0.0:
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.BOT_DIED,
                cause="Bot died during episode execution.",
                suggested_fix="Eat when hungry, equip armor/shield, and retreat from close hostiles or lava.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        if tel.pathfinding_timeouts > 0 or (
            action_result is not None and action_result.ErrorCode in ("PATH_BLOCKED", "TIMEOUT")
        ):
            detail = action_result.Message if action_result is not None else "Navigation or search timed out"
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.PATHFINDING_STUCK,
                cause=detail[:110],
                suggested_fix="ExploreAsync toward open terrain or DigStaircaseToYAsync safely.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        # Deterministic missing target/prerequisite diagnosis
        missing_targets: list[str] = []
        for pat, req in task.target_items.items():
            have = s_curr.count_item(pat)
            if have < req:
                missing_targets.append(f"{pat}({have}/{req})")

        if not task.check_preconditions(s_curr) or (
            action_result is not None and not action_result.Success and action_result.ErrorCode == "MISSING_ITEM"
        ):
            cause_msg = (
                action_result.Message
                if (action_result is not None and not action_result.Success)
                else f"Preconditions unmet ({task.precondition_expr}); deficits: {', '.join(missing_targets)}"
            )
            verdict = CriticEvaluation(
                success=False,
                failure_category=FailureCategory.MISSING_PREREQUISITE,
                cause=cause_msg[:110],
                suggested_fix="Ensure workstation is placed and gather/craft missing prerequisite ingredients first.",
            )
            LOG.info("[CRITIC] FAIL task=%s category=%s (0ms GPU)", task.id, verdict.failure_category.value)
            return verdict

        # 3. Step 2 — Fast 150-token LLM Fallback Critic ONLY when script ran cleanly yet postcondition failed
        if self.llm_client is not None:
            diff = TelemetryStateDiff.compute(s_prev, s_curr)
            compact_curr = CompactTelemetryFormatter.format(s_curr, target_items=task.target_items)
            dynamic_diag = (
                f"[TASK] {task.id} | Post: {task.verification_expr}\n"
                f"[DELTA_INV] {dict(diff.inventory_deltas)}\n"
                f"[STATE_AFTER] {compact_curr}"
            )
            try:
                diag = self.llm_client.generate_sync(
                    static_prefix=FALLBACK_CRITIC_STATIC_PREFIX,
                    dynamic_suffix=dynamic_diag,
                    priority=InferencePriority.FALLBACK_CRITIC,
                    num_predict=150,
                    ct=ct,
                )
                cause = str(diag.get("cause") or "Clean run did not satisfy postcondition")[:110]
                fix = str(diag.get("fix") or "Loop until target item count satisfies postcondition")[:110]
                LOG.info("[CRITIC] FAIL task=%s category=AMBIGUOUS_FAILURE (LLM 150-tok fallback)", task.id)
                return CriticEvaluation(
                    success=False,
                    failure_category=FailureCategory.AMBIGUOUS_FAILURE,
                    cause=cause,
                    suggested_fix=fix,
                    used_llm_fallback=True,
                )
            except Exception as exc:
                LOG.debug("[CRITIC] LLM fallback diagnostic skipped: %s", exc)

        fallback_cause = (
            f"Target counts incomplete: {', '.join(missing_targets)}"
            if missing_targets
            else f"Postcondition '{task.verification_expr}' remained false after clean run."
        )
        return CriticEvaluation(
            success=False,
            failure_category=FailureCategory.AMBIGUOUS_FAILURE,
            cause=fallback_cause[:110],
            suggested_fix="Verify count thresholds after crafting/mining before returning ok.",
            used_llm_fallback=False,
        )

    async def evaluate_async(
        self,
        s_prev: BotTelemetrySnapshot,
        s_curr: BotTelemetrySnapshot,
        task: CurriculumNode,
        action_result: ActionResult | None = None,
        *,
        ct: CancellationToken | None = None,
    ) -> CriticEvaluation:
        import asyncio

        return await asyncio.to_thread(
            self.evaluate_sync,
            s_prev,
            s_curr,
            task,
            action_result,
            ct=ct,
        )
