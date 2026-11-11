"""Module 4: KV-Cache Optimized Voyager Code Synthesizer & Skill Mutator (CodeSkillAgent)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import re
from typing import Any

from curriculum.curriculum_dag import CurriculumNode
from curriculum.mechanics_registry import SurvivalMechanicsRegistry
from execution.sandbox import SandboxSecurityError, validate_skill_ast
from hardware_opt.llm_client import (
    MAX_DYNAMIC_SUFFIX_TOKENS,
    CancellationToken,
    InferencePriority,
    LocalInferenceOptimizer,
    LocalLlmClient,
    estimate_tokens,
)
from hardware_opt.telemetry_formatter import CompactTelemetryFormatter
from rl_rewards.reflexion_log import ReflexionMemoryLog
from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot

from .skill_store import SkillRecord, SkillStore

LOG = logging.getLogger("localai.voyager_synth")

# 100% IDENTICAL STATIC SYSTEM PREFIX ACROSS ALL CALLS FOR OLLAMA / LLAMA.CPP KV-CACHE REUSE
STATIC_SYSTEM_PREFIX = """You are Willow-Bot's Voyager Code Synthesizer. Write a concise, fault-tolerant async Python control function `async def <skill_name>(bot, ct):` using ONLY `IWillowBotActions bot` and `CancellationToken ct`.
API SIGNATURES:
- bot.GetState() -> BotTelemetrySnapshot
- bot.HasItem(pattern, minCount=1) -> bool | bot.GetItemCount(pattern) -> int
- bot.FindBlocks(blockPattern, searchRadius=32, maxResults=16) -> list[BlockInfo]
- bot.FindEntities(entityPattern, searchRadius=24, onlyAlive=True) -> list[PerceivedEntity]
- bot.FindSafeSurfaceOrWater(searchRadius=32) -> Vector3 | None
- await bot.MoveToAsync(targetPos, arrivalRadius=2.0, timeoutMs=20000) -> ActionResult
- await bot.ExploreAsync(preferredDirection, distance=32, timeoutMs=25000) -> ActionResult
- await bot.DigStaircaseToYAsync(targetY, placeTorches=True, ct=ct) -> ActionResult
- await bot.MineBlocksAsync(blockPattern, targetCount, maxSearchRadius=32) -> ActionResult
- await bot.MineBlockAtAsync(blockPos, collectDrops=True) -> ActionResult
- await bot.PlaceBlockAsync(itemId, placeAtPos) -> ActionResult
- await bot.UseBucketAtAsync(bucketType, targetBlockPos) -> ActionResult
- await bot.EnsureWorkstationPlacedAsync(stationItemId) -> ActionResult
- await bot.ReclaimWorkstationAsync(stationBlockId) -> ActionResult
- await bot.CraftAsync(outputItemId, craftOperationsCount=1) -> ActionResult
- await bot.SmeltAsync(inputItemId, fuelItemId, count=1) -> ActionResult
- await bot.EquipBestToolForBlockAsync(blockId) -> ActionResult
- await bot.EquipArmorAndWeaponsAsync() -> ActionResult
- await bot.EatIfHungryAsync(hungerThreshold=15) -> ActionResult
- await bot.EngageEntityAsync(entityId) -> ActionResult
- await bot.BuildSimpleShelterAsync(blockMaterial="cobblestone") -> ActionResult
- await bot.WaitTicksAsync(ticks) -> None | bot.Log(message) -> None
RULES:
1. Check `if not res.Success: return res` on critical steps; return `ActionResult.ok("done")` on completion.
2. Never mine stone/ores without checking/crafting required pickaxe tier first (-8.0 penalty).
3. No imports, no sleep, and every loop MUST contain an `await` or `ct.ThrowIfCancellationRequested()`.
4. Output strictly valid JSON: {"reasoning": "max 2 short sentences", "skill_name": "string", "description": "string", "code": "string"}"""

SYNTHESIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "skill_name": {"type": "string"},
        "description": {"type": "string"},
        "code": {"type": "string"},
    },
    "required": ["reasoning", "skill_name", "description", "code"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class SynthesizedSkillBundle:
    skill_id: str
    skill_name: str
    description: str
    reasoning: str
    code: str
    static_prefix_tokens: int
    dynamic_suffix_tokens: int


def _normalize_identifier(raw_name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", (raw_name or "").strip().lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"skill_{cleaned}" if cleaned else fallback
    return cleaned[:48]


def _clamp_two_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return " ".join(sentences[:2])[:220]


class CodeSkillAgent:
    """Orchestrates KV-cache aligned prompt assembly, AST-verified synthesis retries, and skill mutation."""

    def __init__(
        self,
        llm_client: LocalLlmClient,
        skill_store: SkillStore,
        reflexion_log: ReflexionMemoryLog,
        *,
        max_syntax_retries: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.skill_store = skill_store
        self.reflexion_log = reflexion_log
        self.max_syntax_retries = max(1, int(max_syntax_retries))

    def build_dynamic_suffix(
        self,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        *,
        syntax_feedback: str = "",
        mutating_skill: SkillRecord | None = None,
        mutating_source: str = "",
        failure_trace: str = "",
    ) -> str:
        rules_line = SurvivalMechanicsRegistry.filter_rules_for_task(task)
        compact_state = CompactTelemetryFormatter.format(state, target_items=task.target_items)
        rl_history = self.reflexion_log.format_for_prompt(task.id, max_lines=3)
        rag_sigs = self.skill_store.format_rag_signatures(task, state, top_k=3)

        lines = [
            f"[TASK] {task.id}: {task.description} | Postcondition: {task.verification_description} (Reward: +{task.base_reward:.1f})",
            f"[RULES] {rules_line}",
            f"[STATE] {compact_state}",
            f"[RL_HISTORY] {rl_history}",
            f"[RAG_SKILLS] {rag_sigs}",
        ]
        if mutating_skill is not None and mutating_source:
            lines.append(
                f"[MUTATE_SKILL] {mutating_skill.skill_id} (Q={mutating_skill.q_value:+.1f}, Fails={mutating_skill.consecutive_failures}) "
                f"Trace: {failure_trace[:160]}\n{mutating_source[:900]}"
            )
        if syntax_feedback:
            lines.append(f"[SYNTAX_FIX_REQUIRED] {syntax_feedback[:180]}")

        suffix = "\n".join(lines)
        return LocalInferenceOptimizer.clamp_dynamic_suffix(suffix, max_tokens=MAX_DYNAMIC_SUFFIX_TOKENS)

    def synthesize_skill_sync(
        self,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        *,
        ct: CancellationToken | None = None,
    ) -> SynthesizedSkillBundle:
        syntax_error = ""
        for attempt in range(self.max_syntax_retries):
            dynamic_suffix = self.build_dynamic_suffix(task, state, syntax_feedback=syntax_error)
            temp = 0.15 if attempt == 0 else 0.25
            resp = self.llm_client.generate_sync(
                static_prefix=STATIC_SYSTEM_PREFIX,
                dynamic_suffix=dynamic_suffix,
                priority=InferencePriority.NEW_SKILL_SYNTHESIS,
                temperature=temp,
                format_schema=SYNTHESIS_JSON_SCHEMA,
                ct=ct,
            )
            skill_name = _normalize_identifier(str(resp.get("skill_name") or ""), f"solve_{task.id}")
            desc = str(resp.get("description") or task.description)[:140]
            reasoning = _clamp_two_sentences(str(resp.get("reasoning") or ""))
            code = str(resp.get("code") or "")

            try:
                validate_skill_ast(code)
                prefix_toks = estimate_tokens(STATIC_SYSTEM_PREFIX)
                suffix_toks = estimate_tokens(dynamic_suffix)
                LOG.info(
                    "[VOYAGER-SYNTH] Synthesized '%s' for task='%s' (attempt=%d, prefix_tok=%d, suffix_tok=%d)",
                    skill_name,
                    task.id,
                    attempt + 1,
                    prefix_toks,
                    suffix_toks,
                )
                return SynthesizedSkillBundle(
                    skill_id=skill_name,
                    skill_name=skill_name,
                    description=desc,
                    reasoning=reasoning,
                    code=code,
                    static_prefix_tokens=prefix_toks,
                    dynamic_suffix_tokens=suffix_toks,
                )
            except SandboxSecurityError as exc:
                syntax_error = str(exc)
                LOG.warning(
                    "[VOYAGER-SYNTH] Attempt %d AST/Syntax check failed for '%s': %s",
                    attempt + 1,
                    task.id,
                    syntax_error,
                )

        # Return last payload so ScriptSandbox & RewardCalculator apply deterministic -15.0 compilation penalty
        return SynthesizedSkillBundle(
            skill_id=f"failed_{task.id}",
            skill_name=f"failed_{task.id}",
            description=task.description,
            reasoning="Syntax retry budget exhausted.",
            code=code,
            static_prefix_tokens=estimate_tokens(STATIC_SYSTEM_PREFIX),
            dynamic_suffix_tokens=estimate_tokens(dynamic_suffix),
        )

    async def synthesize_skill_async(
        self,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        *,
        ct: CancellationToken | None = None,
    ) -> SynthesizedSkillBundle:
        return await asyncio.to_thread(self.synthesize_skill_sync, task, state, ct=ct)

    def mutate_skill_sync(
        self,
        skill_record: SkillRecord,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        failure_trace: str,
        *,
        ct: CancellationToken | None = None,
    ) -> SkillRecord:
        old_code = self.skill_store.load_code(skill_record.skill_id)
        syntax_err = ""
        last_code = old_code
        new_name = f"{skill_record.skill_name}_v{skill_record.version + 1}"
        new_desc = skill_record.description

        for attempt in range(self.max_syntax_retries):
            dynamic_suffix = self.build_dynamic_suffix(
                task,
                state,
                syntax_feedback=syntax_err,
                mutating_skill=skill_record,
                mutating_source=old_code,
                failure_trace=failure_trace,
            )
            resp = self.llm_client.generate_sync(
                static_prefix=STATIC_SYSTEM_PREFIX,
                dynamic_suffix=dynamic_suffix,
                priority=InferencePriority.SKILL_MUTATION,
                temperature=0.2,
                format_schema=SYNTHESIS_JSON_SCHEMA,
                ct=ct,
            )
            cand_code = str(resp.get("code") or "")
            cand_name = _normalize_identifier(str(resp.get("skill_name") or ""), new_name)
            cand_desc = str(resp.get("description") or new_desc)[:140]
            try:
                validate_skill_ast(cand_code)
                last_code = cand_code
                new_name = cand_name
                new_desc = cand_desc
                syntax_err = ""
                break
            except SandboxSecurityError as exc:
                syntax_err = str(exc)

        if syntax_err or last_code == old_code:
            LOG.warning(
                "[VOYAGER-SYNTH] Mutation syntax retries exhausted for '%s': %s; retaining existing skill without overwriting.",
                skill_record.skill_id,
                syntax_err,
            )
            return skill_record

        replaced = self.skill_store.replace_with_mutated_skill(
            skill_record.skill_id,
            new_skill_name=new_name,
            new_description=new_desc,
            new_code=last_code,
        )
        LOG.info(
            "[VOYAGER-SYNTH] Mutated skill '%s' -> '%s' (version=%d)",
            skill_record.skill_id,
            replaced.skill_id,
            replaced.version,
        )
        return replaced

    async def mutate_skill_async(
        self,
        skill_record: SkillRecord,
        task: CurriculumNode,
        state: BotTelemetrySnapshot,
        failure_trace: str,
        *,
        ct: CancellationToken | None = None,
    ) -> SkillRecord:
        return await asyncio.to_thread(
            self.mutate_skill_sync,
            skill_record,
            task,
            state,
            failure_trace,
            ct=ct,
        )

    # Alias matching C# specification
    MutateSkillAsync = mutate_skill_async
