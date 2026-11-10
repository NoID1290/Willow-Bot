"""Module 2: Native CurriculumManager with Dynamic Regression Recovery & Sub-Task Decomposer."""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from hardware_opt.llm_client import CancellationToken, InferencePriority, LocalLlmClient
from hardware_opt.telemetry_formatter import CompactTelemetryFormatter
from rl_rewards.q_table import ContextStateHash, SkillQTable
from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot, matches_item_pattern

from .curriculum_dag import CurriculumDAG, CurriculumNode
from .mechanics_registry import RecipeKnowledgeGraph

LOG = logging.getLogger("localai.curriculum")

DECOMPOSER_STATIC_PREFIX = (
    "You are Willow-Bot's Curriculum Decomposer. A survival task failed >=3 times. "
    "Split it into 2 or 3 sequential micro-tasks with concrete item postconditions. "
    'Output strictly valid JSON: {"subtasks": [{"id": "string", "description": "string", '
    '"target_item": "string", "target_count": 1}]}'
)


class CurriculumDecomposer:
    """Splits a repeatedly failing CurriculumNode into 2-3 simpler micro-tasks."""

    @classmethod
    def split_node_sync(
        cls,
        node: CurriculumNode,
        state: BotTelemetrySnapshot,
        reflexion_history: list[str],
        llm_client: LocalLlmClient | None = None,
        ct: CancellationToken | None = None,
    ) -> list[CurriculumNode]:
        if llm_client is not None:
            compact_state = CompactTelemetryFormatter.format(state, target_items=node.target_items)
            hist = " ; ".join(reflexion_history[-3:]) if reflexion_history else "none"
            dynamic_suffix = (
                f"[FAILED_NODE] {node.id} (Tier {node.tier}, Fails={node.consecutive_failures}): "
                f"{node.description} | Post: {node.verification_expr}\n"
                f"[STATE] {compact_state}\n"
                f"[RL_HISTORY] {hist}"
            )
            try:
                resp = llm_client.generate_sync(
                    static_prefix=DECOMPOSER_STATIC_PREFIX,
                    dynamic_suffix=dynamic_suffix,
                    priority=InferencePriority.TASK_DECOMPOSITION,
                    num_predict=384,
                    ct=ct,
                )
                raw_subs = resp.get("subtasks")
                if isinstance(raw_subs, list) and 2 <= len(raw_subs) <= 3:
                    built = cls._build_micro_nodes_from_llm(node, raw_subs)
                    if built:
                        LOG.info(
                            "[CURRICULUM] LLM decomposed '%s' into %d micro-tasks: %s",
                            node.id,
                            len(built),
                            [m.id for m in built],
                        )
                        return built
            except Exception as exc:
                LOG.warning("[CURRICULUM] LLM decomposition fallback to deterministic recipe split: %s", exc)

        fallback = cls._deterministic_split(node, state)
        LOG.info(
            "[CURRICULUM] Deterministically decomposed '%s' into %d micro-tasks: %s",
            node.id,
            len(fallback),
            [m.id for m in fallback],
        )
        return fallback

    @classmethod
    async def split_node_async(
        cls,
        node: CurriculumNode,
        state: BotTelemetrySnapshot,
        reflexion_history: list[str],
        llm_client: LocalLlmClient | None = None,
        ct: CancellationToken | None = None,
    ) -> list[CurriculumNode]:
        import asyncio

        return await asyncio.to_thread(
            cls.split_node_sync,
            node,
            state,
            reflexion_history,
            llm_client,
            ct,
        )

    # Alias matching specification
    SplitNodeAsync = split_node_async

    @staticmethod
    def _build_micro_nodes_from_llm(
        parent: CurriculumNode,
        raw_subs: list[dict[str, Any]],
    ) -> list[CurriculumNode]:
        micro_nodes: list[CurriculumNode] = []
        share_reward = round(max(15.0, parent.base_reward / max(2, len(raw_subs))), 1)
        for idx, sub in enumerate(raw_subs[:3]):
            if not isinstance(sub, dict):
                continue
            target_item = str(sub.get("target_item") or "").strip()
            target_cnt = max(1, int(sub.get("target_count") or 1))
            if not target_item:
                continue
            sub_id = f"{parent.id}_sub{idx + 1}"
            desc = str(sub.get("description") or f"Acquire {target_cnt}x {target_item}")[:140]
            targets = {target_item: target_cnt}

            def _make_post(pat: str, req: int):
                return lambda s: s.count_item(pat) >= req

            micro_nodes.append(
                CurriculumNode(
                    id=sub_id,
                    tier=parent.tier,
                    description=desc,
                    advancement_id=None,
                    prerequisites=parent.prerequisites if idx == 0 else (f"{parent.id}_sub{idx}",),
                    precondition_expr="None",
                    verification_expr=f"Count('{target_item}') >= {target_cnt}",
                    base_reward=share_reward,
                    target_items=targets,
                    requires_movement=parent.requires_movement,
                    persistent_milestone=False,
                    is_micro_task=True,
                    parent_node_id=parent.id,
                    _precondition_fn=lambda _s: True,
                    _postcondition_fn=_make_post(target_item, target_cnt),
                )
            )
        return micro_nodes if len(micro_nodes) >= 2 else []

    @staticmethod
    def _deterministic_split(parent: CurriculumNode, state: BotTelemetrySnapshot) -> list[CurriculumNode]:
        items = list(parent.target_items.items())
        micro_nodes: list[CurriculumNode] = []
        if len(items) >= 2:
            share = round(max(15.0, parent.base_reward / min(3, len(items))), 1)
            for idx, (pat, cnt) in enumerate(items[:3]):
                sub_id = f"{parent.id}_sub{idx + 1}"

                def _make_post(p: str, c: int):
                    return lambda s: s.count_item(p) >= c

                micro_nodes.append(
                    CurriculumNode(
                        id=sub_id,
                        tier=parent.tier,
                        description=f"Step {idx + 1} for {parent.id}: obtain {cnt}x {pat}",
                        advancement_id=None,
                        prerequisites=parent.prerequisites if idx == 0 else (f"{parent.id}_sub{idx}",),
                        precondition_expr="None",
                        verification_expr=f"Count('{pat}') >= {cnt}",
                        base_reward=share,
                        target_items={pat: cnt},
                        requires_movement=parent.requires_movement,
                        is_micro_task=True,
                        parent_node_id=parent.id,
                        _precondition_fn=lambda _s: True,
                        _postcondition_fn=_make_post(pat, cnt),
                    )
                )
            return micro_nodes

        # Single-item or advancement node: split into prerequisite ingredient gather + final completion
        if items:
            pat, cnt = items[0]
            half = max(1, cnt // 2)
            rec = RecipeKnowledgeGraph.get_recipe(pat)
            if rec is not None and rec.ingredients:
                ing_pat, ing_cnt = next(iter(rec.ingredients.items()))
                need_ing = max(1, ing_cnt * cnt)
                s1_id = f"{parent.id}_sub1"
                s2_id = f"{parent.id}_sub2"
                return [
                    CurriculumNode(
                        id=s1_id,
                        tier=parent.tier,
                        description=f"Gather {need_ing}x {ing_pat} required for {pat}",
                        advancement_id=None,
                        prerequisites=parent.prerequisites,
                        precondition_expr="None",
                        verification_expr=f"Count('{ing_pat}') >= {need_ing}",
                        base_reward=round(parent.base_reward * 0.4, 1),
                        target_items={ing_pat: need_ing},
                        requires_movement=True,
                        is_micro_task=True,
                        parent_node_id=parent.id,
                        _precondition_fn=lambda _s: True,
                        _postcondition_fn=lambda s, p=ing_pat, c=need_ing: s.count_item(p) >= c,
                    ),
                    CurriculumNode(
                        id=s2_id,
                        tier=parent.tier,
                        description=parent.description,
                        advancement_id=parent.advancement_id,
                        prerequisites=(s1_id,),
                        precondition_expr=parent.precondition_expr,
                        verification_expr=parent.verification_expr,
                        base_reward=round(parent.base_reward * 0.6, 1),
                        target_items=dict(parent.target_items),
                        requires_movement=parent.requires_movement,
                        is_micro_task=True,
                        parent_node_id=parent.id,
                        _precondition_fn=parent._precondition_fn,
                        _postcondition_fn=parent._postcondition_fn,
                    ),
                ]
            if cnt >= 2:
                s1_id = f"{parent.id}_sub1"
                s2_id = f"{parent.id}_sub2"
                return [
                    CurriculumNode(
                        id=s1_id,
                        tier=parent.tier,
                        description=f"Partial harvest: gather {half}x {pat}",
                        advancement_id=None,
                        prerequisites=parent.prerequisites,
                        precondition_expr=parent.precondition_expr,
                        verification_expr=f"Count('{pat}') >= {half}",
                        base_reward=round(parent.base_reward * 0.45, 1),
                        target_items={pat: half},
                        requires_movement=parent.requires_movement,
                        is_micro_task=True,
                        parent_node_id=parent.id,
                        _precondition_fn=parent._precondition_fn,
                        _postcondition_fn=lambda s, p=pat, c=half: s.count_item(p) >= c,
                    ),
                    CurriculumNode(
                        id=s2_id,
                        tier=parent.tier,
                        description=parent.description,
                        advancement_id=parent.advancement_id,
                        prerequisites=(s1_id,),
                        precondition_expr=parent.precondition_expr,
                        verification_expr=parent.verification_expr,
                        base_reward=round(parent.base_reward * 0.55, 1),
                        target_items={pat: cnt},
                        requires_movement=parent.requires_movement,
                        is_micro_task=True,
                        parent_node_id=parent.id,
                        _precondition_fn=parent._precondition_fn,
                        _postcondition_fn=parent._postcondition_fn,
                    ),
                ]

        # Generic 2-step prep + execute split (e.g., combat or dimension entry)
        _ = state
        s1_id = f"{parent.id}_sub1"
        s2_id = f"{parent.id}_sub2"
        return [
            CurriculumNode(
                id=s1_id,
                tier=parent.tier,
                description=f"Prepare food and equip gear before '{parent.id}'",
                advancement_id=None,
                prerequisites=parent.prerequisites,
                precondition_expr="None",
                verification_expr="Vitals.Health >= 14 && Vitals.FoodLevel >= 14",
                base_reward=25.0,
                target_items={},
                requires_movement=False,
                is_micro_task=True,
                parent_node_id=parent.id,
                _precondition_fn=lambda _s: True,
                _postcondition_fn=lambda s: s.vitals.health >= 14.0 and s.vitals.food_level >= 14,
            ),
            CurriculumNode(
                id=s2_id,
                tier=parent.tier,
                description=parent.description,
                advancement_id=parent.advancement_id,
                prerequisites=(s1_id,),
                precondition_expr=parent.precondition_expr,
                verification_expr=parent.verification_expr,
                base_reward=parent.base_reward,
                target_items=dict(parent.target_items),
                requires_movement=parent.requires_movement,
                is_micro_task=True,
                parent_node_id=parent.id,
                _precondition_fn=parent._precondition_fn,
                _postcondition_fn=parent._postcondition_fn,
            ),
        ]


class CurriculumManager:
    """Zero-LLM native DAG scheduler with Dynamic Regression Recovery & Softmax/Epsilon-Greedy selection."""

    DECOMPOSITION_FAILURE_THRESHOLD = 3

    def __init__(
        self,
        dag: CurriculumDAG | None = None,
        *,
        q_table: SkillQTable | None = None,
        epsilon: float = 0.0,
        softmax_temperature: float = 0.0,
        rng_seed: int | None = None,
    ) -> None:
        self.dag = dag or CurriculumDAG()
        self.q_table = q_table
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.softmax_temperature = max(0.0, float(softmax_temperature))
        self._rng = random.Random(rng_seed)
        self.completed_nodes: set[str] = set()
        self._micro_queue: list[CurriculumNode] = []

    def sync_completed_from_state(self, state: BotTelemetrySnapshot) -> None:
        """Marks already-satisfied nodes complete and invalidates regressed persistent tool/gear milestones."""
        # 1. Check regression in reverse topological order so a directly craftable higher-tier tool
        #    (e.g. t3_iron_tools_and_armor when iron_ingots are already in inventory) recovers directly
        #    without redundantly re-crafting wooden/stone tools first.
        for node in reversed(self.dag.ordered_nodes):
            if node.id in self.completed_nodes and node.persistent_milestone:
                if not node.verify_postcondition(state):
                    LOG.warning(
                        "[CURRICULUM] Regression detected! Persistent milestone '%s' lost (postcondition=%s). Scheduling recovery.",
                        node.id,
                        node.verification_expr,
                    )
                    self._invalidate_regressed_chain(node.id, state)
                    if node.check_preconditions(state):
                        break

        # 2. Fast-forward any nodes whose postconditions & prerequisites are already satisfied in live state
        changed = True
        while changed:
            changed = False
            for node in self.dag.ordered_nodes:
                if node.id in self.completed_nodes:
                    continue
                prereqs_done = all(p in self.completed_nodes for p in node.prerequisites)
                if prereqs_done and node.verify_postcondition(state):
                    self.completed_nodes.add(node.id)
                    node.consecutive_failures = 0
                    changed = True

    def _invalidate_regressed_chain(self, regressed_id: str, state: BotTelemetrySnapshot) -> None:
        """Unmarks regressed node; if its preconditions are also missing, walks back prerequisites to rebuild."""
        self.completed_nodes.discard(regressed_id)
        node = self.dag.get_node(regressed_id)
        if node.check_preconditions(state):
            return
        # Walk backwards through unsatisfied prerequisites so missing raw materials/tools are re-acquired
        stack = list(node.prerequisites)
        visited: set[str] = set()
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            pnode = self.dag.get_node(pid)
            if not pnode.verify_postcondition(state):
                self.completed_nodes.discard(pid)
                if not pnode.check_preconditions(state):
                    stack.extend(pnode.prerequisites)

    def get_next_objective(self, state: BotTelemetrySnapshot) -> CurriculumNode:
        """Purely native 0-LLM objective selector with dynamic regression and frontier utility scoring."""
        # Drain active decomposed micro-tasks first if still valid
        while self._micro_queue:
            head = self._micro_queue[0]
            if head.verify_postcondition(state):
                self._micro_queue.pop(0)
                if head.parent_node_id and not self._micro_queue:
                    parent = self.dag.get_node(head.parent_node_id)
                    if parent.verify_postcondition(state):
                        self.completed_nodes.add(parent.id)
                        parent.consecutive_failures = 0
                continue
            return head

        self.sync_completed_from_state(state)

        frontier: list[CurriculumNode] = []
        for node in self.dag.ordered_nodes:
            if node.id in self.completed_nodes:
                continue
            if all(p in self.completed_nodes for p in node.prerequisites):
                frontier.append(node)

        if not frontier:
            # All 20 nodes complete; hold final Tier 5 objective
            return self.dag.ordered_nodes[-1]

        # Backtrack if a frontier node's preconditions require an earlier consumable resource that was depleted
        satisfiable = [n for n in frontier if n.check_preconditions(state)]
        if not satisfiable:
            recovery = self._find_prerequisite_recovery(frontier[0], state)
            if recovery is not None:
                LOG.info(
                    "[CURRICULUM] Precondition recovery scheduled: '%s' -> '%s' for frontier '%s'",
                    recovery.id,
                    recovery.verification_expr,
                    frontier[0].id,
                )
                return recovery
            candidates = frontier
        else:
            candidates = satisfiable

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = self._select_by_utility(candidates, state)

        LOG.info(
            "[CURRICULUM] Selected objective '%s' (Tier %d, Reward +%.0f, PrecondOK=%s)",
            chosen.id,
            chosen.tier,
            chosen.base_reward,
            chosen.check_preconditions(state),
        )
        return chosen

    # Alias matching C# specification
    GetNextObjective = get_next_objective

    def _find_prerequisite_recovery(
        self,
        blocked_node: CurriculumNode,
        state: BotTelemetrySnapshot,
    ) -> CurriculumNode | None:
        visited: set[str] = set()
        queue_ids = list(blocked_node.prerequisites)
        deepest_unsatisfied: CurriculumNode | None = None
        while queue_ids:
            pid = queue_ids.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            pnode = self.dag.get_node(pid)
            if not pnode.verify_postcondition(state):
                deepest_unsatisfied = pnode
                if not pnode.check_preconditions(state):
                    queue_ids.extend(pnode.prerequisites)
        if deepest_unsatisfied is not None:
            self.completed_nodes.discard(deepest_unsatisfied.id)
        return deepest_unsatisfied

    def _node_utility(self, node: CurriculumNode, state: BotTelemetrySnapshot) -> float:
        tier_bonus = (6 - node.tier) * 25.0
        precond_bonus = 40.0 * node.precondition_match_ratio(state)
        persistent_bonus = 15.0 if node.persistent_milestone else 0.0
        q_bonus = 0.0
        if self.q_table is not None:
            ctx_hash = ContextStateHash.encode(state)
            q_bonus = 0.5 * self.q_table.max_q(ctx_hash, node.id)
        fail_penalty = 8.0 * node.consecutive_failures
        return tier_bonus + precond_bonus + persistent_bonus + (node.base_reward * 0.1) + q_bonus - fail_penalty

    def _select_by_utility(
        self,
        candidates: list[CurriculumNode],
        state: BotTelemetrySnapshot,
    ) -> CurriculumNode:
        utilities = [self._node_utility(c, state) for c in candidates]
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.choice(candidates)
        if self.softmax_temperature > 0.0:
            max_u = max(utilities)
            weights = [math.exp((u - max_u) / self.softmax_temperature) for u in utilities]
            total_w = sum(weights)
            r = self._rng.random() * total_w
            acc = 0.0
            for c, w in zip(candidates, weights):
                acc += w
                if r <= acc:
                    return c
        best_idx = max(range(len(candidates)), key=lambda i: utilities[i])
        return candidates[best_idx]

    def record_outcome(
        self,
        node: CurriculumNode,
        success: bool,
        state_after: BotTelemetrySnapshot,
        *,
        reflexion_lines: list[str] | None = None,
        llm_client: LocalLlmClient | None = None,
        ct: CancellationToken | None = None,
    ) -> list[CurriculumNode]:
        """Updates completion/failure state; triggers automatic sub-task decomposition at >=3 failures."""
        if success and node.verify_postcondition(state_after):
            if node.is_micro_task:
                if self._micro_queue and self._micro_queue[0].id == node.id:
                    self._micro_queue.pop(0)
                if node.parent_node_id and not self._micro_queue:
                    parent = self.dag.get_node(node.parent_node_id)
                    if parent.verify_postcondition(state_after):
                        self.completed_nodes.add(parent.id)
                        parent.consecutive_failures = 0
            else:
                self.completed_nodes.add(node.id)
                node.consecutive_failures = 0
            return []

        target_node = self.dag.get_node(node.parent_node_id) if (node.is_micro_task and node.parent_node_id) else node
        target_node.consecutive_failures += 1

        if (
            not node.is_micro_task
            and target_node.consecutive_failures >= self.DECOMPOSITION_FAILURE_THRESHOLD
            and not self._micro_queue
        ):
            subtasks = CurriculumDecomposer.split_node_sync(
                target_node,
                state_after,
                reflexion_lines or [],
                llm_client=llm_client,
                ct=ct,
            )
            target_node.consecutive_failures = 0
            self._micro_queue.extend(subtasks)
            return subtasks
        return []
