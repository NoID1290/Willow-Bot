"""Module 3: High-Level Self-Healing Bot Control API (IWillowBotActions & WillowBotActions)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Awaitable, Callable, Protocol

from curriculum.mechanics_registry import RecipeKnowledgeGraph, SurvivalMechanicsRegistry
from hardware_opt.llm_client import CancellationToken, NONE_TOKEN
from hardware_opt.memory_pool import MemoryPoolManager
from rl_rewards.telemetry_snapshot import (
    COOKED_FOOD_ITEMS,
    EDIBLE_FOOD_ITEMS,
    BlockInfo,
    BotTelemetrySnapshot,
    EquipmentSlot,
    EquippedItemState,
    ExecutionTelemetryRecorder,
    PerceivedEntity,
    Vector3,
    VitalsState,
    matches_item_pattern,
    normalize_id,
)

from .pathfinder import MAX_SCAN_RADIUS, SpiralBlockScanner, TimeSlicedAStar

LOG = logging.getLogger("localai.actions")

VALID_ERROR_CODES = frozenset(
    {"NONE", "PATH_BLOCKED", "MISSING_ITEM", "NO_TOOL", "HOSTILE_INTERRUPT", "TIMEOUT"}
)


@dataclass(frozen=True, slots=True)
class ActionResult:
    Success: bool
    ErrorCode: str = "NONE"
    Message: str = ""

    @property
    def success(self) -> bool:
        return self.Success

    @property
    def error_code(self) -> str:
        return self.ErrorCode

    @property
    def message(self) -> str:
        return self.Message

    @staticmethod
    def ok(message: str = "OK") -> ActionResult:
        return ActionResult(True, "NONE", message)

    @staticmethod
    def fail(error_code: str, message: str) -> ActionResult:
        code = error_code if error_code in VALID_ERROR_CODES else "MISSING_ITEM"
        return ActionResult(False, code, message)


class CombatTactic(str, Enum):
    SmartAuto = "SmartAuto"
    MeleeKite = "MeleeKite"
    ShieldAndCounter = "ShieldAndCounter"
    Retreat = "Retreat"


class IWillowBotActions(Protocol):
    """High-level defensive & self-healing bot control contract for 7B-8B local models."""

    def GetState(self) -> BotTelemetrySnapshot: ...
    def GetItemCount(self, itemNameOrPattern: str) -> int: ...
    def HasItem(self, itemNameOrPattern: str, minCount: int = 1) -> bool: ...
    def FindBlocks(self, blockIdPattern: str, searchRadius: int = 32, maxResults: int = 16) -> list[BlockInfo]: ...
    def FindEntities(self, entityIdPattern: str, searchRadius: int = 24, onlyAlive: bool = True) -> list[PerceivedEntity]: ...
    def FindSafeSurfaceOrWater(self, searchRadius: int = 32) -> Vector3 | None: ...

    async def MoveToAsync(self, targetPos: Vector3, arrivalRadius: float = 2.0, timeoutMs: int = 20000) -> ActionResult: ...
    async def ExploreAsync(self, preferredDirection: Vector3, distance: int = 32, timeoutMs: int = 25000) -> ActionResult: ...
    async def DigStaircaseToYAsync(self, targetY: int, placeTorches: bool = True, ct: CancellationToken | None = None) -> ActionResult: ...

    async def MineBlocksAsync(self, blockIdPattern: str, targetCount: int, maxSearchRadius: int = 32) -> ActionResult: ...
    async def MineBlockAtAsync(self, blockPos: Vector3, collectDrops: bool = True) -> ActionResult: ...
    async def PlaceBlockAsync(self, itemId: str, placeAtPos: Vector3) -> ActionResult: ...
    async def UseBucketAtAsync(self, bucketType: str, targetBlockPos: Vector3) -> ActionResult: ...

    async def EnsureWorkstationPlacedAsync(self, stationItemId: str) -> ActionResult: ...
    async def ReclaimWorkstationAsync(self, stationBlockId: str) -> ActionResult: ...
    async def CraftAsync(self, outputItemId: str, craftOperationsCount: int = 1) -> ActionResult: ...
    async def SmeltAsync(self, inputItemId: str, fuelItemId: str, count: int = 1) -> ActionResult: ...

    async def EquipBestToolForBlockAsync(self, blockId: str) -> ActionResult: ...
    async def EquipArmorAndWeaponsAsync() -> ActionResult: ...
    async def EatIfHungryAsync(self, hungerThreshold: int = 15) -> ActionResult: ...
    async def EngageEntityAsync(self, entityId: int, tactic: CombatTactic = CombatTactic.SmartAuto) -> ActionResult: ...
    async def BuildSimpleShelterAsync(self, blockMaterial: str = "cobblestone") -> ActionResult: ...
    async def WaitTicksAsync(self, ticks: int) -> None: ...
    def Log(self, message: str) -> None: ...


ActionCommandExecutor = Callable[[dict[str, Any], CancellationToken], Awaitable[ActionResult]]


class WillowBotActions:
    """Concrete high-level bot controller with harvest-tier guards, 2.5ms A*, and auto-2x2 crafting."""

    def __init__(
        self,
        initial_state: BotTelemetrySnapshot,
        *,
        ct: CancellationToken | None = None,
        pool: MemoryPoolManager | None = None,
        telemetry_recorder: ExecutionTelemetryRecorder | None = None,
        command_executor: ActionCommandExecutor | None = None,
        state_provider: Callable[[], BotTelemetrySnapshot] | None = None,
        is_walkable: Callable[[int, int, int], bool] | None = None,
        voxel_lookup: Callable[[int, int, int], str | None] | None = None,
        tick_duration_sec: float = 0.0,
    ) -> None:
        self._ct = ct or NONE_TOKEN
        self._pool = pool
        self._recorder = telemetry_recorder or ExecutionTelemetryRecorder()
        self._command_executor = command_executor
        self._state_provider = state_provider
        self._voxel_lookup = voxel_lookup
        self._tick_duration_sec = max(0.0, float(tick_duration_sec))
        self._pathfinder = TimeSlicedAStar(is_walkable=is_walkable)
        self._logs: list[str] = []
        self._emitted_steps: list[dict[str, Any]] = []

        # Mutable working overlay so multi-step scripts reflect intermediate craft/mine/place updates
        self._pos = initial_state.position
        self._dimension = initial_state.dimension
        self._biome = initial_state.biome
        self._light_level = initial_state.light_level
        self._world_tick = initial_state.world_tick
        self._time_of_day = initial_state.time_of_day
        self._vitals = initial_state.vitals
        self._items: dict[str, int] = dict(initial_state.item_counts)
        self._equipment: dict[EquipmentSlot, EquippedItemState] = dict(initial_state.equipment)
        self._advancements: set[str] = set(initial_state.unlocked_advancements)
        self._discovered: set[str] = set(initial_state.discovered_unique_items)
        self._crafted: set[str] = set(initial_state.crafted_unique_items)
        self._entities: list[PerceivedEntity] = list(initial_state.nearby_entities)
        self._workstations: dict[str, Vector3] = dict(initial_state.nearby_workstations)
        self._blocks: list[BlockInfo] = list(initial_state.nearby_blocks)
        self._enclosed_shelter = initial_state.enclosed_lit_shelter

    @property
    def telemetry_recorder(self) -> ExecutionTelemetryRecorder:
        return self._recorder

    @property
    def emitted_steps(self) -> list[dict[str, Any]]:
        return list(self._emitted_steps)

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    def bind_cancellation_token(self, ct: CancellationToken) -> None:
        self._ct = ct

    def _check_ct(self, override_ct: CancellationToken | None = None) -> CancellationToken:
        active = override_ct if override_ct is not None else self._ct
        active.throw_if_cancellation_requested()
        return active

    def _record_step(self, action: str, target: str = "", pos: Vector3 | None = None) -> dict[str, Any]:
        bx, by, bz = pos.to_block_tuple() if pos is not None else (0, 0, 0)
        step = {"action": action, "target": target, "x": bx, "y": by, "z": bz}
        if len(self._emitted_steps) < 32:
            self._emitted_steps.append(step)
        return step

    async def _dispatch_if_live(self, step: dict[str, Any], ct: CancellationToken) -> ActionResult | None:
        if self._command_executor is None:
            return None
        try:
            res = await self._command_executor(step, ct)
            if self._state_provider is not None:
                self._sync_from_snapshot(self._state_provider())
            return res
        except OperationCanceledError:
            raise
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            self._recorder.record_exception(err_msg)
            raise

    def _sync_from_snapshot(self, snap: BotTelemetrySnapshot) -> None:
        prev_mh = self._equipment.get(EquipmentSlot.MAIN_HAND)
        if prev_mh and prev_mh.item_id and prev_mh.item_id != "minecraft:air":
            new_mh = snap.equipment.get(EquipmentSlot.MAIN_HAND)
            if (not new_mh or new_mh.item_id == "minecraft:air" or new_mh.current_durability <= 0):
                if snap.count_item(prev_mh.item_id) < self.GetItemCount(prev_mh.item_id):
                    self._recorder.record_broken_tool(prev_mh.item_id)

        self._pos = snap.position
        self._dimension = snap.dimension
        self._biome = snap.biome
        self._light_level = snap.light_level
        self._world_tick = snap.world_tick
        self._time_of_day = snap.time_of_day
        self._vitals = snap.vitals
        self._items = dict(snap.item_counts)
        self._equipment = dict(snap.equipment)
        self._advancements |= set(snap.unlocked_advancements)
        self._discovered |= set(snap.discovered_unique_items)
        self._crafted |= set(snap.crafted_unique_items)
        self._entities = list(snap.nearby_entities)
        self._workstations = dict(snap.nearby_workstations)
        self._blocks = list(snap.nearby_blocks)
        self._enclosed_shelter = snap.enclosed_lit_shelter

    def _add_item(self, item_id: str, count: int) -> None:
        if count <= 0:
            return
        norm = normalize_id(item_id)
        self._items[norm] = self._items.get(norm, 0) + count
        self._discovered.add(norm)

    def _consume_pattern(self, pattern: str, count: int) -> bool:
        available = self.GetItemCount(pattern)
        if available < count:
            return False
        rem = count
        for k in list(self._items.keys()):
            if rem <= 0:
                break
            if self._items[k] > 0 and matches_item_pattern(k, pattern):
                take = min(rem, self._items[k])
                self._items[k] -= take
                if self._items[k] <= 0:
                    del self._items[k]
                rem -= take
        return rem == 0

    def _damage_equipped_tool(self, amount: int = 1) -> None:
        mh = self._equipment.get(EquipmentSlot.MAIN_HAND)
        if mh is None or not mh.item_id or mh.max_durability <= 0:
            return
        new_dur = mh.current_durability - amount
        if new_dur <= 0:
            broken_id = mh.item_id
            self._recorder.record_broken_tool(broken_id)
            self._consume_pattern(broken_id, 1)
            self._equipment[EquipmentSlot.MAIN_HAND] = EquippedItemState("minecraft:air", 0, 0)
        else:
            self._equipment[EquipmentSlot.MAIN_HAND] = EquippedItemState(mh.item_id, new_dur, mh.max_durability)

    # ---- Perception & Queries ------------------------------------------------------------

    def GetState(self) -> BotTelemetrySnapshot:
        self._check_ct()
        if self._state_provider is not None:
            snap = self._state_provider()
            self._sync_from_snapshot(snap)
        used_slots = min(36, sum(max(1, (c + 63) // 64) for c in self._items.values() if c > 0))
        return BotTelemetrySnapshot(
            timestamp_ms=int(self._world_tick * 50),
            world_tick=self._world_tick,
            time_of_day=self._time_of_day % 24000,
            position=self._pos,
            dimension=self._dimension,
            biome=self._biome,
            light_level=self._light_level,
            vitals=self._vitals,
            item_counts=dict(self._items),
            empty_slots_count=max(0, 36 - used_slots),
            equipment=dict(self._equipment),
            unlocked_advancements=frozenset(self._advancements),
            discovered_unique_items=frozenset(self._discovered),
            crafted_unique_items=frozenset(self._crafted),
            nearby_entities=tuple(self._entities[:12]),
            nearby_workstations=dict(self._workstations),
            execution_telemetry=self._recorder.freeze(),
            nearby_blocks=tuple(self._blocks),
            enclosed_lit_shelter=self._enclosed_shelter,
        )

    def GetItemCount(self, itemNameOrPattern: str) -> int:
        self._check_ct()
        total = 0
        for item_id, cnt in self._items.items():
            if cnt > 0 and matches_item_pattern(item_id, itemNameOrPattern):
                total += cnt
        return total

    def HasItem(self, itemNameOrPattern: str, minCount: int = 1) -> bool:
        self._check_ct()
        return self.GetItemCount(itemNameOrPattern) >= max(1, int(minCount))

    def FindBlocks(
        self,
        blockIdPattern: str,
        searchRadius: int = 32,
        maxResults: int = 16,
    ) -> list[BlockInfo]:
        self._check_ct()
        capped_radius = max(1, min(MAX_SCAN_RADIUS, int(searchRadius)))
        return SpiralBlockScanner.scan(
            origin=self._pos,
            blocks=self._blocks,
            pattern=blockIdPattern,
            search_radius=capped_radius,
            max_results=maxResults,
            pool=self._pool,
            voxel_lookup=self._voxel_lookup,
        )

    def FindEntities(
        self,
        entityIdPattern: str,
        searchRadius: int = 24,
        onlyAlive: bool = True,
    ) -> list[PerceivedEntity]:
        self._check_ct()
        capped = max(1, min(32, int(searchRadius)))
        buf = self._pool.rent_entity_buffer() if self._pool is not None else []
        try:
            for ent in self._entities:
                if onlyAlive and not ent.is_alive:
                    continue
                if ent.distance <= capped and (
                    entityIdPattern in ("*", "any", "")
                    or matches_item_pattern(ent.entity_type, entityIdPattern)
                    or (ent.entity_type == "minecraft:item" and matches_item_pattern(ent.item_id, entityIdPattern))
                ):
                    buf.append(ent)
            return list(buf)
        finally:
            if self._pool is not None:
                self._pool.return_entity_buffer(buf)

    def FindSafeSurfaceOrWater(self, searchRadius: int = 32) -> Vector3 | None:
        self._check_ct()
        waters = self.FindBlocks("minecraft:water", searchRadius=min(MAX_SCAN_RADIUS, searchRadius), maxResults=4)
        if waters:
            return waters[0].position
        return Vector3(self._pos.x, max(64.0, self._pos.y), self._pos.z)

    # ---- Movement & Exploration ----------------------------------------------------------

    async def MoveToAsync(
        self,
        targetPos: Vector3,
        arrivalRadius: float = 2.0,
        timeoutMs: int = 20000,
    ) -> ActionResult:
        ct = self._check_ct()
        if not isinstance(targetPos, Vector3):
            targetPos = Vector3(float(targetPos[0]), float(targetPos[1]), float(targetPos[2]))

        waypoints, err_code = await self._pathfinder.find_path_async(
            self._pos,
            targetPos,
            arrival_radius=arrival_radius_clamp(arrivalRadius),
            timeout_ms=timeoutMs,
            ct=ct,
        )
        if waypoints is None:
            self._recorder.pathfinding_timeouts += 1
            return ActionResult.fail(err_code, f"No path to {targetPos.to_block_tuple()}")

        step = self._record_step("goto", "", targetPos)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            if not live_res.Success and live_res.ErrorCode in ("TIMEOUT", "PATH_BLOCKED"):
                self._recorder.pathfinding_timeouts += 1
            return live_res

        self._pos = waypoints[-1]
        self._world_tick += max(1, len(waypoints) * 4)
        return ActionResult.ok(f"Moved to {self._pos.to_block_tuple()} ({len(waypoints)} steps)")

    async def ExploreAsync(
        self,
        preferredDirection: Vector3 | None = None,
        distance: int = 32,
        timeoutMs: int = 25000,
    ) -> ActionResult:
        ct = self._check_ct()
        dist = max(4, min(MAX_SCAN_RADIUS, int(distance)))
        if isinstance(preferredDirection, Vector3):
            mag = max(0.001, (preferredDirection.x ** 2 + preferredDirection.z ** 2) ** 0.5)
            dx = (preferredDirection.x / mag) * dist
            dz = (preferredDirection.z / mag) * dist
        else:
            dx, dz = float(dist), 0.0
        dest = Vector3(round(self._pos.x + dx), self._pos.y, round(self._pos.z + dz))
        return await self.MoveToAsync(dest, arrivalRadius=3.0, timeoutMs=timeoutMs)

    async def DigStaircaseToYAsync(
        self,
        targetY: int,
        placeTorches: bool = True,
        ct: CancellationToken | None = None,
    ) -> ActionResult:
        token = self._check_ct(ct)
        state = self.GetState()
        if state.highest_tool_tier("_pickaxe") < 1:
            self._recorder.improper_tool_mine_attempts += 1
            return ActionResult.fail("NO_TOOL", "DigStaircaseToYAsync requires at least a wooden_pickaxe.")

        cur_x, cur_y, cur_z = self._pos.to_block_tuple()
        goal_y = int(targetY)
        if cur_y == goal_y:
            return ActionResult.ok(f"Already at Y={goal_y}")

        step_dy = -1 if goal_y < cur_y else 1
        steps_taken = 0
        while cur_y != goal_y:
            token.throw_if_cancellation_requested()
            next_x = cur_x + 1
            next_y = cur_y + step_dy
            # Safety check: inspect adjacent blocks for lava or water before stepping
            if self._voxel_lookup is not None:
                for adj_dx, adj_dy, adj_dz in ((0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)):
                    adj_id = self._voxel_lookup(next_x + adj_dx, next_y + adj_dy, cur_z + adj_dz)
                    if adj_id and ("lava" in adj_id or "water" in adj_id):
                        return ActionResult.fail(
                            "PATH_BLOCKED",
                            f"Aborted staircase at Y={cur_y}: adjacent fluid {adj_id} detected!",
                        )
            for blk in self._blocks:
                if ("lava" in blk.block_id or "water" in blk.block_id) and blk.position.distance_to(
                    Vector3(float(next_x), float(next_y), float(cur_z))
                ) <= 1.6:
                    return ActionResult.fail("PATH_BLOCKED", f"Hazard fluid {blk.block_id} ahead on staircase.")

            # Check deep deepslate pickaxe requirement below Y=0
            if next_y < 0 and self.GetState().highest_tool_tier("_pickaxe") < 1:
                self._recorder.improper_tool_mine_attempts += 1
                return ActionResult.fail("NO_TOOL", "Pickaxe broke while digging staircase.")

            cur_x, cur_y = next_x, next_y
            steps_taken += 1
            self._recorder.blocks_mined_count += 3
            drop_item = "minecraft:cobbled_deepslate" if cur_y < 0 else "minecraft:cobblestone"
            self._add_item(drop_item, 1)
            self._damage_equipped_tool(1)

            if placeTorches and steps_taken % 6 == 0 and self.HasItem("minecraft:torch", 1):
                self._consume_pattern("minecraft:torch", 1)
                self._light_level = 14

            if steps_taken % 4 == 0:
                await asyncio.sleep(0)

        self._pos = Vector3(float(cur_x), float(cur_y), float(cur_z))
        step_cmd = self._record_step("tunnel", "", self._pos)
        live_res = await self._dispatch_if_live(step_cmd, token)
        if live_res is not None:
            return live_res
        self._world_tick += steps_taken * 10
        return ActionResult.ok(f"Dug safe stepped staircase to Y={cur_y} ({steps_taken} steps)")

    # ---- Block Harvesting & Placement ----------------------------------------------------

    async def MineBlocksAsync(
        self,
        blockIdPattern: str,
        targetCount: int,
        maxSearchRadius: int = 32,
    ) -> ActionResult:
        ct = self._check_ct()
        needed = max(1, int(targetCount))
        snap = self.GetState()
        ok_tool, _req_tier, req_tool = SurvivalMechanicsRegistry.validate_harvest_tool(blockIdPattern, snap)
        if not ok_tool:
            self._recorder.improper_tool_mine_attempts += 1
            return ActionResult.fail(
                "NO_TOOL",
                f"Cannot mine '{blockIdPattern}': requires >= {req_tool} (0 drops with current gear).",
            )

        await self.EquipBestToolForBlockAsync(blockIdPattern)
        found = self.FindBlocks(blockIdPattern, searchRadius=min(MAX_SCAN_RADIUS, maxSearchRadius), maxResults=max(16, needed))
        if not found:
            return ActionResult.fail("MISSING_ITEM", f"No '{blockIdPattern}' blocks found within {min(MAX_SCAN_RADIUS, maxSearchRadius)}m.")

        mined = 0
        for blk in found:
            if mined >= needed:
                break
            ct.throw_if_cancellation_requested()
            res = await self.MineBlockAtAsync(blk.position, collectDrops=True, _expected_block_id=blk.block_id)
            if not res.Success:
                if mined > 0:
                    break
                return res
            mined += 1

        return ActionResult.ok(f"Mined {mined}/{needed} matching '{blockIdPattern}'")

    async def MineBlockAtAsync(
        self,
        blockPos: Vector3,
        collectDrops: bool = True,
        *,
        _expected_block_id: str | None = None,
    ) -> ActionResult:
        ct = self._check_ct()
        target_block_id = _expected_block_id or ""
        matched_idx = -1
        for idx, blk in enumerate(self._blocks):
            if blk.position.to_block_tuple() == blockPos.to_block_tuple():
                target_block_id = blk.block_id
                matched_idx = idx
                break
        if not target_block_id and self._voxel_lookup is not None:
            bx, by, bz = blockPos.to_block_tuple()
            target_block_id = self._voxel_lookup(bx, by, bz) or ""
        if not target_block_id:
            return ActionResult.fail("MISSING_ITEM", f"No mineable block at {blockPos.to_block_tuple()}")

        # Prevent digging straight beneath the bot's own feet
        fx, fy, fz = self._pos.to_block_tuple()
        tx, ty, tz = blockPos.to_block_tuple()
        if tx == fx and tz == fz and ty == fy - 1:
            return ActionResult.fail("PATH_BLOCKED", "Refusing to dig straight down beneath bot's feet.")

        snap = self.GetState()
        ok_tool, _tier, req_tool = SurvivalMechanicsRegistry.validate_harvest_tool(target_block_id, snap)
        if not ok_tool:
            self._recorder.improper_tool_mine_attempts += 1
            return ActionResult.fail(
                "NO_TOOL",
                f"Cannot mine '{target_block_id}': requires >= {req_tool}.",
            )

        step = self._record_step("mine", normalize_id(target_block_id), blockPos)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            if live_res.Success:
                self._recorder.blocks_mined_count += 1
            return live_res

        if matched_idx >= 0:
            self._blocks.pop(matched_idx)
        self._recorder.blocks_mined_count += 1
        self._damage_equipped_tool(1)
        if collectDrops:
            drop_id = _drop_for_block(target_block_id)
            self._add_item(drop_id, 1)
            if "stone" in target_block_id or "cobblestone" in target_block_id:
                self._advancements.add("minecraft:story/mine_stone")
            elif "diamond_ore" in target_block_id:
                self._advancements.add("minecraft:story/mine_diamond")
            elif "obsidian" in target_block_id:
                self._advancements.add("minecraft:story/form_obsidian")
            elif target_block_id.endswith("_log"):
                self._advancements.add("minecraft:story/root")
        self._world_tick += 15
        return ActionResult.ok(f"Mined {target_block_id} at {blockPos.to_block_tuple()}")

    async def PlaceBlockAsync(self, itemId: str, placeAtPos: Vector3) -> ActionResult:
        ct = self._check_ct()
        norm = normalize_id(itemId)
        if not self.HasItem(norm, 1):
            return ActionResult.fail("MISSING_ITEM", f"Block '{norm}' is not in inventory.")
        step = self._record_step("place", norm, placeAtPos)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        self._consume_pattern(norm, 1)
        short = norm.split(":", 1)[-1]
        if short in ("crafting_table", "furnace", "chest") or short.endswith("_bed"):
            key = "bed" if short.endswith("_bed") else short
            self._workstations[key] = placeAtPos
        if short == "torch":
            self._light_level = max(self._light_level, 14)
        self._blocks.append(BlockInfo(block_id=norm, position=placeAtPos))
        self._world_tick += 4
        return ActionResult.ok(f"Placed {norm} at {placeAtPos.to_block_tuple()}")

    async def UseBucketAtAsync(self, bucketType: str, targetBlockPos: Vector3) -> ActionResult:
        ct = self._check_ct()
        norm_bucket = normalize_id(bucketType)
        if not self.HasItem(norm_bucket, 1):
            return ActionResult.fail("MISSING_ITEM", f"Missing '{norm_bucket}' in inventory.")
        step = self._record_step("place", norm_bucket, targetBlockPos)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        if norm_bucket == "minecraft:bucket":
            # Fill water bucket if water nearby, or lava bucket if lava nearby
            has_water = any(
                "water" in b.block_id and b.position.distance_to(targetBlockPos) <= 4.0 for b in self._blocks
            )
            has_lava = any(
                "lava" in b.block_id and b.position.distance_to(targetBlockPos) <= 4.0 for b in self._blocks
            )
            if has_water:
                self._consume_pattern("minecraft:bucket", 1)
                self._add_item("minecraft:water_bucket", 1)
                return ActionResult.ok("Filled water_bucket")
            if has_lava:
                self._consume_pattern("minecraft:bucket", 1)
                self._add_item("minecraft:lava_bucket", 1)
                self._advancements.add("minecraft:story/lava_bucket")
                return ActionResult.ok("Filled lava_bucket")
            return ActionResult.fail("MISSING_ITEM", "No fluid source at target position to fill bucket.")

        if norm_bucket == "minecraft:water_bucket":
            # Convert nearby lava into obsidian if present
            converted = 0
            for idx, b in enumerate(self._blocks):
                if "lava" in b.block_id and b.position.distance_to(targetBlockPos) <= 4.0:
                    self._blocks[idx] = BlockInfo("minecraft:obsidian", b.position)
                    converted += 1
            if converted > 0:
                self._advancements.add("minecraft:story/form_obsidian")
            return ActionResult.ok(f"Used water_bucket at {targetBlockPos.to_block_tuple()} (obsidian_formed={converted})")

        return ActionResult.ok(f"Used {norm_bucket} at {targetBlockPos.to_block_tuple()}")

    # ---- Workstations, Crafting & Smelting -----------------------------------------------

    async def EnsureWorkstationPlacedAsync(self, stationItemId: str) -> ActionResult:
        ct = self._check_ct()
        norm = normalize_id(stationItemId)
        short = norm.split(":", 1)[-1]
        ws_key = "bed" if short.endswith("_bed") else short

        existing = self._workstations.get(ws_key)
        if existing is not None and self._pos.distance_to(existing) <= 4.5:
            return ActionResult.ok(f"{ws_key} already within reach at {existing.to_block_tuple()}")
        if existing is not None and self._pos.distance_to(existing) <= 16.0:
            move_res = await self.MoveToAsync(existing, arrivalRadius=2.5)
            if move_res.Success:
                return ActionResult.ok(f"Approached existing {ws_key}")

        if not self.HasItem(norm, 1):
            craft_res = await self.CraftAsync(norm, 1)
            if not craft_res.Success:
                return craft_res

        place_pos = Vector3(self._pos.x + 1.0, self._pos.y, self._pos.z)
        return await self.PlaceBlockAsync(norm, place_pos)

    async def ReclaimWorkstationAsync(self, stationBlockId: str) -> ActionResult:
        self._check_ct()
        norm = normalize_id(stationBlockId)
        short = norm.split(":", 1)[-1]
        ws_key = "bed" if short.endswith("_bed") else short
        ws_pos = self._workstations.get(ws_key)
        if ws_pos is None:
            found = self.FindBlocks(norm, searchRadius=8, maxResults=1)
            if not found:
                return ActionResult.fail("MISSING_ITEM", f"No placed '{norm}' nearby to reclaim.")
            ws_pos = found[0].position
        res = await self.MineBlockAtAsync(ws_pos, collectDrops=True, _expected_block_id=norm)
        if res.Success:
            self._workstations.pop(ws_key, None)
        return res

    async def CraftAsync(self, outputItemId: str, craftOperationsCount: int = 1) -> ActionResult:
        ct = self._check_ct()
        ops = max(1, int(craftOperationsCount))
        spec = RecipeKnowledgeGraph.get_recipe(outputItemId)
        if spec is None:
            return ActionResult.fail("MISSING_ITEM", f"Unknown crafting recipe for '{outputItemId}'.")

        # Automatically resolve intermediate 2x2 sticks FIRST, then planks, so stick crafting never steals
        # planks needed directly by the recipe when uncrafted logs remain in inventory!
        if "minecraft:stick" in spec.ingredients:
            stick_req = spec.ingredients["minecraft:stick"] * ops
            stick_have = self.GetItemCount("minecraft:stick")
            if stick_have < stick_req:
                stick_batches = (stick_req - stick_have + 3) // 4
                planks_for_sticks = stick_batches * 2
                direct_planks_req = spec.ingredients.get("*_planks", 0) * ops
                total_planks_needed = planks_for_sticks + direct_planks_req
                if self.GetItemCount("*_planks") < total_planks_needed:
                    plank_deficit = total_planks_needed - self.GetItemCount("*_planks")
                    logs_needed = (plank_deficit + 3) // 4
                    if self.GetItemCount("*_log") >= logs_needed:
                        self._consume_pattern("*_log", logs_needed)
                        self._add_item("minecraft:oak_planks", logs_needed * 4)
                        self._crafted.add("minecraft:oak_planks")
                if self.GetItemCount("*_planks") >= planks_for_sticks:
                    self._consume_pattern("*_planks", planks_for_sticks)
                    self._add_item("minecraft:stick", stick_batches * 4)
                    self._crafted.add("minecraft:stick")

        if "*_planks" in spec.ingredients:
            plank_req = spec.ingredients["*_planks"] * ops
            plank_have = self.GetItemCount("*_planks")
            if plank_have < plank_req:
                logs_needed = (plank_req - plank_have + 3) // 4
                if self.GetItemCount("*_log") >= logs_needed:
                    self._consume_pattern("*_log", logs_needed)
                    self._add_item("minecraft:oak_planks", logs_needed * 4)
                    self._crafted.add("minecraft:oak_planks")

        if spec.requires_crafting_table:
            ws_pos = self._workstations.get("crafting_table")
            if ws_pos is None or self._pos.distance_to(ws_pos) > 4.5:
                if self.HasItem("minecraft:crafting_table", 1):
                    place_res = await self.EnsureWorkstationPlacedAsync("minecraft:crafting_table")
                    if not place_res.Success:
                        return place_res
                else:
                    return ActionResult.fail(
                        "MISSING_ITEM",
                        f"Crafting '{spec.output_item}' requires a placed crafting_table within 4 blocks.",
                    )

        # Verify all required ingredients (supporting cobbled_deepslate substitute for cobblestone)
        for ing_pat, ing_cnt in spec.ingredients.items():
            req = ing_cnt * ops
            if ing_pat == "minecraft:cobblestone":
                avail = self.GetItemCount("minecraft:cobblestone") + self.GetItemCount("minecraft:cobbled_deepslate")
            else:
                avail = self.GetItemCount(ing_pat)
            if avail < req:
                return ActionResult.fail(
                    "MISSING_ITEM",
                    f"Missing ingredient '{ing_pat}' ({avail}/{req}) to craft '{spec.output_item}'.",
                )

        step = self._record_step("craft", spec.output_item)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        for ing_pat, ing_cnt in spec.ingredients.items():
            req = ing_cnt * ops
            if ing_pat == "minecraft:cobblestone":
                cobble_have = self.GetItemCount("minecraft:cobblestone")
                take_cobble = min(req, cobble_have)
                if take_cobble > 0:
                    self._consume_pattern("minecraft:cobblestone", take_cobble)
                if req - take_cobble > 0:
                    self._consume_pattern("minecraft:cobbled_deepslate", req - take_cobble)
            else:
                self._consume_pattern(ing_pat, req)

        produced = spec.output_count * ops
        self._add_item(spec.output_item, produced)
        self._crafted.add(spec.output_item)
        _grant_craft_advancements(spec.output_item, self._advancements)
        self._world_tick += 2 * ops
        return ActionResult.ok(f"Crafted {produced}x {spec.output_item}")

    async def SmeltAsync(self, inputItemId: str, fuelItemId: str, count: int = 1) -> ActionResult:
        ct = self._check_ct()
        amt = max(1, int(count))
        norm_in = normalize_id(inputItemId)
        norm_fuel = normalize_id(fuelItemId) if fuelItemId else "minecraft:coal"

        out_id = SurvivalMechanicsRegistry.SMELTING_OUTPUTS.get(norm_in)
        if out_id is None and norm_in.endswith("_log"):
            out_id = "minecraft:charcoal"
        if out_id is None:
            return ActionResult.fail("MISSING_ITEM", f"Item '{norm_in}' cannot be smelted.")

        ws_pos = self._workstations.get("furnace")
        if ws_pos is None or self._pos.distance_to(ws_pos) > 4.5:
            ensure_res = await self.EnsureWorkstationPlacedAsync("minecraft:furnace")
            if not ensure_res.Success:
                return ensure_res

        if not self.HasItem(norm_in, amt):
            return ActionResult.fail("MISSING_ITEM", f"Insufficient '{norm_in}' ({self.GetItemCount(norm_in)}/{amt}).")

        items_per_fuel = SurvivalMechanicsRegistry.SMELTING_FUELS.get(norm_fuel, 1.5)
        fuel_needed = max(1, int((amt + items_per_fuel - 0.01) // items_per_fuel))
        if not self.HasItem(norm_fuel, fuel_needed):
            return ActionResult.fail("MISSING_ITEM", f"Insufficient fuel '{norm_fuel}' ({self.GetItemCount(norm_fuel)}/{fuel_needed}).")

        step = self._record_step("smelt", norm_in)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        self._consume_pattern(norm_in, amt)
        self._consume_pattern(norm_fuel, fuel_needed)
        self._add_item(out_id, amt)
        if out_id == "minecraft:iron_ingot":
            self._advancements.add("minecraft:story/smelt_iron")
        self._world_tick += 20 * amt
        return ActionResult.ok(f"Smelted {amt}x {norm_in} -> {amt}x {out_id}")

    # ---- Equipment, Sustenance & Combat --------------------------------------------------

    async def EquipBestToolForBlockAsync(self, blockId: str) -> ActionResult:
        self._check_ct()
        short = blockId.split(":", 1)[-1]
        preferred_suffix = "_pickaxe"
        if short.endswith("_log") or "planks" in short:
            preferred_suffix = "_axe"
        elif short in ("dirt", "grass_block", "sand", "gravel"):
            preferred_suffix = "_shovel"

        tier_order = ("netherite", "diamond", "iron", "stone", "wooden", "golden")
        durability_defaults = {"netherite": 2031, "diamond": 1561, "iron": 250, "stone": 131, "wooden": 59, "golden": 32}
        for tier in tier_order:
            cand = f"minecraft:{tier}{preferred_suffix}"
            if self.HasItem(cand, 1):
                dur = durability_defaults.get(tier, 100)
                self._equipment[EquipmentSlot.MAIN_HAND] = EquippedItemState(cand, dur, dur)
                return ActionResult.ok(f"Equipped {cand}")

        if preferred_suffix != "_pickaxe":
            return ActionResult.ok("Using hand or current item")
        ok, _, req_name = SurvivalMechanicsRegistry.validate_harvest_tool(blockId, self.GetState())
        if not ok:
            return ActionResult.fail("NO_TOOL", f"Missing >= {req_name} for {blockId}")
        return ActionResult.ok("Using current hand")

    async def EquipArmorAndWeaponsAsync(self) -> ActionResult:
        self._check_ct()
        armor_slots = (
            (EquipmentSlot.HEAD, "_helmet", {"netherite": 3, "diamond": 3, "iron": 2, "leather": 1}),
            (EquipmentSlot.CHEST, "_chestplate", {"netherite": 8, "diamond": 8, "iron": 6, "leather": 3}),
            (EquipmentSlot.LEGS, "_leggings", {"netherite": 6, "diamond": 6, "iron": 5, "leather": 2}),
            (EquipmentSlot.FEET, "_boots", {"netherite": 3, "diamond": 3, "iron": 2, "leather": 1}),
        )
        armor_total = 0
        for slot, suffix, pts_map in armor_slots:
            for tier in ("netherite", "diamond", "iron", "leather"):
                cand = f"minecraft:{tier}{suffix}"
                if self.HasItem(cand, 1) or self._equipment.get(slot, EquippedItemState("")).item_id == cand:
                    self._equipment[slot] = EquippedItemState(cand, 165, 165)
                    armor_total += pts_map[tier]
                    break
        if armor_total > 0:
            self._advancements.add("minecraft:story/obtain_armor")
        self._vitals = VitalsState(
            health=self._vitals.health,
            max_health=self._vitals.max_health,
            absorption=self._vitals.absorption,
            food_level=self._vitals.food_level,
            saturation=self._vitals.saturation,
            armor_points=max(self._vitals.armor_points, armor_total),
            oxygen=self._vitals.oxygen,
            is_alive=self._vitals.is_alive,
        )
        if self.HasItem("minecraft:shield", 1):
            self._equipment[EquipmentSlot.OFF_HAND] = EquippedItemState("minecraft:shield", 336, 336)
        for tier in ("netherite", "diamond", "iron", "stone", "wooden"):
            sword = f"minecraft:{tier}_sword"
            if self.HasItem(sword, 1):
                self._equipment[EquipmentSlot.MAIN_HAND] = EquippedItemState(sword, 250, 250)
                break
        return ActionResult.ok(f"Equipped armor ({self._vitals.armor_points} pts) and weapons")

    async def EatIfHungryAsync(self, hungerThreshold: int = 15) -> ActionResult:
        ct = self._check_ct()
        if self._vitals.food_level > int(hungerThreshold):
            return ActionResult.ok(f"Food sufficient ({self._vitals.food_level}/20)")

        chosen_food = ""
        restore = 4
        for food_id in tuple(COOKED_FOOD_ITEMS) + tuple(EDIBLE_FOOD_ITEMS):
            if self.HasItem(food_id, 1):
                chosen_food = food_id
                restore = 8 if food_id in COOKED_FOOD_ITEMS else 4
                break
        if not chosen_food:
            return ActionResult.fail("MISSING_ITEM", "No edible food in inventory.")

        step = self._record_step("eat", chosen_food)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        self._consume_pattern(chosen_food, 1)
        new_food = min(20, self._vitals.food_level + restore)
        self._vitals = VitalsState(
            health=min(self._vitals.max_health, self._vitals.health + (2.0 if new_food >= 18 else 0.0)),
            max_health=self._vitals.max_health,
            absorption=self._vitals.absorption,
            food_level=new_food,
            saturation=min(20.0, self._vitals.saturation + float(restore)),
            armor_points=self._vitals.armor_points,
            oxygen=self._vitals.oxygen,
            is_alive=self._vitals.is_alive,
        )
        self._world_tick += 32
        return ActionResult.ok(f"Consumed {chosen_food} -> food={new_food}/20")

    async def EngageEntityAsync(
        self,
        entityId: int,
        tactic: CombatTactic = CombatTactic.SmartAuto,
    ) -> ActionResult:
        ct = self._check_ct()
        _ = tactic
        target_ent: PerceivedEntity | None = None
        target_idx = -1
        for idx, ent in enumerate(self._entities):
            if ent.entity_id == int(entityId) and ent.is_alive:
                target_ent = ent
                target_idx = idx
                break
        if target_ent is None:
            return ActionResult.fail("MISSING_ITEM", f"Entity {entityId} not found or already dead.")

        step = self._record_step("attack", str(entityId), target_ent.position)
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        self._entities.pop(target_idx)
        self._damage_equipped_tool(2)
        if target_ent.is_hostile:
            self._advancements.add("minecraft:adventure/kill_a_mob")
        else:
            short_type = target_ent.entity_type.split(":", 1)[-1]
            meat_drops = {
                "cow": "minecraft:beef",
                "pig": "minecraft:porkchop",
                "sheep": "minecraft:mutton",
                "chicken": "minecraft:chicken",
                "rabbit": "minecraft:rabbit",
            }
            if short_type in meat_drops:
                self._add_item(meat_drops[short_type], 2)
                self._advancements.add("minecraft:husbandry/root")
            if short_type == "sheep":
                self._add_item("minecraft:white_wool", 1)
        self._world_tick += 25
        return ActionResult.ok(f"Defeated {target_ent.entity_type} (id={entityId})")

    async def BuildSimpleShelterAsync(self, blockMaterial: str = "cobblestone") -> ActionResult:
        ct = self._check_ct()
        norm_mat = normalize_id(blockMaterial)
        needed = 12
        if not self.HasItem(norm_mat, needed):
            for alt in ("minecraft:cobblestone", "minecraft:cobbled_deepslate", "minecraft:dirt", "*_planks"):
                if self.HasItem(alt, needed):
                    norm_mat = alt
                    break
            else:
                return ActionResult.fail("MISSING_ITEM", f"Need >= {needed} solid blocks for shelter.")

        step = self._record_step("shelter", "")
        live_res = await self._dispatch_if_live(step, ct)
        if live_res is not None:
            return live_res

        self._consume_pattern(norm_mat, needed)
        if self.HasItem("minecraft:torch", 1):
            self._consume_pattern("minecraft:torch", 1)
            self._light_level = 14
        if self._light_level >= 8:
            self._enclosed_shelter = True
        self._world_tick += 40
        return ActionResult.ok(f"Constructed enclosed shelter using {norm_mat}")

    async def WaitTicksAsync(self, ticks: int) -> None:
        ct = self._check_ct()
        clamped = max(1, min(1200, int(ticks)))
        self._world_tick += clamped
        if self._tick_duration_sec > 0.0:
            await asyncio.sleep(clamped * self._tick_duration_sec)
        else:
            await asyncio.sleep(0)
        ct.throw_if_cancellation_requested()

    def Log(self, message: str) -> None:
        self._check_ct()
        msg = str(message)[:240]
        if len(self._logs) < 64:
            self._logs.append(msg)
        LOG.info("[BOT-LOG] %s", msg)


def arrival_radius_clamp(radius: float) -> float:
    return max(0.5, min(8.0, float(radius)))


def _drop_for_block(block_id: str) -> str:
    short = block_id.split(":", 1)[-1]
    rule = SurvivalMechanicsRegistry.HARVEST_MATRIX.get(short)
    if rule is not None:
        return rule.drops_item
    return normalize_id(block_id)


def _grant_craft_advancements(output_item: str, adv_set: set[str]) -> None:
    short = output_item.split(":", 1)[-1]
    if short == "stone_pickaxe":
        adv_set.add("minecraft:story/upgrade_tools")
    elif short == "iron_pickaxe":
        adv_set.add("minecraft:story/iron_tools")
    elif short in ("iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots"):
        adv_set.add("minecraft:story/obtain_armor")
    elif short in ("diamond_helmet", "diamond_chestplate", "diamond_leggings", "diamond_boots", "diamond_pickaxe", "diamond_sword"):
        adv_set.add("minecraft:story/shiny_gear")
