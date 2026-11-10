"""Module 3: 2.5ms/Tick Time-Sliced A* Pathfinder & Pre-Computed Spiral Chunk-Section Scanner."""

from __future__ import annotations

import asyncio
import heapq
import math
import time
from typing import Callable

from hardware_opt.llm_client import CancellationToken, NONE_TOKEN
from hardware_opt.memory_pool import MemoryPoolManager
from rl_rewards.telemetry_snapshot import BlockInfo, Vector3, matches_item_pattern

MAX_SCAN_RADIUS = 32
MAX_PATH_TIME_SLICE_NS = 2_500_000  # Strict 2.5 ms per tick budget


def _build_spiral_offsets(max_radius: int) -> tuple[tuple[int, int, int], ...]:
    """Pre-computes concentric horizontal spiral offsets (dx, dz, ring_radius) once at import time."""
    coords: list[tuple[int, int, int]] = [(0, 0, 0)]
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            coords.append((dx, -r, r))
            coords.append((dx, r, r))
        for dz in range(-r + 1, r):
            coords.append((-r, dz, r))
            coords.append((r, dz, r))
    return tuple(coords)


SPIRAL_OFFSETS_32: tuple[tuple[int, int, int], ...] = _build_spiral_offsets(MAX_SCAN_RADIUS)


class SpiralBlockScanner:
    """Scans observed or chunk-indexed blocks in concentric rings (<= 32 radius) with pooled buffers."""

    @staticmethod
    def scan(
        origin: Vector3,
        blocks: tuple[BlockInfo, ...] | list[BlockInfo],
        pattern: str,
        search_radius: int = 32,
        max_results: int = 16,
        pool: MemoryPoolManager | None = None,
        voxel_lookup: Callable[[int, int, int], str | None] | None = None,
    ) -> list[BlockInfo]:
        capped_radius = max(1, min(MAX_SCAN_RADIUS, int(search_radius)))
        capped_max = max(1, min(64, int(max_results)))
        max_dist_sq = float(capped_radius * capped_radius)

        buf = pool.rent_block_buffer() if pool is not None else []
        try:
            if voxel_lookup is not None:
                ox, oy, oz = origin.to_block_tuple()
                for dx, dz, ring in SPIRAL_OFFSETS_32:
                    if ring > capped_radius:
                        break
                    for dy in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                        bx, by, bz = ox + dx, oy + dy, oz + dz
                        bid = voxel_lookup(bx, by, bz)
                        if bid and matches_item_pattern(bid, pattern):
                            bpos = Vector3(float(bx), float(by), float(bz))
                            if origin.distance_squared(bpos) <= max_dist_sq:
                                buf.append(BlockInfo(block_id=bid, position=bpos))
                                if len(buf) >= capped_max:
                                    return list(buf)
                return list(buf)

            # Fast scan over observed snapshot blocks sorted by spiral distance
            for blk in blocks:
                if origin.distance_squared(blk.position) <= max_dist_sq and matches_item_pattern(blk.block_id, pattern):
                    buf.append(blk)
            buf.sort(key=lambda b: origin.distance_squared(b.position))
            return list(buf[:capped_max])
        finally:
            if pool is not None:
                pool.return_block_buffer(buf)


class TimeSlicedAStar:
    """3D Minecraft A* pathfinder strictly capped at 2.5 ms CPU time per game tick."""

    DIRECTIONS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(
        self,
        is_walkable: Callable[[int, int, int], bool] | None = None,
        *,
        max_slice_ns: int = MAX_PATH_TIME_SLICE_NS,
        max_nodes: int = 4096,
    ) -> None:
        self._is_walkable = is_walkable or (lambda _x, _y, _z: True)
        self.max_slice_ns = max(100_000, int(max_slice_ns))
        self.max_nodes = max(64, int(max_nodes))
        self.last_slices_used = 0
        self.last_max_slice_duration_ns = 0

    async def find_path_async(
        self,
        start: Vector3,
        goal: Vector3,
        *,
        arrival_radius: float = 2.0,
        timeout_ms: int = 20_000,
        ct: CancellationToken | None = None,
    ) -> tuple[list[Vector3] | None, str]:
        """Returns (waypoints, error_code) yielding control whenever a tick slice hits 2.5 ms."""
        token = ct or NONE_TOKEN
        token.throw_if_cancellation_requested()

        sx, sy, sz = start.to_block_tuple()
        gx, gy, gz = goal.to_block_tuple()
        start_node = (sx, sy, sz)
        goal_node = (gx, gy, gz)
        arr_sq = max(1.0, float(arrival_radius * arrival_radius))

        if (sx - gx) ** 2 + (sy - gy) ** 2 + (sz - gz) ** 2 <= arr_sq:
            self.last_slices_used = 1
            return [Vector3(float(gx), float(gy), float(gz))], "NONE"

        deadline_ns = time.perf_counter_ns() + int(timeout_ms) * 1_000_000
        open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        g_score: dict[tuple[int, int, int], float] = {start_node: 0.0}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        closed: set[tuple[int, int, int]] = set()

        counter = 0
        h0 = self._heuristic(start_node, goal_node)
        heapq.heappush(open_heap, (h0, counter, start_node))

        slice_start_ns = time.perf_counter_ns()
        slices = 1
        max_observed_slice_ns = 0
        expanded = 0

        while open_heap and expanded < self.max_nodes:
            now_ns = time.perf_counter_ns()
            if now_ns >= deadline_ns:
                self.last_slices_used = slices
                return None, "TIMEOUT"

            slice_elapsed = now_ns - slice_start_ns
            if slice_elapsed >= self.max_slice_ns:
                if slice_elapsed > max_observed_slice_ns:
                    max_observed_slice_ns = slice_elapsed
                token.throw_if_cancellation_requested()
                # Yield to main event loop / next game tick so CPU never starves server ticks
                await asyncio.sleep(0)
                slices += 1
                slice_start_ns = time.perf_counter_ns()

            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expanded += 1

            cx, cy, cz = current
            dist_sq = (cx - gx) ** 2 + (cy - gy) ** 2 + (cz - gz) ** 2
            if dist_sq <= arr_sq:
                final_elapsed = time.perf_counter_ns() - slice_start_ns
                self.last_max_slice_duration_ns = max(max_observed_slice_ns, final_elapsed)
                self.last_slices_used = slices
                return self._reconstruct(came_from, current), "NONE"

            cur_g = g_score[current]
            for dx, dz in self.DIRECTIONS_4:
                for dy in (0, 1, -1, -2, -3):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    neighbor = (nx, ny, nz)
                    if neighbor in closed:
                        continue
                    if not self._is_walkable(nx, ny, nz):
                        continue
                    step_cost = 1.0 + (0.6 * abs(dy))
                    tentative_g = cur_g + step_cost
                    if tentative_g < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        counter += 1
                        f_score = tentative_g + self._heuristic(neighbor, goal_node)
                        heapq.heappush(open_heap, (f_score, counter, neighbor))

        self.last_slices_used = slices
        return None, "PATH_BLOCKED"

    @staticmethod
    def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        return math.fabs(a[0] - b[0]) + math.fabs(a[1] - b[1]) + math.fabs(a[2] - b[2])

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        current: tuple[int, int, int],
    ) -> list[Vector3]:
        rev: list[Vector3] = [Vector3(float(current[0]), float(current[1]), float(current[2]))]
        while current in came_from:
            current = came_from[current]
            rev.append(Vector3(float(current[0]), float(current[1]), float(current[2])))
        rev.reverse()
        return rev
