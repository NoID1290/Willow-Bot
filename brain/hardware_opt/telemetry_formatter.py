"""Module 0: Ultra-Compact Telemetry Serializer (< 150 tokens) for KV-Cache Dynamic Suffixes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .llm_client import estimate_tokens

if TYPE_CHECKING:
    from rl_rewards.telemetry_snapshot import BotTelemetrySnapshot


def _short_id(item_id: str) -> str:
    if not item_id:
        return "none"
    return item_id.split(":", 1)[-1]


def _time_phase(time_of_day: int) -> str:
    tod = int(time_of_day) % 24000
    if 12542 <= tod <= 23460:
        return "night"
    if 11800 <= tod < 12542 or 23460 < tod <= 24000:
        return "dusk"
    return "day"


PRIORITY_ITEM_KEYWORDS = (
    "pickaxe",
    "sword",
    "axe",
    "shovel",
    "shield",
    "bucket",
    "diamond",
    "obsidian",
    "flint_and_steel",
    "iron_ingot",
    "raw_iron",
    "coal",
    "charcoal",
    "torch",
    "crafting_table",
    "furnace",
    "bed",
    "cooked_",
    "bread",
    "steak",
    "beef",
    "porkchop",
    "mutton",
    "chicken",
    "log",
    "planks",
    "stick",
    "cobblestone",
)


def _item_sort_key(entry: tuple[str, int], priority_targets: set[str]) -> tuple[int, int, str]:
    raw_id, count = entry
    short = _short_id(raw_id)
    if raw_id in priority_targets or short in priority_targets:
        return (0, -count, short)
    for idx, kw in enumerate(PRIORITY_ITEM_KEYWORDS):
        if kw in short:
            return (1 + idx, -count, short)
    return (100, -count, short)


class CompactTelemetryFormatter:
    """Compresses BotTelemetrySnapshot into <150 token dense strings without raw 36-slot JSON bloat."""

    MAX_TOKENS = 145

    @classmethod
    def format(
        cls,
        snapshot: BotTelemetrySnapshot,
        target_items: dict[str, int] | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        pos = snapshot.position
        dim = _short_id(snapshot.dimension)
        phase = _time_phase(snapshot.time_of_day)
        pos_seg = f"POS:{int(round(pos.x))},{int(round(pos.y))},{int(round(pos.z))}({dim},{phase},light:{snapshot.light_level})"

        v = snapshot.vitals
        vitals_seg = (
            f"HP:{int(round(v.health))}/{int(round(v.max_health))} "
            f"FOOD:{v.food_level}/20 ARMOR:{v.armor_points}"
        )

        equip_parts: list[str] = []
        for slot_item in snapshot.equipment.values():
            if not slot_item.item_id or slot_item.item_id in ("minecraft:air", "air", "none"):
                continue
            sid = _short_id(slot_item.item_id)
            if slot_item.max_durability > 0:
                equip_parts.append(f"{sid}({slot_item.current_durability}/{slot_item.max_durability})")
            else:
                equip_parts.append(sid)
        equip_seg = "EQUIP:" + (",".join(equip_parts[:6]) if equip_parts else "none")

        priority_targets = set()
        if target_items:
            for k in target_items:
                priority_targets.add(k)
                priority_targets.add(_short_id(k))

        sorted_inv = sorted(
            ((k, c) for k, c in snapshot.item_counts.items() if c > 0),
            key=lambda item: _item_sort_key(item, priority_targets),
        )
        inv_parts = [f"{_short_id(k)}:{c}" for k, c in sorted_inv[:12]]
        inv_seg = "INV:" + (",".join(inv_parts) if inv_parts else "empty")

        ws_parts = [
            f"{_short_id(station)}@({int(round(vec.x))},{int(round(vec.y))},{int(round(vec.z))})"
            for station, vec in list(snapshot.nearby_workstations.items())[:4]
        ]
        ws_seg = "WORKSTATIONS:" + (",".join(ws_parts) if ws_parts else "none")

        hostiles = [e for e in snapshot.nearby_entities if e.is_hostile]
        hostiles.sort(key=lambda e: e.distance)
        threat_parts = [f"{_short_id(e.entity_type)}@{int(round(e.distance))}m" for e in hostiles[:3]]
        threat_seg = "THREATS:" + (",".join(threat_parts) if threat_parts else "none")

        line = " | ".join((pos_seg, vitals_seg, equip_seg, inv_seg, ws_seg, threat_seg))
        if estimate_tokens(line) <= max_tokens:
            return line

        # Progressive deterministic tightening if inventory list is unusually dense
        for inv_cap in (8, 5, 3):
            shorter_inv = "INV:" + (",".join(inv_parts[:inv_cap]) if inv_parts else "empty")
            candidate = " | ".join((pos_seg, vitals_seg, equip_seg[:48], shorter_inv, ws_seg[:48], threat_seg))
            if estimate_tokens(candidate) <= max_tokens:
                return candidate

        max_chars = int(max_tokens * 3.2)
        return line[:max_chars]
