"""Willow's long-term world memory, kept forever in the world's brain database.

Everything here is measured by the game (placements, mining, inventory changes,
positions) except notes and waypoint names, which Willow chooses himself.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from experience import inventory

# Generous safety caps so a very long world cannot grow the brain without bound.
MAX_PLACED = 100_000
MAX_SEEN = 20_000
MAX_EVENTS = 50_000
MAX_NOTES = 2_000
MAX_WAYPOINTS = 256
WATCH_RANGE = 24
COMMON = {"minecraft:dirt", "minecraft:grass_block", "minecraft:stone", "minecraft:sand", "minecraft:gravel",
          "minecraft:short_grass", "minecraft:tall_grass", "minecraft:snow", "minecraft:deepslate",
          "minecraft:netherrack", "minecraft:water", "minecraft:fern", "minecraft:large_fern",
          "minecraft:sandstone", "minecraft:granite", "minecraft:diorite", "minecraft:andesite",
          "minecraft:tuff", "minecraft:bedrock", "minecraft:coarse_dirt", "minecraft:podzol",
          "minecraft:leaf_litter", "minecraft:wildflowers", "minecraft:pink_petals", "minecraft:bush",
          "minecraft:firefly_bush", "minecraft:short_dry_grass", "minecraft:tall_dry_grass",
          "minecraft:dead_bush", "minecraft:vine", "minecraft:glow_lichen", "minecraft:seagrass",
          "minecraft:tall_seagrass", "minecraft:kelp", "minecraft:kelp_plant"}
WAYPOINT_NAME = re.compile(r"[A-Za-z0-9 _'-]{1,48}")


def coordinates(text):
    """Parses Minecraft's BlockPos.toShortString(): "x, y, z"."""
    if not isinstance(text, str):
        return None
    parts = text.split(",")
    try:
        values = [int(part.strip()) for part in parts]
    except ValueError:
        return None
    return values if len(values) == 3 else None


def leaves(kind):
    return kind.endswith("_leaves")


class Memory:
    def __init__(self, db, export_path=None):
        self.db = db
        self.export_path = Path(export_path) if export_path else None
        self._observations_since_prune = 0
        db.executescript("""
            CREATE TABLE IF NOT EXISTS placed_blocks
              (dimension TEXT, x INTEGER, y INTEGER, z INTEGER, block TEXT, placed REAL,
               checked REAL, state TEXT DEFAULT 'present', PRIMARY KEY(dimension,x,y,z));
            CREATE TABLE IF NOT EXISTS waypoints
              (name TEXT PRIMARY KEY, dimension TEXT, x INTEGER, y INTEGER, z INTEGER,
               note TEXT, created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS notes
              (id TEXT PRIMARY KEY, text TEXT, created REAL, used REAL, times INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS knowledge
              (action TEXT, target TEXT, attempts INTEGER DEFAULT 0, successes INTEGER DEFAULT 0,
               damage REAL DEFAULT 0, gains TEXT DEFAULT '{}', last_detail TEXT, updated REAL,
               PRIMARY KEY(action,target));
            CREATE TABLE IF NOT EXISTS discovered_items
              (item TEXT PRIMARY KEY, first REAL, dimension TEXT, x INTEGER, y INTEGER, z INTEGER);
            CREATE TABLE IF NOT EXISTS seen_blocks
              (dimension TEXT, block TEXT, x INTEGER, y INTEGER, z INTEGER, seen REAL,
               PRIMARY KEY(dimension,x,y,z));
            CREATE TABLE IF NOT EXISTS events
              (id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, kind TEXT, text TEXT,
               dimension TEXT, x INTEGER, y INTEGER, z INTEGER);
        """)
        db.commit()

    # ---- writing -------------------------------------------------------------------------

    def event(self, kind, text, dimension=None, position=None):
        x, y, z = position if position else (None, None, None)
        self.db.execute("INSERT INTO events (time,kind,text,dimension,x,y,z) VALUES (?,?,?,?,?,?,?)",
                        (time.time(), kind, text[:300], dimension, x, y, z))

    def observe(self, observation):
        """Stores what this observation proves: positions, discoveries, block checks."""
        now = time.time()
        dimension = str(observation.get("dimension", "unknown"))
        here = observation.get("position")
        if not (isinstance(here, list) and len(here) == 3 and all(type(v) is int for v in here)):
            return
        if not self.db.execute("SELECT 1 FROM waypoints WHERE name='first_spawn'").fetchone():
            self.save_waypoint("first_spawn", observation, "Where Willow first woke up in this world", event=False)
        self.db.execute("INSERT OR REPLACE INTO waypoints VALUES ('last_position',?,?,?,?,?,"
                        "COALESCE((SELECT created FROM waypoints WHERE name='last_position'),?),?)",
                        (dimension, *here, "Updated automatically every decision", now, now))
        for item, count in inventory(observation).items():
            if count > 0 and self.db.execute(
                    "INSERT OR IGNORE INTO discovered_items VALUES (?,?,?,?,?,?)",
                    (item, now, dimension, *here)).rowcount:
                self.event("discovery", f"First obtained {item}", dimension, here)
        nearby_positions = set()
        for block in observation.get("nearby_blocks", []):
            if not isinstance(block, dict):
                continue
            kind, at = block.get("id"), block.get("at")
            if isinstance(at, list) and len(at) == 3 and all(type(v) is int for v in at):
                nearby_positions.add(tuple(at))
                if isinstance(kind, str) and len(kind) <= 96 and kind not in COMMON and not leaves(kind):
                    self.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)",
                                    (dimension, kind, *at, now))
        if isinstance(observation.get("nearby_blocks"), list):
            candidates = self.db.execute(
                "SELECT x,y,z FROM seen_blocks WHERE dimension=? AND abs(x-?)<=3 AND abs(y-?)<=3 AND abs(z-?)<=3 "
                "AND NOT EXISTS (SELECT 1 FROM placed_blocks WHERE placed_blocks.dimension=seen_blocks.dimension "
                "AND placed_blocks.x=seen_blocks.x AND placed_blocks.y=seen_blocks.y AND placed_blocks.z=seen_blocks.z "
                "AND placed_blocks.state='present')",
                (dimension, here[0], here[1], here[2]),
            ).fetchall()
            to_delete = [(x, y, z) for x, y, z in candidates if (x, y, z) not in nearby_positions]
            if to_delete:
                placeholders = ",".join("(?,?,?)" for _ in to_delete)
                flat_params = [dimension] + [coord for pos in to_delete for coord in pos]
                self.db.execute(
                    f"DELETE FROM seen_blocks WHERE dimension=? AND (x, y, z) IN (VALUES {placeholders})",
                    flat_params,
                )
        for checked in observation.get("remembered_blocks", []):
            if not isinstance(checked, dict):
                continue
            at, actual = checked.get("at"), checked.get("id")
            if not (isinstance(at, list) and len(at) == 3 and all(type(v) is int for v in at)
                    and isinstance(actual, str)):
                continue
            row = self.db.execute("SELECT block,state FROM placed_blocks WHERE dimension=? AND x=? AND y=? AND z=?",
                                  (dimension, *at)).fetchone()
            if not row:
                continue
            state = "present" if actual == row[0] else ("gone" if actual.endswith("air") else "replaced:" + actual)
            if state != row[1]:
                self.event("placed_block_changed", f"{row[0]} I placed at {at} is now {state}", dimension, at)
            self.db.execute("UPDATE placed_blocks SET state=?, checked=? WHERE dimension=? AND x=? AND y=? AND z=?",
                            (state, now, dimension, *at))
            if actual not in COMMON and not leaves(actual) and not actual.endswith("air"):
                self.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)",
                                (dimension, actual, *at, now))
            else:
                self.db.execute("DELETE FROM seen_blocks WHERE dimension=? AND x=? AND y=? AND z=?",
                                (dimension, *at))
        # Caps are generous, so five COUNT(*) scans per decision are unnecessary; prune periodically.
        self._observations_since_prune += 1
        if self._observations_since_prune >= 25:
            self._observations_since_prune = 0
            self.prune()

    def record(self, action, target, result, delta, dimension):
        """Stores an action's measured outcome permanently and keeps the placed-block map exact."""
        if result.get("status") == "interrupted":
            return
        now = time.time()
        success = result.get("status") == "success"
        detail = str(result.get("detail", ""))[:240]
        row = self.db.execute("SELECT gains FROM knowledge WHERE action=? AND target=?", (action, target)).fetchone()
        gains = json.loads(row[0]) if row else {}
        if success:
            for item, change in (delta or {}).items():
                gains[item] = gains.get(item, 0) + change
        self.db.execute("""
            INSERT INTO knowledge (action,target,attempts,successes,damage,gains,last_detail,updated)
            VALUES (?,?,1,?,?,?,?,?)
            ON CONFLICT(action,target) DO UPDATE SET attempts=attempts+1, successes=successes+excluded.successes,
                damage=damage+excluded.damage, gains=excluded.gains, last_detail=excluded.last_detail,
                updated=excluded.updated
        """, (action, target, int(success), float(result.get("damage", 0) or 0), json.dumps(gains, sort_keys=True),
              detail, now))
        at = coordinates(result.get("block_position"))
        if not success or at is None:
            return
        if action == "place":
            self.db.execute("INSERT OR REPLACE INTO placed_blocks VALUES (?,?,?,?,?,?,?, 'present')",
                            (dimension, *at, target, now, now))
            self.db.execute("INSERT OR REPLACE INTO seen_blocks VALUES (?,?,?,?,?,?)", (dimension, target, *at, now))
            self.event("placed", f"Placed {target} at {at}", dimension, at)
        elif action in ("mine", "escape"):
            placed = self.db.execute("SELECT block FROM placed_blocks WHERE dimension=? AND x=? AND y=? AND z=? "
                                     "AND state='present'", (dimension, *at)).fetchone()
            if placed:
                self.db.execute("UPDATE placed_blocks SET state='gone', checked=? WHERE dimension=? AND x=? AND y=? "
                                "AND z=?", (now, dimension, *at))
                self.event("removed_own_block", f"Mined my own {placed[0]} at {at}", dimension, at)
            self.db.execute("DELETE FROM seen_blocks WHERE dimension=? AND x=? AND y=? AND z=?", (dimension, *at))

    def note(self, text, observation):
        text = " ".join(str(text).split())[:240]
        if len(text) < 3:
            return
        now = time.time()
        key = hashlib.sha256(text.lower().encode()).hexdigest()
        if self.db.execute("UPDATE notes SET used=?, times=times+1 WHERE id=?", (now, key)).rowcount == 0:
            self.db.execute("INSERT INTO notes VALUES (?,?,?,?,1)", (key, text, now, now))
            self.event("note", text, observation.get("dimension"), observation.get("position"))

    def save_waypoint(self, name, observation, note="", event=True):
        name = " ".join(str(name).split())[:48]
        here = observation.get("position")
        if not WAYPOINT_NAME.fullmatch(name) or name == "last_position" or not (
                isinstance(here, list) and len(here) == 3 and all(type(v) is int for v in here)):
            return False
        now = time.time()
        dimension = str(observation.get("dimension", "unknown"))
        existing = self.db.execute("SELECT dimension,x,y,z,updated FROM waypoints WHERE name=?", (name,)).fetchone()
        if existing:
            if list(existing[:4]) == [dimension, *here]:
                return False
            chat = " ".join(observation.get("chat", [])).lower()
            chat_requests_waypoint = any(w in chat for w in ("waypoint", "home", "base", "set", "save", name.lower()))
            if existing[0] == dimension and not chat_requests_waypoint:
                if max(abs(existing[1] - here[0]), abs(existing[2] - here[1]), abs(existing[3] - here[2])) < 8:
                    return False
                # Models tend to re-save the same name every decision; a named place should not drift that fast.
                if now - float(existing[4] or 0) < 180:
                    return False
        self.db.execute("""
            INSERT INTO waypoints VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET dimension=excluded.dimension, x=excluded.x, y=excluded.y,
                z=excluded.z, note=excluded.note, updated=excluded.updated
        """, (name, dimension, *here, note[:160], now, now))
        if event:
            self.event("waypoint", f"Saved waypoint {name} at {here}", dimension, here)
        return True

    def prune(self):
        for table, order, limit in (("placed_blocks", "placed", MAX_PLACED), ("seen_blocks", "seen", MAX_SEEN),
                                    ("events", "id", MAX_EVENTS), ("notes", "used", MAX_NOTES),
                                    ("waypoints", "updated", MAX_WAYPOINTS)):
            count = self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count > limit:
                self.db.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} "
                                f"ORDER BY {order} ASC LIMIT ?)", (count - limit,))

    # ---- reading -------------------------------------------------------------------------

    def watch_list(self, observation):
        """Remembered placements near Willow; the game reports their real current block."""
        here = observation.get("position", [0, 0, 0])
        rows = self.db.execute("""
            SELECT x,y,z FROM placed_blocks WHERE dimension=? AND state!='gone'
              AND abs(x-?)<=? AND abs(z-?)<=? AND abs(y-?)<=48
            ORDER BY (x-?)*(x-?)+(y-?)*(y-?)+(z-?)*(z-?) LIMIT 48
        """, (observation.get("dimension", "unknown"), here[0], WATCH_RANGE, here[2], WATCH_RANGE, here[1],
              here[0], here[0], here[1], here[1], here[2], here[2])).fetchall()
        return [list(row) for row in rows]

    def context(self, observation):
        dimension = observation.get("dimension", "unknown")
        here = observation.get("position", [0, 0, 0])

        def distance(row):
            return max(abs(row[0] - here[0]), abs(row[1] - here[1]), abs(row[2] - here[2]))

        placed = self.db.execute("""
            SELECT block,x,y,z,state FROM placed_blocks WHERE dimension=? AND state='present'
            ORDER BY (x-?)*(x-?)+(y-?)*(y-?)+(z-?)*(z-?) LIMIT 24
        """, (dimension, here[0], here[0], here[1], here[1], here[2], here[2])).fetchall()
        totals = dict(self.db.execute("SELECT block,COUNT(*) FROM placed_blocks WHERE state='present' "
                                      "GROUP BY block ORDER BY COUNT(*) DESC LIMIT 12").fetchall())
        waypoints = sorted(self.db.execute("SELECT name,dimension,x,y,z,note FROM waypoints WHERE name!='last_position'")
                           .fetchall(), key=lambda r: (r[1] != dimension, distance(r[2:5])))[:16]
        visible = {b.get("id") for b in observation.get("nearby_blocks", []) if isinstance(b, dict)}
        remembered = []
        for kind, x, y, z in self.db.execute("""
                SELECT block,x,y,z FROM (
                  SELECT block,x,y,z,ROW_NUMBER() OVER (PARTITION BY block ORDER BY d) AS n, d FROM (
                    SELECT block,x,y,z,(x-?)*(x-?)+(y-?)*(y-?)+(z-?)*(z-?) AS d FROM seen_blocks
                    WHERE dimension=?))
                WHERE n=1 ORDER BY d LIMIT 40
            """, (*[v for v in here for _ in (0, 1)], dimension)):
            dist = distance((x, y, z))
            if kind not in visible and dist > 3:
                remembered.append({"id": kind, "at": [x, y, z], "distance": dist})
        notes = [r[0] for r in self.db.execute("SELECT text FROM notes ORDER BY used DESC LIMIT 16")]
        knowledge = []
        for action, target, attempts, successes, damage, gains, detail in self.db.execute(
                "SELECT action,target,attempts,successes,damage,gains,last_detail FROM knowledge "
                "ORDER BY updated DESC LIMIT 24"):
            entry = f"{action} {target}".strip() + f": {successes}/{attempts} ok"
            total = {k: v for k, v in json.loads(gains).items() if v}
            if total and successes:
                entry += " avg " + " ".join(f"{v / successes:+.0f} {k.split(':')[-1]}" for k, v in total.items())
            if damage:
                entry += f" dmg {damage:.0f}"
            if successes < attempts and detail:
                entry += f" last: {detail[:80]}"
            knowledge.append(entry)
        events = [r[0] for r in self.db.execute(
            "SELECT text FROM events WHERE kind!='note' ORDER BY id DESC LIMIT 10")]
        return {
            "my_placed_blocks": [{"id": r[0], "at": list(r[1:4]), "distance": distance(r[1:4])} for r in placed],
            "my_placed_totals": totals,
            "waypoints": [{"name": r[0], "at": list(r[2:5]), "distance": distance(r[2:5]),
                           **({"dimension": r[1]} if r[1] != dimension else {}),
                           **({"note": r[5]} if r[5] else {})} for r in waypoints],
            "remembered_resources": remembered[:16],
            "my_notes": notes,
            "knowledge": knowledge,
            "discovered_items": self.db.execute("SELECT COUNT(*) FROM discovered_items").fetchone()[0],
            "memory_events": events,
        }

    def export(self):
        """Writes a human-readable copy of the whole memory next to the brain database."""
        if not self.export_path:
            return
        data = {
            "waypoints": [dict(zip(("name", "dimension", "x", "y", "z", "note"), r)) for r in self.db.execute(
                "SELECT name,dimension,x,y,z,note FROM waypoints ORDER BY name")],
            "placed_blocks": [dict(zip(("dimension", "x", "y", "z", "block", "state"), r)) for r in self.db.execute(
                "SELECT dimension,x,y,z,block,state FROM placed_blocks ORDER BY placed")],
            "notes": [r[0] for r in self.db.execute("SELECT text FROM notes ORDER BY created")],
            "knowledge": [dict(zip(("action", "target", "attempts", "successes", "damage", "gains", "last_detail"),
                                   (*r[:5], json.loads(r[5]), r[6]))) for r in self.db.execute(
                "SELECT action,target,attempts,successes,damage,gains,last_detail FROM knowledge ORDER BY action,target")],
            "discovered_items": [r[0] for r in self.db.execute("SELECT item FROM discovered_items ORDER BY first")],
            "recent_events": [dict(zip(("time", "kind", "text"), r)) for r in self.db.execute(
                "SELECT time,kind,text FROM events ORDER BY id DESC LIMIT 500")],
        }
        temporary = self.export_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=1))
        os.replace(temporary, self.export_path)
