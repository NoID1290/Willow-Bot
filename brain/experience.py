"""Persistent, measured situation/action lookup; never treat model claims as facts."""

import json
import time


def inventory(observation):
    result = {}
    for entry in observation.get("inventory", []):
        item, amount = entry.rsplit("=", 1)
        result[item] = result.get(item, 0) + int(amount)
    return result


def situation(observation):
    position = observation.get("position", [0, 0, 0])
    if not isinstance(position, list) or len(position) != 3 or any(type(v) is not int for v in position):
        raise ValueError("Observation position must contain three integers")
    region = [position[0] // 8, position[1] // 4, position[2] // 8]
    tools = sorted(item for item in inventory(observation) if item.endswith(("_pickaxe", "_axe", "_shovel", "_sword", "_hoe")))
    return json.dumps([observation.get("dimension", "unknown"), region, tools], separators=(",", ":"))


class Experience:
    def __init__(self, db, q_table=None):
        self.db = db
        self.q_table = q_table
        db.executescript("""
            CREATE TABLE IF NOT EXISTS decision_context
              (id TEXT PRIMARY KEY, situation TEXT, inventory TEXT, position TEXT);
            CREATE TABLE IF NOT EXISTS action_lookup
              (situation TEXT, action TEXT, target TEXT, attempts INTEGER DEFAULT 0,
               successes INTEGER DEFAULT 0, failures INTEGER DEFAULT 0, consecutive_failures INTEGER DEFAULT 0,
               damage REAL DEFAULT 0, last_detail TEXT, last_delta TEXT, updated REAL, blocked_until REAL DEFAULT 0,
               PRIMARY KEY(situation,action,target));
        """)

    def remember(self, decision_id, observation):
        self.db.execute("INSERT INTO decision_context VALUES (?,?,?,?)",
                        (decision_id, situation(observation), json.dumps(inventory(observation)),
                         json.dumps(observation.get("position", [0, 0, 0]))))
        self.db.execute("DELETE FROM decision_context WHERE id NOT IN (SELECT id FROM decisions)")

    def learn(self, decision_id, action, target, observation):
        before = self.db.execute("SELECT situation,inventory FROM decision_context WHERE id=?",
                                 (decision_id,)).fetchone()
        if not before:
            return
        result = observation["last_result"]
        old, new = json.loads(before[1]), inventory(observation)
        delta = {item: new.get(item, 0) - old.get(item, 0) for item in old.keys() | new.keys()
                 if new.get(item, 0) != old.get(item, 0)}
        status = result.get("status")
        if status == "interrupted":
            # Eating reflex / player chat / pause cut the action short: no evidence either way.
            return delta
        success = status == "success"
        now = time.time()
        self.db.execute("""
            INSERT INTO action_lookup (situation,action,target,attempts,successes,failures,
                consecutive_failures,damage,last_detail,last_delta,updated)
            VALUES (?,?,?,1,?,?,?,?,?,?,?)
            ON CONFLICT(situation,action,target) DO UPDATE SET
                attempts=attempts+1, successes=successes+excluded.successes,
                failures=failures+excluded.failures,
                consecutive_failures=CASE WHEN excluded.successes=1 THEN 0 ELSE consecutive_failures+1 END,
                damage=damage+excluded.damage, last_detail=excluded.last_detail,
                last_delta=excluded.last_delta, updated=excluded.updated
        """, (before[0], action, target, int(success), int(not success), int(not success),
              result.get("damage", 0), str(result.get("detail", ""))[:240], json.dumps(delta), now))
        self.db.execute("""
            UPDATE action_lookup SET blocked_until=CASE WHEN consecutive_failures>=2 THEN ? ELSE 0 END
            WHERE situation=? AND action=? AND target=?
        """, (now + 120, before[0], action, target))
        self.db.execute("DELETE FROM action_lookup WHERE rowid NOT IN "
                        "(SELECT rowid FROM action_lookup ORDER BY updated DESC LIMIT 10000)")

        if self.q_table is not None and not success:
            try:
                target_str = target.split(":", 1)[-1] if target else ""
                skill_id = f"{action}_{target_str}" if target_str else f"action_{action}"
                ctx_hash = hash(before[0]) & 0xFFFFFFFF
                self.q_table.update(
                    context_prev=ctx_hash,
                    task_prev=action,
                    skill_id=skill_id,
                    reward_total=-20.0,
                    context_next=ctx_hash,
                    task_next=action,
                )
            except Exception:
                pass

        return delta

    def context(self, observation):
        key = situation(observation)
        rows = self.db.execute("""
            SELECT action,target,attempts,successes,failures,damage,last_detail,last_delta,blocked_until
            FROM action_lookup WHERE situation=? ORDER BY updated DESC LIMIT 24
        """, (key,)).fetchall()
        lookup = [dict(action=r[0], target=r[1], attempts=r[2], successes=r[3], failures=r[4],
                       damage=r[5], last_detail=r[6], inventory_change=json.loads(r[7])) for r in rows]
        blocked = [dict(action=r[0], target=r[1], reason=r[6]) for r in rows if r[8] > time.time()]
        if self.q_table is not None:
            try:
                ctx_hash = hash(key) & 0xFFFFFFFF
                for item in lookup:
                    t_str = item.get("target", "").split(":", 1)[-1] if item.get("target") else ""
                    sid = f"{item['action']}_{t_str}" if t_str else f"action_{item['action']}"
                    q_val = self.q_table.get_context_skill_q(ctx_hash, sid)
                    if q_val <= -15.0:
                        if not any(b["action"] == item["action"] and b.get("target") == item.get("target") for b in blocked):
                            blocked.append(dict(action=item["action"], target=item.get("target", ""), reason=f"Low Q-value ({q_val:+.1f})"))
            except Exception:
                pass
        return lookup, blocked

    def changes(self, observation):
        before = self.db.execute("SELECT inventory FROM decision_context WHERE id=?",
                                 (observation.get("last_result", {}).get("id", ""),)).fetchone()
        if not before:
            return {}
        old, new = json.loads(before[0]), inventory(observation)
        return {item: new.get(item, 0) - old.get(item, 0) for item in old.keys() | new.keys()
                if new.get(item, 0) != old.get(item, 0)}
