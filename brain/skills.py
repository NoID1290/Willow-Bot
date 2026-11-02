"""Small interpreted skill language. Generated source is never executed as Python."""

import ast
import hashlib
import json
import logging
from pathlib import Path
import re
import textwrap
import threading
import time
import urllib.request

LOG = logging.getLogger("localai")
SEED = "def skill():\n    return foundation()\n"
ARITY = {"count": 1, "can_craft": 1, "craft": 1, "mine": 1, "foundation": 0,
         "place_table": 0, "pickup": 0, "explore": 0, "escape": 0,
         "health": 0, "food": 0, "table_reachable": 0, "stuck": 0, "nearby": 1, "drops": 1,
         "failures": 2, "successes": 2, "pickup_item": 1, "pillar": 1,
         "can_pillar": 0, "can_pillar_up": 0,
         "jump": 0, "run": 0, "sprint": 0, "sneak": 0,
         "smelt": 1, "can_smelt": 1, "furnace_reachable": 0, "place_furnace": 0,
         "chest": 1, "chest_reachable": 0, "place_chest": 0, "free_slots": 0,
         "sleep": 0, "can_sleep": 0, "is_night": 0, "bed_reachable": 0,
         "trade": 1, "can_trade": 1,
         "boat": 0, "in_water": 0,
         "tunnel": 0, "bridge": 0, "shelter": 0, "portal": 0}
HELPERS = ", ".join(f"{name}({', '.join(['a', 'b'][:arity])})" for name, arity in ARITY.items())
# Plausible-looking IDs that do not exist in Minecraft; they silently evaluate to 0 / False
# and turn guards into always-true loops (e.g. `if count("minecraft:logs") > 0: return None`).
FAKE_IDS = {"minecraft:logs": "#logs", "minecraft:log": "#logs", "minecraft:wood": "#logs",
            "minecraft:planks": "#planks", "minecraft:plank": "#planks",
            "minecraft:sticks": "minecraft:stick", "minecraft:pickaxes": "#pickaxes",
            "minecraft:pickaxe": "#pickaxes", "minecraft:cobble": "minecraft:cobblestone"}
# Python constructs small models commonly write, with an actionable hint.
HINTS = {
    "For": "loops are not allowed; a skill decides ONE action per call and is called again next turn",
    "While": "loops are not allowed; a skill decides ONE action per call and is called again next turn",
    "Import": "imports are not allowed", "ImportFrom": "imports are not allowed",
    "FunctionDef": "only the single def skill() is allowed; inline helper logic instead",
    "Try": "try/except is not allowed", "With": "with blocks are not allowed",
    "Subscript": "indexing like x[0] or inventory[...] is not allowed; use count(\"minecraft:item\")",
    "Attribute": "attributes like x.y are not allowed; use the helper functions",
    "JoinedStr": "f-strings are not allowed; use plain string literals",
    "List": "lists are not allowed", "Dict": "dicts are not allowed", "Tuple": "tuples are not allowed",
    "Lambda": "lambda is not allowed", "ListComp": "comprehensions are not allowed",
    "GeneratorExp": "comprehensions are not allowed",
}


def extract_description(code):
    """Extract a one-line description from docstring, leading comment, or return statement."""
    if not isinstance(code, str):
        return "Skill program"
    # Check docstring inside def skill():
    m = re.search(r'def\s+skill\s*\(\s*\)\s*:\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', code, re.S)
    if m:
        line = m.group(1).strip().splitlines()[0].strip()
        if line:
            return line[:120]
    # Check leading comments
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if comment:
                return comment[:120]
        elif line.startswith("def skill"):
            continue
        elif line:
            break
    # Fallback to action in return
    m_ret = re.search(r'return\s+([a-zA-Z0-9_]+)\s*\((.*?)\)', code)
    if m_ret:
        helper, args = m_ret.group(1), m_ret.group(2).strip()
        args = re.sub(r'["\']', '', args)
        if args:
            return f"{helper.capitalize()} {args}"[:120]
        return f"{helper.capitalize()}"[:120]
    return "Survival skill program"


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


EMBEDDING_MODELS = ("qwen3-embedding:0.6b", "nomic-embed-text-v2-moe:latest", "nomic-embed-text", "all-minilm")
_EMBED_PROBE_TTL = 600.0
_EMBED_CACHE_MAX = 512
_embed_state = {"model": None, "checked": 0.0, "cache": {}, "lock": threading.Lock()}


def _embedding_model(ollama, timeout=1.5):
    """Installed embedding model name, or None. Probed via /api/tags at most every 10 minutes so a
    missing model never costs an HTTP round-trip per skill per decision."""
    now = time.monotonic()
    with _embed_state["lock"]:
        if now - _embed_state["checked"] < _EMBED_PROBE_TTL:
            return _embed_state["model"]
        _embed_state["checked"] = now
    found = None
    try:
        with urllib.request.urlopen(f"{ollama}/api/tags", timeout=timeout) as resp:
            names = {str(m.get("name") or m.get("model") or "").lower()
                     for m in json.loads(resp.read(1_048_576)).get("models", [])}
        for candidate in EMBEDDING_MODELS:
            base = candidate.split(":")[0]
            if candidate in names or any(n == base or n.startswith(base + ":") for n in names):
                found = candidate
                break
    except Exception:
        found = None
    with _embed_state["lock"]:
        _embed_state["model"] = found
    return found


def _get_embedding(text, ollama="http://127.0.0.1:11434", timeout=2):
    """Embedding vector from the locally installed embedding model, memoized per text, or None."""
    model = _embedding_model(ollama)
    if not model:
        return None
    key = (model, text[:512])
    with _embed_state["lock"]:
        cached = _embed_state["cache"].get(key)
    if cached is not None:
        return cached
    try:
        req_data = json.dumps({"model": model, "input": text[:512], "keep_alive": "10m"}).encode("utf-8")
        req = urllib.request.Request(f"{ollama}/api/embed", data=req_data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read(262_144))
        vectors = data.get("embeddings")
        vector = vectors[0] if isinstance(vectors, list) and vectors and isinstance(vectors[0], list) else None
    except Exception:
        vector = None
    if vector is None:
        return None
    with _embed_state["lock"]:
        if len(_embed_state["cache"]) >= _EMBED_CACHE_MAX:
            _embed_state["cache"].clear()
        _embed_state["cache"][key] = vector
    return vector


def _bm25_score(query_tokens, doc_tokens, avg_doc_len=15.0):
    """Simple BM25/keyword overlap scoring for skill retrieval."""
    if not query_tokens or not doc_tokens:
        return 0.0
    k1 = 1.2
    b = 0.75
    doc_len = len(doc_tokens)
    score = 0.0
    doc_counts = {}
    for t in doc_tokens:
        doc_counts[t] = doc_counts.get(t, 0) + 1
    for q in query_tokens:
        if q in doc_counts:
            tf = doc_counts[q]
            # TF weighting component of BM25
            tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            score += tf_comp
    return score


def rank_skills(skills, observation, query="", top_k=5, ollama="http://127.0.0.1:11434"):
    """Rank skills by relevance to observation/query using local embedding or BM25 keyword fallback."""
    if not skills:
        return []
    if len(skills) <= top_k:
        return skills

    # Construct query representation
    obs_parts = []
    if query:
        obs_parts.append(str(query))
    if isinstance(observation, dict):
        if observation.get("inventory"):
            obs_parts.append("inventory: " + " ".join(str(i) for i in observation["inventory"][:8]))
        if observation.get("craftable"):
            obs_parts.append("craftable: " + " ".join(str(c) for c in observation["craftable"][:8]))
        if observation.get("nearby_blocks"):
            blocks = [b.get("id", "") for b in observation["nearby_blocks"] if isinstance(b, dict)]
            obs_parts.append("nearby: " + " ".join(blocks[:8]))
        if observation.get("chat"):
            obs_parts.append("chat: " + " ".join(str(c) for c in observation["chat"]))
    query_text = " ".join(obs_parts) if obs_parts else "gather survive craft mine"

    # Try embedding first
    q_emb = _get_embedding(query_text, ollama=ollama, timeout=1.5)
    scored = []
    if q_emb is not None:
        for s in skills:
            desc = s.get("description") or extract_description(s.get("code", ""))
            s_emb = _get_embedding(desc, ollama=ollama, timeout=1.0)
            if s_emb:
                sim = _cosine_similarity(q_emb, s_emb)
                scored.append((sim, s))
            else:
                scored.append((0.0, s))
        if any(sc[0] > 0 for sc in scored):
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored[:top_k]]

    # Fallback: tokenized keyword overlap (BM25)
    q_tokens = [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', query_text) if len(w) > 2]
    for s in skills:
        desc = s.get("description") or extract_description(s.get("code", ""))
        doc_text = f"{desc} {s.get('code', '')}"
        doc_tokens = [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', doc_text) if len(w) > 2]
        score = _bm25_score(q_tokens, doc_tokens)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]


EXAMPLE_SKILLS = [
    {
        "description": "Mine cobblestone when wooden pickaxe is owned and stone is nearby",
        "code": textwrap.dedent('''\
            def skill():
                """Mine cobblestone when wooden pickaxe is owned and stone is nearby."""
                if count("#pickaxes") > 0 and count("#cobblestone") < 16 and nearby("minecraft:stone"):
                    return mine("minecraft:stone")
                return None
        '''),
    },
    {
        "description": "Craft wooden planks when logs are available and planks needed",
        "code": textwrap.dedent('''\
            def skill():
                """Craft wooden planks when logs are available and planks needed."""
                if count("#logs") > 0 and count("#planks") < 8:
                    if can_craft("minecraft:oak_planks"):
                        return craft("minecraft:oak_planks")
                return None
        '''),
    },
    {
        "description": "Smelt iron ore in furnace when fuel and raw ore are available",
        "code": textwrap.dedent('''\
            def skill():
                """Smelt iron ore in furnace when fuel and raw ore are available."""
                if count("#raw_ores") > 0 and count("#fuels") > 0 and furnace_reachable():
                    return smelt("minecraft:raw_iron")
                return None
        '''),
    },
]


def retrieve_few_shot_examples(observation, query="", top_k=2, ollama="http://127.0.0.1:11434"):
    """Retrieve 2-3 relevant few-shot example skills for the strategist prompt."""
    return rank_skills(EXAMPLE_SKILLS, observation, query=query, top_k=top_k, ollama=ollama)


def dry_run(code, observations, validate_action_fn=None, allow_bounded_repetition=False):
    """Run candidate skill in interpreter against N past observations.
    
    Accepts only if it never throws and every returned action passes validate_action_fn (if given).
    Returns (True, "Dry run passed") or (False, error_reason).
    """
    try:
        body = parse(code, allow_bounded_repetition=allow_bounded_repetition)
    except Exception as exc:
        return False, f"Validation error during parse: {exc}"

    if not observations:
        return True, "No past observations to dry-run"

    for idx, obs in enumerate(observations):
        try:
            val = interpret(body, obs, allow_bounded_repetition=allow_bounded_repetition)
            if val is not None:
                if isinstance(val, list):
                    if not allow_bounded_repetition:
                        return False, f"Observation {idx}: Skill returned a list of actions without bounded repetition enabled"
                    for a_idx, act in enumerate(val):
                        if not isinstance(act, dict):
                            return False, f"Observation {idx} action {a_idx}: Returned item is not a dict"
                        if validate_action_fn:
                            validate_action_fn(act, obs)
                elif isinstance(val, dict):
                    if validate_action_fn:
                        validate_action_fn(val, obs)
                else:
                    return False, f"Observation {idx}: Skill returned invalid type {type(val).__name__}"
        except Exception as exc:
            return False, f"Observation {idx}: {exc}"

    return True, "Dry run passed"


class SkillError(ValueError):
    """A validation error that names the offending line and how to fix it."""


def fail(node, message):
    line = getattr(node, "lineno", None)
    raise SkillError(f"line {line}: {message}" if line else message)


def normalize(code):
    """Strip markdown fences/prose around a generated skill so harmless formatting is not fatal."""
    if not isinstance(code, str):
        return code
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")
    if "\\t" in text:
        text = text.replace("\\t", "    ")
    text = text.replace("\t", "    ")
    fenced = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif "```" in text:
        fenced = re.search(r"```(?:python|py)?[ \t]*(.*?)```", text, re.S)
        if fenced:
            text = fenced.group(1)
    start = text.find("def skill")
    if start > 0:
        text = text[start:]
    return textwrap.dedent(text).strip() + "\n" if text.strip() else ""


def parse(code, allow_bounded_repetition=False):
    if not isinstance(code, str) or len(code.encode()) > 4096:
        raise SkillError("Skill source exceeds 4096 bytes")
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError) as exc:
        detail = f"line {exc.lineno}: {exc.msg}" if isinstance(exc, SyntaxError) and exc.lineno else "unparseable"
        raise SkillError(f"Invalid skill syntax ({detail}); return plain Python source, no markdown") from exc
    max_nodes = 384 if not allow_bounded_repetition else 256
    if len(list(ast.walk(tree))) > max_nodes:
        raise SkillError(f"Skill is too large (over {max_nodes} syntax nodes); keep it short and focused")
    if len(tree.body) != 1:
        raise SkillError("Skill must be exactly one top-level def skill(): with nothing before or after it")
    function = tree.body[0]
    if (not isinstance(function, ast.FunctionDef) or function.name != "skill"
            or function.decorator_list or function.returns or getattr(function, "type_params", [])
            or any(ast.iter_child_nodes(function.args))):
        fail(function, "Expected def skill() without arguments, annotations or decorators")

    names = set()

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id in names:
                return
            if node.id in ARITY:
                fail(node, f"{node.id} is a helper; call it as {node.id}(...)")
            fail(node, f"unknown name '{node.id}'; assign a local variable before using it")
        if isinstance(node, ast.Constant):
            if node.value is None or type(node.value) is bool:
                return
            if type(node.value) is int and abs(node.value) <= 100000:
                return
            if isinstance(node.value, str) and len(node.value) <= 96:
                if node.value in FAKE_IDS:
                    fail(node, f"'{node.value}' is not a real item ID (it always counts 0); "
                               f"use '{FAKE_IDS[node.value]}'")
                return
            fail(node, "only None, True/False, small integers and short strings are allowed as literals")
        elif isinstance(node, ast.List) and allow_bounded_repetition:
            if len(node.elts) > 4:
                fail(node, "action lists under bounded repetition cannot exceed 4 items")
            for elt in node.elts:
                expression(elt)
            return
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ARITY:
                called = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
                fail(node, f"unknown function '{called}'; available helpers: {HELPERS}")
            if node.keywords or len(node.args) != ARITY[node.func.id]:
                fail(node, f"{node.func.id} takes exactly {ARITY[node.func.id]} positional argument(s)")
            for arg in node.args:
                expression(arg)
            return
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                expression(value)
            return
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            expression(node.operand)
            return
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            expression(node.left)
            expression(node.right)
            return
        elif isinstance(node, ast.IfExp):
            expression(node.test)
            expression(node.body)
            expression(node.orelse)
            return
        elif isinstance(node, ast.Compare):
            for op, value in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if not (isinstance(value, ast.Constant) and value.value is None):
                        fail(node, "'is' / 'is not' may only compare with None")
                elif not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    fail(node, "only == != < <= > >= and 'is None' comparisons are allowed ('in' is not)")
            expression(node.left)
            for value in node.comparators:
                expression(value)
            return
        kind = type(node).__name__
        if isinstance(node, ast.BinOp):
            fail(node, "only integer + and - arithmetic is allowed")
        fail(node, f"unsupported expression ({kind}): {HINTS.get(kind, 'not part of the skill language')}")

    def local(node, target):
        if not isinstance(target, ast.Name) or target.id.startswith("_") or target.id in ARITY:
            fail(node, "assignments must target a simple local variable name (not a helper name)")
        names.add(target.id)
        if len(names) > 16:
            fail(node, "too many local skill variables (max 16)")

    def statements(body):
        for node in body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    fail(node, "chained or multiple assignment is not allowed")
                expression(node.value)
                local(node, node.targets[0])
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.op, (ast.Add, ast.Sub)):
                    fail(node, "only += and -= are allowed")
                if not isinstance(node.target, ast.Name) or node.target.id not in names:
                    fail(node, "+= / -= needs a local variable assigned earlier")
                expression(node.value)
            elif isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fail(node, "a bare call does nothing; write 'return helper(...)' to choose an action")
            elif isinstance(node, ast.If):
                expression(node.test)
                statements(node.body)
                statements(node.orelse)
            elif isinstance(node, ast.For) and allow_bounded_repetition:
                # Bounded repetition construct: for _ in range(N): where N <= 4
                if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                        and node.iter.func.id == "range" and len(node.iter.args) == 1
                        and isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[0].value, int)
                        and 1 <= node.iter.args[0].value <= 4):
                    fail(node, "bounded repetition allows only 'for _ in range(N):' with integer N between 1 and 4")
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                statements(node.body)
            elif isinstance(node, ast.Return):
                if node.value is not None:
                    expression(node.value)
            else:
                kind = type(node).__name__
                fail(node, f"'{kind}' statement not allowed: "
                           f"{HINTS.get(kind, 'skills allow only local assignments, if/elif/else, pass and return')}")

    statements(function.body)
    return function.body


def action(name, target="", position=(0, 0, 0)):
    return dict(action=name, target=target, x=position[0], y=position[1], z=position[2],
                goal="Establish survival tools", say="", memory="")


def interpret(body, observation, allow_bounded_repetition=False):
    inventory = {}
    for entry in observation.get("inventory", []):
        name, count = entry.rsplit("=", 1)
        inventory[name] = inventory.get(name, 0) + int(count)
    crafting = observation.get("crafting", {})
    variables = {}

    def call(name, args):
        if any(not isinstance(arg, str) or len(arg) > 96 for arg in args):
            raise ValueError("Skill helper arguments must resolve to short strings")
        if name == "count":
            tag = args[0]
            if tag in ("#logs", "#planks", "#pickaxes", "#cobblestone", "#swords", "#axes", "#shovels"):
                if tag[1:] in crafting:
                    return crafting.get(tag[1:], 0)
            if tag == "#logs":
                return sum(v for k, v in inventory.items() if k.endswith(("_log", "_stem")))
            if tag == "#planks":
                return sum(v for k, v in inventory.items() if k.endswith("_planks"))
            if tag == "#pickaxes":
                return sum(v for k, v in inventory.items() if k.endswith("_pickaxe"))
            if tag == "#cobblestone":
                return inventory.get("minecraft:cobblestone", 0) + inventory.get("minecraft:cobbled_deepslate", 0)
            if tag == "#swords":
                return sum(v for k, v in inventory.items() if k.endswith("_sword"))
            if tag == "#axes":
                return sum(v for k, v in inventory.items() if k.endswith("_axe"))
            if tag == "#shovels":
                return sum(v for k, v in inventory.items() if k.endswith("_shovel"))
            if tag == "#beds":
                return sum(v for k, v in inventory.items() if k.endswith("_bed"))
            if tag == "#boats":
                return sum(v for k, v in inventory.items() if k.endswith(("_boat", "_chest_boat", "_raft", "_chest_raft")))
            if tag == "#raw_ores":
                return sum(v for k, v in inventory.items() if k in (
                    "minecraft:raw_iron", "minecraft:raw_gold", "minecraft:raw_copper",
                    "minecraft:iron_ore", "minecraft:deepslate_iron_ore",
                    "minecraft:gold_ore", "minecraft:deepslate_gold_ore",
                    "minecraft:copper_ore", "minecraft:deepslate_copper_ore"))
            if tag == "#ingots":
                return sum(v for k, v in inventory.items() if k.endswith("_ingot"))
            if tag == "#fuels":
                return sum(v for k, v in inventory.items() if k in (
                    "minecraft:coal", "minecraft:charcoal", "minecraft:coal_block",
                    "minecraft:lava_bucket", "minecraft:blaze_rod", "minecraft:stick")
                    or k.endswith(("_log", "_stem", "_planks")))
            return inventory.get(tag, 0)
        if name == "can_craft":
            return args[0] in observation.get("craftable", [])
        if name == "can_smelt":
            smeltable = observation.get("smeltable", [])
            return bool(smeltable) if not args[0] else args[0] in smeltable
        if name == "can_trade":
            trades = observation.get("trades", [])
            return bool(trades) if not args[0] else args[0] in trades
        if name == "nearby":
            return any(block.get("id") == args[0] for block in observation.get("nearby_blocks", []) if isinstance(block, dict))
        if name == "drops":
            return any(entity.get("item") == args[0] for entity in observation.get("entities", []) if isinstance(entity, dict))
        if name in ("failures", "successes"):
            return sum(row[name] for row in observation.get("lookup", [])
                       if row["action"] == args[0] and row["target"] == args[1])
        if name == "table_reachable":
            return bool(crafting.get("table_reachable", False))
        if name == "furnace_reachable":
            return bool(crafting.get("furnace_reachable", False) or any(
                isinstance(b, dict) and b.get("id") in ("minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker")
                for b in observation.get("nearby_blocks", [])))
        if name == "chest_reachable":
            return bool(crafting.get("chest_reachable", False) or any(
                isinstance(b, dict) and b.get("id") in ("minecraft:chest", "minecraft:barrel", "minecraft:trapped_chest")
                for b in observation.get("nearby_blocks", [])))
        if name == "bed_reachable":
            return bool(crafting.get("bed_reachable", False) or any(
                isinstance(b, dict) and str(b.get("id", "")).endswith("_bed")
                for b in observation.get("nearby_blocks", [])))
        if name == "is_night":
            t = int(observation.get("time", 0) or 0)
            return (12542 <= t <= 23458) or observation.get("weather") == "thunder"
        if name == "can_sleep":
            t = int(observation.get("time", 0) or 0)
            night = (12542 <= t <= 23458) or observation.get("weather") == "thunder"
            has_bed = bool(crafting.get("bed_reachable", False)
                           or any(isinstance(b, dict) and str(b.get("id", "")).endswith("_bed") for b in observation.get("nearby_blocks", []))
                           or any(k.endswith("_bed") and v > 0 for k, v in inventory.items()))
            return night and has_bed
        if name == "in_water":
            return bool(observation.get("in_water", False))
        if name == "free_slots":
            return int(observation.get("free_slots", 36))
        if name == "stuck":
            return observation.get("diagnosis", {}).get("stuck", False)
        if name in ("food", "health"):
            return observation.get(name, 20)
        if name == "foundation":
            step = crafting.get("next_step")
            if step:
                return action(step["action"], step["target"], (step["x"], step["y"], step["z"]))
            if crafting and crafting.get("pickaxes", 0) == 0:
                for entity in observation.get("entities", []):
                    if not isinstance(entity, dict):
                        continue
                    item = entity.get("item", "")
                    if item.endswith(("_log", "_stem", "_planks", ":stick", ":crafting_table")):
                        return action("pickup", item)
            if observation.get("smeltable") and (crafting.get("furnace_reachable") or any(
                    isinstance(b, dict) and b.get("id") in ("minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker")
                    for b in observation.get("nearby_blocks", []))):
                return action("smelt", observation["smeltable"][0])
            return None
        if name == "place_table":
            sites = crafting.get("placement_sites", [])
            pos = observation.get("position")
            ground_sites = [s for s in sites if s != pos]
            target_site = ground_sites[0] if ground_sites else (sites[0] if sites else None)
            if target_site and inventory.get("minecraft:crafting_table", 0):
                return action("place", "minecraft:crafting_table", target_site)
            return None
        if name == "place_furnace":
            sites = crafting.get("placement_sites", [])
            pos = observation.get("position")
            ground_sites = [s for s in sites if s != pos]
            target_site = ground_sites[0] if ground_sites else (sites[0] if sites else None)
            for f_item in ("minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker"):
                if target_site and inventory.get(f_item, 0) > 0:
                    return action("place", f_item, target_site)
            return None
        if name == "place_chest":
            sites = crafting.get("placement_sites", [])
            pos = observation.get("position")
            ground_sites = [s for s in sites if s != pos]
            target_site = ground_sites[0] if ground_sites else (sites[0] if sites else None)
            for c_item in ("minecraft:chest", "minecraft:barrel"):
                if target_site and inventory.get(c_item, 0) > 0:
                    return action("place", c_item, target_site)
            return None
        if name in ("can_pillar", "can_pillar_up"):
            pos = observation.get("position")
            sites = crafting.get("placement_sites", [])
            placeable = observation.get("placeable", [])
            return bool(pos and pos in sites and any(item in placeable and count > 0 for item, count in inventory.items()))
        if name == "pillar":
            item = args[0] if args else ""
            pos = observation.get("position")
            sites = crafting.get("placement_sites", [])
            placeable = observation.get("placeable", [])
            if not item:
                for p in placeable:
                    if inventory.get(p, 0) > 0:
                        item = p
                        break
            if pos and pos in sites and inventory.get(item, 0) > 0 and item in placeable:
                return action("place", item, pos)
            return None
        if name in ("jump", "run", "sprint", "sneak", "sleep", "boat", "tunnel", "bridge", "shelter", "portal"):
            return action(name)
        if name == "pickup_item":
            return action("pickup", args[0])
        return action(name, args[0] if args else "")

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError("Uninitialized skill variable: " + node.id)
            return variables[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List) and allow_bounded_repetition:
            return [expression(elt) for elt in node.elts]
        if isinstance(node, ast.Call):
            return call(node.func.id, [expression(arg) for arg in node.args])
        if isinstance(node, ast.BoolOp):
            # Python semantics: return the deciding operand, so `foundation() or mine(...)` is an action.
            value = None
            for operand in node.values:
                value = expression(operand)
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.IfExp):
            return expression(node.body) if expression(node.test) else expression(node.orelse)
        if isinstance(node, ast.UnaryOp):
            value = expression(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) is not int:
                raise ValueError("Skill negation requires an integer")
            return -value
        if isinstance(node, ast.BinOp):
            left, right = expression(node.left), expression(node.right)
            if type(left) is not int or type(right) is not int:
                raise ValueError("Skill arithmetic requires integers")
            value = left + right if isinstance(node.op, ast.Add) else left - right
            if abs(value) > 100000:
                raise ValueError("Skill arithmetic exceeds limit")
            return value
        left = expression(node.left)
        for operator, right_node in zip(node.ops, node.comparators):
            right = expression(right_node)
            if isinstance(operator, (ast.Is, ast.IsNot)):
                if (left is None) != isinstance(operator, ast.Is):
                    return False
                left = right
                continue
            if type(left) not in (int, bool, str) or type(right) is not type(left):
                raise ValueError("Skill comparisons require matching scalar types")
            operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                          ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                          ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}
            if not operations[type(operator)]():
                return False
            left = right
        return True

    def statements(nodes):
        for node in nodes:
            if isinstance(node, ast.Assign):
                variables[node.targets[0].id] = expression(node.value)
                continue
            if isinstance(node, ast.AugAssign):
                current, delta = variables.get(node.target.id), expression(node.value)
                if type(current) is not int or type(delta) is not int:
                    raise ValueError("Skill += / -= requires integers")
                value = current + delta if isinstance(node.op, ast.Add) else current - delta
                if abs(value) > 100000:
                    raise ValueError("Skill arithmetic exceeds limit")
                variables[node.target.id] = value
                continue
            if isinstance(node, (ast.Pass, ast.Expr)):
                continue
            if isinstance(node, ast.For) and allow_bounded_repetition:
                count = expression(node.iter.args[0])
                actions = []
                for idx in range(min(4, max(1, count))):
                    if isinstance(node.target, ast.Name):
                        variables[node.target.id] = idx
                    returned, val = statements(node.body)
                    if returned:
                        if isinstance(val, list):
                            actions.extend(val)
                        elif val is not None:
                            actions.append(val)
                        if len(actions) >= 4:
                            return True, actions[:4]
                if actions:
                    return True, actions[:4]
                continue
            if isinstance(node, ast.Return):
                val = expression(node.value) if node.value else None
                return True, val
            returned, value = statements(node.body if expression(node.test) else node.orelse)
            if returned:
                return True, value
        return False, None

    return statements(body)[1]


class SkillLibrary:
    def __init__(self, db, directory):
        self.db = db
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS skills
              (id TEXT PRIMARY KEY, code TEXT NOT NULL, created REAL, successes INTEGER DEFAULT 0,
               failures INTEGER DEFAULT 0, disabled INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS decision_skills (decision_id TEXT PRIMARY KEY, skill_id TEXT);
        """)
        # Backward-compatible migrations for existing brain.sqlite3
        cursor = db.execute("PRAGMA table_info(skills)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "description" not in existing_cols:
            try:
                db.execute("ALTER TABLE skills ADD COLUMN description TEXT DEFAULT ''")
            except Exception:
                pass
        if "status" not in existing_cols:
            try:
                db.execute("ALTER TABLE skills ADD COLUMN status TEXT DEFAULT 'probation'")
            except Exception:
                pass
        if "dry_run_passed" not in existing_cols:
            try:
                db.execute("ALTER TABLE skills ADD COLUMN dry_run_passed INTEGER DEFAULT 1")
            except Exception:
                pass
        # Ensure seed skill has status='active' and is never left disabled by older versions
        db.execute("UPDATE skills SET status='active', disabled=0 WHERE created=0")
        db.commit()
        self._parsed = {}

        if not db.execute("SELECT 1 FROM skills LIMIT 1").fetchone():
            self.install(SEED, created=0, description="Default foundation gathering and survival skill")

    def install(self, code, created=None, description=None, dry_run_passed=1, allow_bounded_repetition=False):
        parse(code, allow_bounded_repetition=allow_bounded_repetition)
        key = hashlib.sha256(code.encode()).hexdigest()[:16]
        desc = (description or extract_description(code))[:120]
        status = "active" if created == 0 else "probation"
        if self.db.execute("SELECT 1 FROM skills WHERE id=?", (key,)).fetchone():
            return key
        path = self.directory / (key + ".py")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(code, encoding="utf-8")
        temporary.replace(path)
        with self.db:
            self.db.execute(
                "INSERT INTO skills (id,code,created,description,status,dry_run_passed) VALUES (?,?,?,?,?,?)",
                (key, code, time.time() if created is None else created, desc, status, int(bool(dry_run_passed)))
            )
            old = self.db.execute("SELECT id FROM skills WHERE created>0 ORDER BY created DESC LIMIT -1 OFFSET 15").fetchall()
            for (expired,) in old:
                (self.directory / (expired + ".py")).unlink(missing_ok=True)
                self.db.execute("DELETE FROM skills WHERE id=?", (expired,))
        LOG.info("Skill installed (%s): %s - %s", status, path, desc)
        return key

    def candidates(self, observation, allow_bounded_repetition=False):
        rows = self.db.execute("SELECT id,code,status,successes FROM skills WHERE disabled=0 "
                               "ORDER BY CASE WHEN created=0 THEN 0 ELSE 1 END, created DESC").fetchall()
        for key, code, status, successes in rows:
            try:
                cache_key = (key, bool(allow_bounded_repetition))
                body = self._parsed.get(cache_key)
                if body is None:
                    body = parse(code, allow_bounded_repetition=allow_bounded_repetition)
                    if len(self._parsed) > 64:
                        self._parsed.clear()
                    self._parsed[cache_key] = body
                value = interpret(body, observation, allow_bounded_repetition=allow_bounded_repetition)
                if value is not None:
                    if not isinstance(value, (dict, list)):
                        raise ValueError("Skill must return an action, a list of actions, or None")
                    yield key, value
            except (ValueError, TypeError, KeyError) as exc:
                self.reject(key, str(exc))

    def reject(self, key, reason):
        if self.is_seed(key):
            # The built-in foundation routine is trusted code; a transient observation mismatch skips it
            # for this turn instead of silently disabling bootstrap crafting forever.
            LOG.warning("Foundation skill %s skipped this turn: %s", key, reason)
            return
        LOG.error("Disabling invalid skill %s: %s", key, reason)
        with self.db:
            self.db.execute("UPDATE skills SET disabled=1, status='disabled' WHERE id=?", (key,))

    def is_seed(self, key):
        row = self.db.execute("SELECT created FROM skills WHERE id=?", (key,)).fetchone()
        return bool(row) and row[0] == 0

    def outcome(self, decision_id, success):
        """success may be True, False, or None (interrupted: neither counted)."""
        if success is None:
            return
        row = self.db.execute("SELECT skill_id FROM decision_skills WHERE decision_id=?", (decision_id,)).fetchone()
        if not row:
            return
        key = row[0]
        field = "successes" if success else "failures"
        self.db.execute(f"UPDATE skills SET {field}={field}+1 WHERE id=?", (key,))
        # Three failures disable a generated version; the seed stays available (its targets are
        # cooled down per situation by the action lookup instead).
        self.db.execute("UPDATE skills SET disabled=1, status='disabled' WHERE id=? AND failures>=3 AND created>0", (key,))
        # Task B3 Probation promotion: promote to 'active' after 2 observed successes
        if success:
            self.db.execute("UPDATE skills SET status='active' WHERE id=? AND successes>=2 AND status='probation'", (key,))
        LOG.info("Skill outcome: %s success=%s (three failures disable; two successes promote)", key, success)

    def report(self, observation=None, query="", top_k=4, ollama="http://127.0.0.1:11434"):
        all_skills = [
            dict(id=row[0], code=row[1], successes=row[2], failures=row[3], disabled=bool(row[4]),
                 description=row[5] or extract_description(row[1]), status=row[6] or "active")
            for row in self.db.execute(
                "SELECT id,code,successes,failures,disabled,description,status FROM skills ORDER BY created DESC"
            )
        ]
        if not observation:
            return all_skills[:top_k]
        return rank_skills(all_skills, observation, query=query, top_k=top_k, ollama=ollama)
