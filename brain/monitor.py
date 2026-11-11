#!/usr/bin/env python3
"""Willow AI / Ollama live telemetry console: HUD dashboard, token speed sparklines, vitals & structured brain stream.

    python3 monitor.py --log "<world>/localai/brain.log"
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[78]")
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Log parsing patterns
LOG_PREFIX_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})(?:,\d+)?\s+"
    r"(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s+(.*)$"
)
ROLE_RE = re.compile(
    r"^role=(\w+)\s+prompt_tokens=(\S+)(?:\s+cached=(\S+))?\s+generated_tokens=(\S+)\s+elapsed=(\S+?)s"
    r"(?:\s+prompt_tps=(\S+)\s+gen_tps=(\S+))?"
)
STRATA_ROLE_RE = re.compile(r"^role=(\w+)\s+STRATA\s+elapsed=(\S+?)s")
DECISION_RE = re.compile(
    r"^decision=(\S+)\s+action=(\S+)\s+target=(.*?)\s+source=(\S+)\s+elapsed=(\S+?)s?$"
)
CRITIC_RE = re.compile(
    r"^\[CRITIC\]\s+(PASS|FAIL)\s+task=(\S+)\s+category=(\S+)\s*\(([^)]*)\)"
)
REWARD_RE = re.compile(
    r"^\[RL-REWARD\]\s+task=(\S+)\s+total=([+-]?\d+(?:\.\d+)?)\s*\(([^)]*)\)"
)
CURRIC_SELECT_RE = re.compile(
    r"^\[CURRICULUM\]\s+Selected objective '([^']+)'\s*\(Tier\s*(\d+),\s*Reward\s*([+-]?\d+(?:\.\d+)?),\s*PrecondOK=(\w+)\)"
)
CURRIC_RECOV_RE = re.compile(
    r"^\[CURRICULUM\]\s+Precondition recovery scheduled:\s*'([^']+)'\s*->\s*'([^']+)'\s*for frontier\s*'([^']+)'"
)
CURRIC_REGRESS_RE = re.compile(
    r"^\[CURRICULUM\]\s+Regression detected!\s*Persistent milestone '([^']+)' lost\s*\(postcondition=([^)]+)\)"
)
CURRIC_SPLIT_RE = re.compile(
    r"^\[CURRICULUM\]\s+(?:LLM|Deterministically) decomposed '([^']+)' into (\d+) micro-tasks:\s*(.+)$"
)
QFAST_RE = re.compile(
    r"^\[Q-FASTPATH\]\s+Bypassing LLM!\s+Context=(\S+)\s+Task=(\S+)\s+Skill=(\S+)\s+Q=([+-]?\d+(?:\.\d+)?)"
)
VRAM_INFER_RE = re.compile(
    r"^\[VRAM-OPT\]\s+model=(\S+)\s+prio=(\S+)\s+ctx=(\d+)\s+predict=(\d+)\s+threads=(\d+)\s+"
    r"kv_prefix_hit=(\S+)\s+ttft=([\d.]+)ms\s+total=([\d.]+)ms"
)
STRAT_GOAL_RE = re.compile(r"^Strategist goal=(.*?)\s+reason=(.*)$")
AUTO_SELECT_RE = re.compile(
    r"^Mine coordinate (\[[^\]]+\]) is no longer observed; using automatic (\S+) selection$"
)
REDIRECT_WALL_RE = re.compile(r"^Mine (\S+) at (\S+) redirected to the exit wall at (\S+)$")
REDIRECT_VIS_RE = re.compile(r"^Redirected (\S+) to mine visible (?:tree log )?(\S+) at (\[[^\]]+\])")
OUT_OF_RANGE_RE = re.compile(r"^(\S+) (\S+) is outside the (\d+)-block range; heading toward it$")
CHAT_IN_RE = re.compile(r"^Processing observation with chat:\s*(.+)$")
CHAT_OUT_RE = re.compile(r'^Willow chat reply to player:\s*"(.*)"$')
LISTENING_RE = re.compile(r"^Local brain listening on (\S+),\s*model=(\S+)$")
AUTOTUNE_RE = re.compile(
    r"^\[AUTOTUNE\].*?gpu=([^(]+)\((\d+)MiB.*?model=(\S+)\s+ctx=(\d+)\(trained=(\d+).*?"
    r"speed=\(prompt_tps=([\d.]+),\s*gen_tps=([\d.]+)"
)
MODEL_PROFILE_RE = re.compile(
    r"^Model profile model=(\S+)\s+arch=(\S+)\s+params=(\S+)\s+quant=(\S+)\s+trained_ctx=(\S+)\s+ctx=(\S+)\s+kv_cache=(\S+)"
)

LOCK = threading.Lock()


class Palette:
    """Explicit 256-color dark telemetry theme so host terminal default foreground (e.g. Konsole green) never bleeds."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        if self.enabled:
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.BG_HUD = "\033[48;5;235m"
            self.FG_BORDER = "\033[38;5;240m"
            self.FG_HUD_BORDER = "\033[38;5;60m"
            self.FG_DIM = "\033[38;5;242m"
            self.FG_MUTED = "\033[38;5;246m"
            self.FG_TEXT = "\033[38;5;252m"
            self.FG_BRIGHT = "\033[1;38;5;255m"
            self.FG_BRAND = "\033[1;38;5;81m"
            self.FG_CYAN = "\033[38;5;81m"
            self.FG_SKY = "\033[38;5;117m"
            self.FG_TEAL = "\033[38;5;79m"
            self.FG_GREEN = "\033[38;5;83m"
            self.FG_GREEN_BOLD = "\033[1;38;5;83m"
            self.FG_VIOLET = "\033[38;5;141m"
            self.FG_VIOLET_BOLD = "\033[1;38;5;141m"
            self.FG_GOLD = "\033[38;5;221m"
            self.FG_GOLD_BOLD = "\033[1;38;5;221m"
            self.FG_AMBER = "\033[38;5;215m"
            self.FG_RED = "\033[38;5;203m"
            self.FG_RED_BOLD = "\033[1;38;5;203m"
            self.FG_PINK = "\033[38;5;212m"
            self.FG_SLATE = "\033[38;5;110m"
        else:
            for attr in (
                "RESET", "BOLD", "BG_HUD", "FG_BORDER", "FG_HUD_BORDER", "FG_DIM", "FG_MUTED",
                "FG_TEXT", "FG_BRIGHT", "FG_BRAND", "FG_CYAN", "FG_SKY", "FG_TEAL", "FG_GREEN",
                "FG_GREEN_BOLD", "FG_VIOLET", "FG_VIOLET_BOLD", "FG_GOLD", "FG_GOLD_BOLD",
                "FG_AMBER", "FG_RED", "FG_RED_BOLD", "FG_PINK", "FG_SLATE",
            ):
                setattr(self, attr, "")

    def paint(self, text, *codes, bg=""):
        if not self.enabled or not text:
            return str(text)
        prefix = bg + "".join(codes)
        return f"{prefix}{text}{self.RESET}"


def visible_len(text):
    return len(ANSI_ESCAPE_RE.sub("", text))


def truncate_ansi(text, max_chars, ellipsis="…"):
    """Truncates an ANSI-styled string to at most `max_chars` visible characters while preserving escapes."""
    if max_chars <= 0:
        return ""
    if visible_len(text) <= max_chars:
        return text
    limit = max(0, max_chars - len(ellipsis))
    out = []
    seen = 0
    pos = 0
    while pos < len(text) and seen < limit:
        match = ANSI_ESCAPE_RE.match(text, pos)
        if match:
            out.append(match.group(0))
            pos = match.end()
            continue
        out.append(text[pos])
        seen += 1
        pos += 1
    if limit < max_chars:
        out.append(ellipsis[: max_chars - limit])
    return "".join(out)


def pad_ansi(text, width, fill=" ", fill_prefix=""):
    trimmed = truncate_ansi(text, width)
    pad = max(0, width - visible_len(trimmed))
    if pad == 0:
        return trimmed
    return trimmed + (f"{fill_prefix}{fill * pad}" if fill_prefix else fill * pad)


def get(path, base_url=DEFAULT_OLLAMA, timeout=2):
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read(1_048_576))


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit in ("B", "KB") else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def short_id(raw):
    if not raw:
        return ""
    text = str(raw).strip()
    if text.startswith("minecraft:"):
        text = text[10:]
    return text


def compact_ctx(tokens):
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return str(tokens or "?")
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    return f"{n:,}"


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m:02d}m {s:02d}s"
    return f"{s}s"


def sparkline(values, width=8):
    vals = [float(v) for v in values if v is not None and v > 0][-width:]
    if not vals:
        return "·" * width
    lo, hi = min(vals), max(vals)
    span = hi - lo
    chars = []
    for v in vals:
        if span < 1.0:
            idx = 4
        else:
            idx = min(7, max(0, int(round(7 * (v - lo) / span))))
        chars.append(SPARK_CHARS[idx])
    return ("·" * max(0, width - len(chars))) + "".join(chars)


def progress_bar(value, maximum, width=6, full="■", empty="·"):
    if maximum <= 0:
        return empty * width
    ratio = max(0.0, min(1.0, float(value) / float(maximum)))
    filled = int(round(ratio * width))
    return (full * filled) + (empty * (width - filled))


def parse_reward_terms(detail_str):
    """Extracts only non-zero reward terms from `(adv=+0.0 task=+0.0 ... | vital=-0.0 ...)`."""
    nonzero = []
    for key, val_str in re.findall(r"(\w+)=([+-]?\d+(?:\.\d+)?)", detail_str or ""):
        try:
            val = float(val_str)
        except ValueError:
            continue
        if abs(val) >= 0.005:
            sign = "+" if val_str.startswith("+") or val > 0 else ""
            formatted = f"{val:.2f}".rstrip("0").rstrip(".")
            if val > 0 and not formatted.startswith("+"):
                formatted = "+" + formatted
            nonzero.append((key, val, f"{key} {formatted if sign != '+' or formatted.startswith('+') else '+' + formatted}"))
    return nonzero


def friendly_category(cat):
    mapping = {
        "NONE": "verified",
        "MISSING_PREREQUISITE": "prereq",
        "INCOMPLETE_QUANTITY": "gathering",
        "CRAFTING_STATION_MISSING": "no station",
        "TOOL_TIER_MISMATCH": "wrong tool",
        "PATHFINDING_TIMEOUT": "path timeout",
        "COMBAT_DEATH": "combat death",
        "SYNTAX_OR_SANDBOX": "sandbox err",
        "RUNTIME_EXCEPTION": "runtime err",
        "AMBIGUOUS_FAILURE": "ambiguous",
        "INTERRUPTED": "interrupted",
    }
    return mapping.get(cat, cat.lower().replace("_", " "))


class Console:
    def __init__(self, log, ollama=DEFAULT_OLLAMA, color=None):
        self.log = Path(log) if log else None
        self.ollama = ollama
        self.tty = sys.stdout.isatty()
        use_color = (self.tty and "NO_COLOR" not in os.environ) if color is None else bool(color)
        self.p = Palette(enabled=use_color)
        self.started_mono = time.monotonic()
        self.started_wall = time.time()

        # Live inference & decision metrics
        self.last = "waiting for Willow's first model call..."
        self.calls = []  # (wall_ts, gen_tps, prompt_tps, elapsed_s, prompt_tok, cached_tok, gen_tok, role)
        self.decisions_count = 0
        self.last_action_summary = ""
        self.last_goal = ""

        # RL & Curriculum live counters
        self.rl_total = 0.0
        self.rl_last = None
        self.critic_passes = 0
        self.critic_steps = 0
        self.critic_fails = 0
        self.q_fast_hits = 0
        self.active_task_id = ""
        self.active_tier = None
        self.completed_tasks = None
        self._last_curric_key = None
        self._last_info_dedup = ""
        self._pending_critic = None  # dict with stamp, passed, task, category, mode, mono

        # Cached external state (Ollama HTTP & status.json)
        self._ollama_cache = {"ts": 0.0, "ok": False, "version": "?", "models": []}
        self.status_lines = []
        self._rendered_header = []
        self.panel = 0
        self._region = None
        self._dirty = True

    def _columns_rows(self):
        cols, rows = shutil.get_terminal_size((110, 34))
        return max(68, cols), max(16, rows)

    def _refresh_ollama_cache(self, force=False):
        now = time.monotonic()
        if not force and (now - self._ollama_cache["ts"]) < 2.5:
            return self._ollama_cache
        base_url = self.ollama
        live_status = self._read_live_status()
        if live_status and live_status.get("ollama"):
            base_url = str(live_status["ollama"])
        try:
            version = get("/api/version", base_url=base_url, timeout=1.5).get("version", "?")
            models = get("/api/ps", base_url=base_url, timeout=1.5).get("models", [])
            self._ollama_cache = {"ts": now, "ok": True, "version": version, "models": models, "url": base_url}
        except (OSError, ValueError):
            self._ollama_cache = {"ts": now, "ok": False, "version": "?", "models": [], "url": base_url}
        return self._ollama_cache

    def _read_profile(self):
        if not self.log:
            return None
        path = self.log.parent / "model-profile.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _read_live_status(self):
        if not self.log:
            return None
        path = self.log.parent / "status.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write(self, line):
        with LOCK:
            sys.stdout.write(line + self.p.RESET + "\n")
            sys.stdout.flush()

    def _hud_row(self, label, label_color, content, columns):
        """Builds a boxed HUD row: `│ LABEL   │ <content padded> │` with uniform dark surface background."""
        p = self.p
        bg = p.BG_HUD
        inner_w = max(20, columns - 14)
        left_border = f"{bg}{p.FG_HUD_BORDER}│"
        lbl = f"{bg} {label_color}{label:<7}{bg} "
        mid_border = f"{bg}{p.FG_HUD_BORDER}│{bg} "
        padded = pad_ansi(content, inner_w, fill=" ", fill_prefix=bg)
        right_border = f"{bg} {p.FG_HUD_BORDER}│{p.RESET}"
        return f"{left_border}{lbl}{mid_border}{padded}{right_border}"

    def _hud_top_border(self, left_content, right_content, columns):
        p = self.p
        bg = p.BG_HUD
        left_vis = visible_len(left_content)
        right_vis = visible_len(right_content)
        # ╭─ <left> ───...─── <right> ─╮
        avail = columns - 4
        if left_vis + right_vis + 2 > avail:
            right_content = ""
            right_vis = 0
        if left_vis > avail:
            left_content = truncate_ansi(left_content, avail)
            left_vis = visible_len(left_content)
        dashes = max(1, avail - left_vis - right_vis)
        if right_vis:
            return (
                f"{bg}{p.FG_HUD_BORDER}╭─{left_content}"
                f"{bg}{p.FG_HUD_BORDER}{'─' * dashes}{right_content}"
                f"{bg}{p.FG_HUD_BORDER}─╮{p.RESET}"
            )
        return (
            f"{bg}{p.FG_HUD_BORDER}╭─{left_content}"
            f"{bg}{p.FG_HUD_BORDER}{'─' * max(0, columns - 3 - left_vis)}╮{p.RESET}"
        )

    def _hud_bottom_border(self, columns):
        """Builds `╰─ TIME ──┼─ SOURCE ─┬─ LIVE BRAIN STREAM ───...─── METRICS ─╯` aligned to Col 11 and Col 22."""
        p = self.p
        bg = p.BG_HUD
        bdr = f"{bg}{p.FG_HUD_BORDER}"
        lbl = f"{bg}{p.FG_MUTED}"
        # Col 1..10: `╰─ TIME ──` (10) -> Col 11: `┼` -> Col 12..21: `─ SOURCE ─` (10) -> Col 22: `┬`
        prefix = f"{bdr}╰─{lbl} TIME {bdr}──┼─{lbl} SOURCE {bdr}─┬─{lbl} LIVE STREAM "
        suffix_plain = " METRICS ─╯"
        used = 22 + len("─ LIVE STREAM ") + len(suffix_plain)
        if columns >= used + 2:
            dashes = columns - used
            return f"{prefix}{bdr}{'─' * dashes}{lbl} METRICS {bdr}─╯{p.RESET}"
        rem = max(0, columns - 23)
        return f"{bdr}╰─{lbl} TIME {bdr}──┼─{lbl} SOURCE {bdr}─┬{'─' * rem}╯{p.RESET}"

    def status(self, columns=None):
        """Builds the 7-row sticky HUD dashboard."""
        p = self.p
        bg = p.BG_HUD
        if columns is None:
            columns, _ = self._columns_rows()
        ollama_info = self._refresh_ollama_cache()
        profile = self._read_profile()
        live = self._read_live_status()
        now = time.time()

        # ── 1. Top Border: Brand + Ollama Status + VRAM + Session Uptime ──
        brand = f"{bg} {p.FG_BRAND}◆ WILLOW AI{bg} "
        if ollama_info["ok"]:
            ver = ollama_info["version"]
            models = ollama_info["models"]
            if models:
                m0 = models[0]
                size, vram = m0.get("size", 0), m0.get("size_vram", 0)
                gpu_pct = f"{100 * vram / size:.0f}% GPU" if size else "GPU"
                extra = f" +{len(models) - 1}" if len(models) > 1 else ""
                backend = (
                    f"{bg}{p.FG_GREEN_BOLD}●{bg} {p.FG_TEXT}Ollama {ver}{bg} "
                    f"{p.FG_DIM}·{bg} {p.FG_CYAN}{human(vram or size)} VRAM{bg} "
                    f"{p.FG_MUTED}({gpu_pct}{extra}){bg} "
                )
            else:
                recent_call = bool(self.calls and (now - self.calls[-1][0]) < 90)
                just_started = (time.monotonic() - self.started_mono) < 25
                if profile and (recent_call or just_started):
                    backend = (
                        f"{bg}{p.FG_GOLD_BOLD}◐{bg} {p.FG_TEXT}Ollama {ver}{bg} "
                        f"{p.FG_AMBER}· warming model into VRAM...{bg} "
                    )
                else:
                    backend = (
                        f"{bg}{p.FG_MUTED}○{bg} {p.FG_TEXT}Ollama {ver}{bg} "
                        f"{p.FG_MUTED}· idle (loads on next turn){bg} "
                    )
        else:
            backend = f"{bg}{p.FG_RED_BOLD}✗ Ollama offline{bg} {p.FG_MUTED}({ollama_info.get('url', self.ollama)}){bg} "

        start_ref = float(live["started_at"]) if (live and live.get("started_at") and (now - live.get("updated_at", 0)) < 600) else self.started_wall
        uptime_str = format_duration(now - start_ref)
        turns_str = f"{self.decisions_count} turns · " if self.decisions_count else ""
        right_top = f"{bg} {p.FG_MUTED}{turns_str}up {uptime_str}{bg} "
        row_top = self._hud_top_border(f"{brand}{p.FG_HUD_BORDER}──{bg} {backend}", right_top, columns)

        # ── 2. MODEL Row ──
        sep = f"  {p.FG_DIM}│{bg}  " if columns >= 104 else f" {p.FG_DIM}·{bg} "
        if profile:
            model_name = profile.get("model") or (live or {}).get("model") or "?"
            arch = profile.get("architecture", "?")
            params = profile.get("parameter_size", "?")
            quant = profile.get("quantization", "?")
            moe = f" MoE {profile['moe_experts']}x" if profile.get("moe_experts") else ""
            ctx_cur = compact_ctx(profile.get("context"))
            ctx_max = compact_ctx(profile.get("trained_context"))
            think_on = bool(profile.get("thinking"))
            think_tag = f"{p.FG_GREEN}think:on" if think_on else f"{p.FG_MUTED}think:off"
            kv_mib = profile.get("kv_cache_mib", "?")
            out_tok = f"{compact_ctx(profile.get('actor_tokens'))}/{compact_ctx(profile.get('plan_tokens'))}"
            vis_tag = f" {p.FG_DIM}·{bg} {p.FG_PINK}vision" if profile.get("vision_enabled") else ""
            model_body = (
                f"{bg}{p.FG_CYAN}{p.BOLD}{model_name}{p.RESET}{bg} "
                f"{p.FG_VIOLET}[{arch} {params} {quant}{moe}]{bg}"
                f"{sep}{p.FG_MUTED}ctx{bg} {p.FG_TEXT}{ctx_cur}/{ctx_max}{bg} "
                f"{p.FG_DIM}·{bg} {p.FG_MUTED}out{bg} {p.FG_TEXT}{out_tok}{bg} "
                f"{p.FG_DIM}·{bg} {p.FG_MUTED}KV{bg} {p.FG_TEXT}~{kv_mib}M{bg} "
                f"{p.FG_DIM}·{bg} {think_tag}{vis_tag}"
            )
        elif ollama_info["models"]:
            m0 = ollama_info["models"][0]
            model_body = (
                f"{bg}{p.FG_CYAN}{p.BOLD}{m0.get('name', '?')}{p.RESET}{bg}"
                f"{sep}{p.FG_MUTED}size {human(m0.get('size', 0))} · ctx {compact_ctx(m0.get('context_length'))}{bg}"
            )
        else:
            model_body = f"{bg}{p.FG_MUTED}No active model profile yet (created automatically on world start){bg}"
        row_model = self._hud_row("MODEL", p.FG_CYAN, model_body, columns)

        # ── 3. INFER Row: Sparkline + Gen/Prompt Tok/s + KV Prefix Cache + Calls/min ──
        recent_60s = [c for c in self.calls if c[0] > now - 60]
        all_speeds = [c[1] for c in self.calls[-12:] if c[1] is not None]
        if self.calls:
            last_c = self.calls[-1]
            _, last_gen_tps, last_prompt_tps, last_elapsed, last_ptok, last_cached, last_gtok, last_role = last_c
            spark = sparkline(all_speeds, width=8)
            min_speeds = [c[1] for c in recent_60s if c[1] is not None]
            avg_gen = f"{sum(min_speeds) / len(min_speeds):.1f}" if min_speeds else (f"{last_gen_tps:.1f}" if last_gen_tps else "-")
            gen_val = f"{last_gen_tps:.1f}" if last_gen_tps is not None else "-"
            gen_color = p.FG_GREEN_BOLD if (last_gen_tps or 0) >= 80 else (p.FG_CYAN if (last_gen_tps or 0) >= 35 else p.FG_AMBER)
            p_tps_str = f"{last_prompt_tps / 1000:.1f}k" if (last_prompt_tps and last_prompt_tps >= 1000) else (
                f"{last_prompt_tps:.0f}" if last_prompt_tps else "-"
            )

            # Rolling KV cache hit ratio across recent calls with valid token counts
            tok_samples = [(c[4], c[5]) for c in self.calls[-20:] if c[4] and c[4] > 0 and c[5] is not None]
            if tok_samples:
                tot_p = sum(x[0] for x in tok_samples)
                tot_c = sum(x[1] for x in tok_samples)
                kv_pct = 100.0 * tot_c / tot_p if tot_p else 0.0
                kv_col = p.FG_TEAL if kv_pct >= 25 else p.FG_MUTED
                cache_seg = f"{p.FG_MUTED}KV{bg} {kv_col}{kv_pct:.0f}%{bg}"
            else:
                cache_seg = f"{p.FG_MUTED}KV --{bg}"

            infer_body = (
                f"{bg}{p.FG_TEAL}{spark}{bg} "
                f"{p.FG_MUTED}Gen{bg} {gen_color}{gen_val}{p.RESET}{bg} {p.FG_MUTED}tok/s (avg {avg_gen}){bg}"
                f"{sep}{p.FG_MUTED}Prompt{bg} {p.FG_TEXT}{p_tps_str}/s{bg} "
                f"{p.FG_DIM}·{bg} {cache_seg}"
                f"{sep}{p.FG_TEXT}{len(recent_60s)}/min{bg} "
                f"{p.FG_DIM}·{bg} {p.FG_VIOLET}{last_role}{bg} {p.FG_TEXT}{last_elapsed:.2f}s{bg}"
            )
        else:
            bench_hint = ""
            if profile and profile.get("prompt_tps") and profile.get("gen_tps"):
                bench_hint = (
                    f"{sep}{p.FG_MUTED}Bench:{bg} "
                    f"{p.FG_TEXT}{profile['gen_tps']} tok/s gen · {profile['prompt_tps']:,.0f} tok/s prompt{bg}"
                )
            infer_body = f"{bg}{p.FG_DIM}{'·' * 8}{bg}  {p.FG_MUTED}Waiting for first LLM call...{bg}{bench_hint}"
        row_infer = self._hud_row("INFER", p.FG_VIOLET, infer_body, columns)

        # ── 4. VITALS Row: Health / Hunger / Armor / Coordinates / Day-Night / Tools ──
        if live and live.get("health") is not None:
            hp = float(live.get("health", 20))
            max_hp = float(live.get("max_health") or 20)
            food = int(live.get("food", 20))
            armor = int(live.get("armor") or 0)
            hp_col = p.FG_GREEN if hp >= 14 else (p.FG_AMBER if hp >= 8 else p.FG_RED_BOLD)
            fd_col = p.FG_GREEN if food >= 14 else (p.FG_AMBER if food >= 7 else p.FG_RED_BOLD)
            hp_bar = progress_bar(hp, max_hp, width=5)

            pos = live.get("position")
            pos_str = f"[{pos[0]},{pos[1]},{pos[2]}]" if isinstance(pos, list) and len(pos) == 3 else "[?,?,?]"
            dim = short_id(live.get("dimension") or "overworld").replace("the_", "")
            tod = live.get("time")
            if isinstance(tod, (int, float)):
                phase = "Day" if (tod < 12500 or tod >= 23500) else ("Dusk" if tod < 13500 else ("Night" if tod < 22500 else "Dawn"))
                phase_col = p.FG_GOLD if phase in ("Day", "Dawn") else p.FG_VIOLET
            else:
                phase, phase_col = "", p.FG_MUTED
            weather = live.get("weather", "clear")
            wx_str = f" · {weather}" if weather and weather != "clear" else ""

            tools = live.get("owned_tools") or {}
            best_tool = next(
                (short_id(t) for tier in ("netherite", "diamond", "iron", "stone", "wooden")
                 for t in sorted(tools) if tier in t and "_pickaxe" in t),
                next((short_id(t) for t in sorted(tools)), "bare hands"),
            )
            slots = live.get("free_slots")
            slots_str = f" · {slots} free" if slots is not None else ""
            stuck_seg = f"{sep}{p.FG_RED_BOLD}! STUCK{p.RESET}{bg}" if live.get("stuck") else ""

            vitals_body = (
                f"{bg}{hp_col}♥ {hp_bar} {hp:.0f}/{max_hp:.0f}{bg} "
                f"{p.FG_DIM}·{bg} {fd_col}Food {food}/20{bg}"
                + (f" {p.FG_DIM}·{bg} {p.FG_SKY}Arm {armor}{bg}" if armor > 0 else "")
                + f"{sep}{p.FG_TEXT}{dim} {pos_str}{bg}"
                + (f" {phase_col}{phase}{bg}" if phase else "")
                + f"{p.FG_SKY}{wx_str}{bg}"
                + f"{sep}{p.FG_TEAL}{best_tool}{bg}{p.FG_MUTED}{slots_str}{bg}{stuck_seg}"
            )
        else:
            log_label = str(self.log) if self.log else "no world open"
            vitals_body = f"{bg}{p.FG_MUTED}Log: {log_label}  (waiting for live player telemetry){bg}"
        row_vitals = self._hud_row("VITALS", p.FG_GREEN, vitals_body, columns)

        # ── 5. CURRIC Row: Active Tier/Node + Progress + RL Cumulative Reward + Critic Tally ──
        cur = (live or {}).get("curriculum") or {}
        task_id = cur.get("active_node_id") or self.active_task_id or "initializing"
        tier = cur.get("tier") if cur.get("tier") is not None else self.active_tier
        done_cnt = cur.get("completed_count") if cur.get("completed_count") is not None else self.completed_tasks
        tier_badge = f"{p.FG_GOLD_BOLD}[T{tier}]{p.RESET}{bg} " if tier is not None else ""
        prog_seg = ""
        if done_cnt is not None:
            bar = progress_bar(done_cnt, 20, width=6)
            prog_seg = f" {p.FG_TEAL}{bar}{bg} {p.FG_MUTED}{done_cnt}/20{bg}"

        rl_col = p.FG_GREEN if self.rl_total >= 0.0 else (p.FG_AMBER if self.rl_total > -5.0 else p.FG_RED)
        last_rl_str = f" ({self.rl_last:+.2f})" if self.rl_last is not None else ""
        plan_q = (live or {}).get("plan_queued")
        plan_seg = f"{sep}{p.FG_SKY}Plan {plan_q}{bg}" if plan_q else (
            f"{sep}{p.FG_CYAN}Q-Fast {self.q_fast_hits}{bg}" if self.q_fast_hits else ""
        )
        curric_body = (
            f"{bg}{tier_badge}{p.FG_BRIGHT}{task_id}{p.RESET}{bg}{prog_seg}"
            f"{sep}{p.FG_MUTED}RL{bg} {rl_col}{self.rl_total:+.2f}{bg}{p.FG_MUTED}{last_rl_str}{bg}"
            f"{sep}{p.FG_GREEN}{self.critic_passes}✓{bg} "
            f"{p.FG_MUTED}{self.critic_steps}·{bg} "
            f"{p.FG_RED}{self.critic_fails}✗{bg}{plan_seg}"
        )
        row_curric = self._hud_row("CURRIC", p.FG_GOLD, curric_body, columns)

        # ── 6. GOAL Row: Active Strategic/Actor Goal + Latest Action ──
        goal_txt = ((live or {}).get("goal") or self.last_goal or cur.get("objective") or "Exploring & gathering survival essentials").strip()
        last_act = self.last_action_summary
        if not last_act and live and live.get("action"):
            tgt = short_id(live.get("target", ""))
            last_act = f"{live['action']}" + (f" {tgt}" if tgt else "") + f" ({live.get('source', 'actor')})"
        if last_act:
            act_part = f"{sep}{p.FG_MUTED}Last:{bg} {p.FG_GREEN}▸ {last_act}{bg}"
            goal_max = max(18, (columns - 14) - visible_len(act_part))
            goal_trimmed = truncate_ansi(goal_txt, goal_max)
            goal_body = f"{bg}{p.FG_TEXT}{goal_trimmed}{bg}{act_part}"
        else:
            goal_body = f"{bg}{p.FG_TEXT}{goal_txt}{bg}"
        row_goal = self._hud_row("GOAL", p.FG_SKY, goal_body, columns)

        # ── 7. Bottom Border aligned with Stream Columns ──
        row_bottom = self._hud_bottom_border(columns)

        return [row_top, row_model, row_infer, row_vitals, row_curric, row_goal, row_bottom]

    def draw_status(self, force=False):
        if not self.tty:
            lines = self.status(columns=110)
            plain_sig = [ANSI_ESCAPE_RE.sub("", ln) for ln in lines]
            if plain_sig != self.status_lines:
                self.write("\n".join(lines))
                self.status_lines = plain_sig
            return

        columns, rows = self._columns_rows()
        lines = self.status(columns=columns)
        panel = len(lines)
        region_key = (panel, rows, columns)

        with LOCK:
            out = []
            if region_key != self._region or force:
                self.panel = panel
                self._region = region_key
                self._rendered_header = [""] * panel
                out.append(f"\033[{panel + 1};{rows}r")  # scroll region strictly below HUD
                for index, line in enumerate(lines):
                    out.append(f"\033[{index + 1};1H\033[2K{line}")
                    self._rendered_header[index] = line
                # Place cursor at bottom of scroll region so live stream continues smoothly
                out.append(f"\033[{rows};1H")
            else:
                changed = [i for i, ln in enumerate(lines) if i >= len(self._rendered_header) or self._rendered_header[i] != ln]
                if changed:
                    out.append("\0337")  # save cursor inside scroll region
                    for index in changed:
                        out.append(f"\033[{index + 1};1H\033[2K{lines[index]}")
                    out.append("\0338")  # restore cursor
                    self._rendered_header = list(lines)
            if out:
                sys.stdout.write("".join(out))
                sys.stdout.flush()
        self.status_lines = lines
        self._dirty = False

    def _stream_line(self, stamp, tag, tag_color, body, right=""):
        """Formats a single event line aligned to the HUD's Col 11 (`│`) and Col 22 (`│`) dividers."""
        p = self.p
        cols, _ = self._columns_rows()
        ts = f"{stamp or '--:--:--':>8}"
        prefix = (
            f" {p.FG_DIM}{ts} "
            f"{p.FG_BORDER}│ "
            f"{tag_color}{tag:<8}{p.RESET} "
            f"{p.FG_BORDER}│ "
        )
        # Prefix occupies exact 23 visible columns; leave 1 col margin at right edge
        avail = max(24, cols - 24)
        if right:
            right_vis = visible_len(right)
            if right_vis + 4 < avail:
                body_max = avail - right_vis - 2
                body_trimmed = truncate_ansi(body, body_max)
                gap = max(2, avail - visible_len(body_trimmed) - right_vis)
                return f"{prefix}{body_trimmed}{' ' * gap}{right}{p.RESET}"
        full = f"{body}  {right}" if right else body
        if self.tty:
            full = truncate_ansi(full, avail)
        return f"{prefix}{full}{p.RESET}"

    def _format_critic_and_reward(self, stamp, task_id, passed, category, mode, total_val=None, detail_str=""):
        """Formats either a standalone Critic verdict, standalone RL-Reward, or fused Critic+Reward event."""
        p = self.p
        self.active_task_id = task_id or self.active_task_id
        terms = parse_reward_terms(detail_str)
        terms_text = " · ".join(t[2] for t in terms)

        if total_val is not None:
            self.rl_total += total_val
            self.rl_last = total_val

        cat_label = friendly_category(category) if category else ""
        is_soft_step = (
            passed is False
            and category in ("MISSING_PREREQUISITE", "INCOMPLETE_QUANTITY", "INTERRUPTED")
            and (total_val is None or total_val > -0.5)
        )

        if passed is True or (passed is None and total_val is not None and total_val >= 0.5):
            if passed is not None:
                self.critic_passes += 1
            score_str = f"{total_val:+.2f}" if total_val is not None else "PASS"
            body = (
                f"{p.FG_GREEN_BOLD}✓ {p.FG_BRIGHT}{task_id}{p.RESET}  "
                f"{p.FG_GREEN_BOLD}{score_str}{p.RESET}"
                + (f"  {p.FG_TEAL}{terms_text}" if terms_text else f"  {p.FG_TEAL}{cat_label}")
            )
            right = f"{p.FG_MUTED}{mode}" if mode and mode != "0ms GPU" else ""
            self._dirty = True
            return self._stream_line(stamp, "REWARD", p.FG_GREEN_BOLD, body, right)

        if is_soft_step or (passed is None and total_val is not None and -0.5 < total_val < 0.5):
            if passed is not None:
                self.critic_steps += 1
            score_str = f"{total_val:+.2f}" if total_val is not None else ""
            note_parts = [x for x in (cat_label, terms_text) if x]
            note = f"({', '.join(note_parts)})" if note_parts else ""
            body = (
                f"{p.FG_MUTED}· {p.FG_TEXT}{task_id} "
                + (f"{p.FG_AMBER}{score_str} " if score_str else "")
                + f"{p.FG_MUTED}{note}"
            )
            self._dirty = True
            return self._stream_line(stamp, "EVAL", p.FG_MUTED, body)

        # Hard failure or significant penalty
        if passed is not None:
            self.critic_fails += 1
        score_str = f"{total_val:+.2f}" if total_val is not None else "FAIL"
        note_parts = [x for x in (cat_label, terms_text) if x]
        body = (
            f"{p.FG_RED_BOLD}✗ {p.FG_BRIGHT}{task_id}{p.RESET}  "
            f"{p.FG_RED_BOLD}{score_str}{p.RESET}  "
            f"{p.FG_AMBER}{' · '.join(note_parts)}"
        )
        right = f"{p.FG_MUTED}{mode}" if mode and mode != "0ms GPU" else ""
        self._dirty = True
        return self._stream_line(stamp, "PENALTY", p.FG_RED_BOLD, body, right)

    def _parse_line_parts(self, line):
        match = LOG_PREFIX_RE.match(line)
        if match:
            date_s, time_s, level, body = match.groups()
            return f"{date_s} {time_s}", time_s, level, body
        if re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", line):
            stamp = line[11:19]
            rest = line[20:].lstrip(",0123456789 ")
            for lvl in ("INFO ", "WARNING ", "WARN ", "ERROR ", "DEBUG ", "CRITICAL "):
                if rest.startswith(lvl):
                    return line[:19], stamp, lvl.strip(), rest[len(lvl):]
            return line[:19], stamp, "INFO", rest
        return "", "", "INFO", line

    def format(self, line):
        """Formats a single log line into a rich ANSI table row (returns None for suppressed HTTP/duplicate noise)."""
        p = self.p
        if not line or not line.strip():
            return None
        if line.startswith((" ", "\t")):
            return f"{' ' * 23}{p.FG_DIM}{line.rstrip()}{p.RESET}"

        full_ts, stamp, level, body = self._parse_line_parts(line)
        if '"POST /decide' in body or '"GET /health' in body:
            return None

        # 1. LLM Role Inference
        match = ROLE_RE.match(body)
        if match:
            role, prompt, cached, generated, elapsed, prompt_tps, gen_tps = match.groups()
            g_speed = float(gen_tps) if gen_tps not in (None, "-") else None
            p_speed = float(prompt_tps) if prompt_tps not in (None, "-") else None
            el_val = float(elapsed) if elapsed else 0.0
            p_tok = int(prompt) if prompt and prompt.isdigit() else None
            c_tok = int(cached) if cached and cached.isdigit() else (0 if cached == "0" else None)
            g_tok = int(generated) if generated and generated.isdigit() else None
            try:
                when = time.mktime(time.strptime(full_ts, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                when = time.time()
            self.calls.append((when, g_speed, p_speed, el_val, p_tok, c_tok, g_tok, role))
            self.calls = self.calls[-200:]
            self.last = (
                f"{role} prompt {prompt} tok"
                + (f" @ {prompt_tps} tok/s" if p_speed else "")
                + f", generated {generated} tok"
                + (f" @ {gen_tps} tok/s" if g_speed else "")
                + f" in {elapsed}s"
            )
            self._dirty = True

            role_upper = "STRAT" if role == "strategist" else role.upper()[:8]
            role_col = p.FG_GOLD_BOLD if role == "strategist" else p.FG_VIOLET_BOLD
            p_fmt = f"{p_tok:,}" if p_tok is not None else str(prompt)
            g_fmt = f"{g_tok:,}" if g_tok is not None else str(generated)
            if p_tok and c_tok is not None:
                pct = int(round(100.0 * c_tok / p_tok))
                cache_str = f" {p.FG_TEAL}({pct}% KV){p.RESET}" if pct > 0 else f" {p.FG_DIM}(0% KV)"
            else:
                cache_str = ""
            msg = f"{p.FG_TEXT}{p_fmt}{p.FG_MUTED}→{p.FG_BRIGHT}{g_fmt}{p.RESET}{p.FG_MUTED} tok{cache_str}"

            spd_col = p.FG_GREEN_BOLD if (g_speed or 0) >= 80 else (p.FG_CYAN if (g_speed or 0) >= 35 else p.FG_AMBER)
            right_parts = []
            if g_speed is not None:
                right_parts.append(f"{spd_col}{g_speed:.1f} tok/s{p.RESET}")
            if p_speed is not None:
                p_k = f"{p_speed / 1000:.1f}k" if p_speed >= 1000 else f"{p_speed:.0f}"
                right_parts.append(f"{p.FG_MUTED}{p_k} p/s")
            right_parts.append(f"{p.FG_TEXT}{elapsed}s")
            return self._stream_line(stamp, role_upper, role_col, msg, f" {p.FG_DIM}· ".join(right_parts))

        match = STRATA_ROLE_RE.match(body)
        if match:
            role, elapsed = match.groups()
            role_upper = "STRAT" if role == "strategist" else role.upper()[:8]
            return self._stream_line(
                stamp,
                role_upper,
                p.FG_VIOLET_BOLD,
                f"{p.FG_TEXT}STRATA completion finished",
                f"{p.FG_TEXT}{elapsed}s",
            )

        # 2. Decision / Action
        match = DECISION_RE.match(body)
        if match:
            _, action, raw_target, source, elapsed = match.groups()
            self.decisions_count += 1
            clean_tgt = short_id(raw_target)
            self.last_action_summary = f"{action}" + (f" {clean_tgt}" if clean_tgt else "") + f" ({source}, {elapsed}s)"
            self._dirty = True

            if source == "plan":
                tag, tag_col = "PLAN", p.FG_SKY
            elif "reflex" in source:
                tag, tag_col = "REFLEX", p.FG_RED_BOLD
            elif "skill" in source:
                tag, tag_col = "SKILL", p.FG_TEAL
            else:
                tag, tag_col = "ACTION", p.FG_GREEN_BOLD

            if action in ("escape", "attack", "eat", "sleep", "shelter"):
                act_col = p.FG_GOLD_BOLD
            elif action in ("goto", "explore", "jump", "sneak", "pillar", "tunnel", "bridge", "boat", "portal"):
                act_col = p.FG_SKY
            else:
                act_col = p.FG_GREEN_BOLD

            tgt_disp = f"{p.FG_BRIGHT}{clean_tgt}" if clean_tgt else f"{p.FG_DIM}—"
            msg = f"{act_col}▸ {action:<10}{p.RESET} {tgt_disp}"
            src_disp = "queued" if source == "plan" else source
            right = f"{p.FG_MUTED}{src_disp} {p.FG_DIM}· {p.FG_TEXT}{elapsed}s"
            return self._stream_line(stamp, tag, tag_col, msg, right)

        # 3. Critic & RL-Reward (standalone formatting)
        match = CRITIC_RE.match(body)
        if match:
            verdict, task_id, category, mode = match.groups()
            return self._format_critic_and_reward(stamp, task_id, verdict == "PASS", category, mode)

        match = REWARD_RE.match(body)
        if match:
            task_id, total_s, details = match.groups()
            return self._format_critic_and_reward(stamp, task_id, None, "", "", float(total_s), details)

        # 4. Strategist Goal
        match = STRAT_GOAL_RE.match(body)
        if match:
            goal, reason = match.groups()
            self.last_goal = goal.strip()
            self._dirty = True
            msg = f"{p.FG_CYAN}★ {p.FG_BRIGHT}{goal}{p.RESET}" + (f" {p.FG_MUTED}— {reason}" if reason else "")
            return self._stream_line(stamp, "GOAL", p.FG_CYAN, msg)

        # 5. Curriculum Events
        match = CURRIC_SELECT_RE.match(body)
        if match:
            obj_id, tier_s, rew_s, pre_ok = match.groups()
            self.active_task_id = obj_id
            self.active_tier = int(tier_s)
            self._dirty = True
            cur_key = (obj_id, pre_ok)
            if cur_key == self._last_curric_key:
                return None
            self._last_curric_key = cur_key
            ready = f"{p.FG_GREEN}prereqs ready" if pre_ok == "True" else f"{p.FG_AMBER}prereqs pending"
            msg = f"{p.FG_GOLD}◆ Tier {tier_s} → {p.FG_BRIGHT}{obj_id}{p.RESET}  {p.FG_MUTED}(+{float(rew_s):.0f} pts · {ready}{p.FG_MUTED})"
            return self._stream_line(stamp, "CURRIC", p.FG_GOLD, msg)

        match = CURRIC_RECOV_RE.match(body)
        if match:
            rec_id, expr, front_id = match.groups()
            self.active_task_id = rec_id
            self._dirty = True
            cur_key = ("recov", rec_id, front_id)
            if cur_key == self._last_curric_key:
                return None
            self._last_curric_key = cur_key
            msg = f"{p.FG_AMBER}↺ Recover {p.FG_BRIGHT}{rec_id}{p.RESET} {p.FG_MUTED}(need {expr} for {front_id})"
            return self._stream_line(stamp, "CURRIC", p.FG_GOLD, msg)

        match = CURRIC_REGRESS_RE.match(body)
        if match:
            lost_id, expr = match.groups()
            self._last_curric_key = None
            msg = f"{p.FG_RED}↺ Milestone lost: {p.FG_BRIGHT}{lost_id}{p.RESET} {p.FG_MUTED}({expr}) — rebuilding"
            return self._stream_line(stamp, "CURRIC", p.FG_AMBER, msg)

        match = CURRIC_SPLIT_RE.match(body)
        if match:
            parent_id, cnt, subs = match.groups()
            msg = f"{p.FG_CYAN}▸ Split {p.FG_BRIGHT}{parent_id}{p.RESET} {p.FG_TEXT}into {cnt} micro-tasks: {p.FG_MUTED}{subs}"
            return self._stream_line(stamp, "CURRIC", p.FG_GOLD, msg)

        # 6. Q-Table Fast Path
        match = QFAST_RE.match(body)
        if match:
            ctx_hex, task_id, skill_id, q_val = match.groups()
            self.q_fast_hits += 1
            self._dirty = True
            msg = f"{p.FG_CYAN}▸ {p.FG_BRIGHT}{skill_id}{p.RESET}  {p.FG_MUTED}task {task_id} · 0ms LLM bypass"
            right = f"{p.FG_GREEN_BOLD}Q {float(q_val):+.2f}{p.RESET} {p.FG_DIM}· {p.FG_MUTED}{ctx_hex}"
            return self._stream_line(stamp, "Q-FAST", p.FG_CYAN, msg, right)

        # 7. VRAM / Hardware Opt
        match = VRAM_INFER_RE.match(body)
        if match:
            _, prio, ctx, pred, _, kv_hit, ttft, total_ms = match.groups()
            kv_badge = f"{p.FG_TEAL}KV hit" if kv_hit == "True" else f"{p.FG_MUTED}KV cold"
            msg = f"{p.FG_VIOLET}{prio}{p.RESET} {p.FG_MUTED}· ctx {compact_ctx(ctx)} · max {pred} tok · {kv_badge}"
            right = f"{p.FG_MUTED}TTFT {float(ttft):.0f}ms · {p.FG_TEXT}{float(total_ms) / 1000:.2f}s"
            return self._stream_line(stamp, "VRAM", p.FG_VIOLET, msg, right)

        if body.startswith("[VRAM-OPT] "):
            clean = body[len("[VRAM-OPT] "):]
            return self._stream_line(stamp, "VRAM", p.FG_VIOLET, f"{p.FG_TEXT}{clean}")

        # 8. Reflex & Sensory Emergencies
        if body.startswith("[REFLEX-INTERRUPT] ") or "emergency reflex" in body.lower() or body.startswith("AUDIO EMERGENCY:"):
            clean = re.sub(r"^\[REFLEX-INTERRUPT\]\s*", "", body)
            return self._stream_line(stamp, "REFLEX", p.FG_RED_BOLD, f"{p.FG_GOLD_BOLD}! {clean}")

        # 9. Target Redirects & Plan Replans
        match = AUTO_SELECT_RE.match(body)
        if match:
            coords, block_id = match.groups()
            msg = f"{p.FG_SLATE}↳ Auto-selected {p.FG_TEXT}{short_id(block_id)} {p.FG_DIM}({coords} out of sight)"
            return self._stream_line(stamp, "TARGET", p.FG_SLATE, msg)

        match = REDIRECT_WALL_RE.match(body)
        if match:
            blk, p1, p2 = match.groups()
            msg = f"{p.FG_SLATE}↳ Redirected {p.FG_TEXT}{short_id(blk)} {p.FG_MUTED}{p1} → exit wall {p2}"
            return self._stream_line(stamp, "TARGET", p.FG_SLATE, msg)

        match = REDIRECT_VIS_RE.match(body)
        if match:
            _, blk, coords = match.groups()
            msg = f"{p.FG_SLATE}↳ Targeted visible {p.FG_TEXT}{short_id(blk)} {p.FG_MUTED}at {coords}"
            return self._stream_line(stamp, "TARGET", p.FG_SLATE, msg)

        match = OUT_OF_RANGE_RE.match(body)
        if match:
            act, dest, rng = match.groups()
            msg = f"{p.FG_SLATE}↳ Heading toward {p.FG_TEXT}{short_id(dest)} {p.FG_DIM}({act} > {rng}m range)"
            return self._stream_line(stamp, "TARGET", p.FG_SLATE, msg)

        if body.startswith("Plan precondition changed; replanning: "):
            reason = body.split("replanning: ", 1)[1]
            return self._stream_line(stamp, "REPLAN", p.FG_SKY, f"{p.FG_SKY}↺ {p.FG_TEXT}{reason}")

        if body.startswith("Plan interrupted by "):
            return self._stream_line(
                stamp, "REPLAN", p.FG_SKY, f"{p.FG_SKY}↺ {p.FG_MUTED}Plan interrupted (outcome, changed situation, or danger)"
            )

        # 10. Player & Willow Chat
        match = CHAT_IN_RE.match(body)
        if match:
            return self._stream_line(stamp, "PLAYER", p.FG_PINK, f'{p.FG_BRIGHT}"{match.group(1)}"')

        match = CHAT_OUT_RE.match(body)
        if match:
            return self._stream_line(stamp, "WILLOW", p.FG_PINK, f'{p.FG_PINK}❝ {p.FG_BRIGHT}{match.group(1)}{p.RESET} {p.FG_PINK}❞')

        # 11. System / Startup / Model Profile / Skills
        match = LISTENING_RE.match(body)
        if match:
            addr, mdl = match.groups()
            msg = f"{p.FG_GREEN_BOLD}● {p.FG_TEXT}Brain online at {p.FG_CYAN}{addr}{p.FG_TEXT} · model {p.FG_BRIGHT}{mdl}"
            return self._stream_line(stamp, "SYSTEM", p.FG_GREEN_BOLD, msg)

        match = AUTOTUNE_RE.match(body)
        if match:
            gpu, vram_mib, mdl, ctx, max_ctx, p_tps, g_tps = match.groups()
            vram_gb = f"{int(vram_mib) / 1024:.0f}GB" if vram_mib.isdigit() else f"{vram_mib}MiB"
            msg = (
                f"{p.FG_GREEN}✓ {p.FG_CYAN}{mdl}{p.FG_TEXT} on {p.FG_BRIGHT}{gpu.strip()}{p.RESET}{p.FG_MUTED} ({vram_gb}) · "
                f"ctx {compact_ctx(ctx)}/{compact_ctx(max_ctx)}"
            )
            right = f"{p.FG_GREEN_BOLD}{float(g_tps):.1f} tok/s{p.RESET} {p.FG_DIM}· {p.FG_MUTED}{float(p_tps) / 1000:.1f}k p/s"
            return self._stream_line(stamp, "TUNING", p.FG_CYAN, msg, right)

        match = MODEL_PROFILE_RE.match(body)
        if match:
            mdl, arch, params, quant, max_ctx, ctx, kv = match.groups()
            msg = (
                f"{p.FG_CYAN}◆ {p.FG_BRIGHT}{mdl}{p.RESET} {p.FG_VIOLET}[{arch} {params} {quant}] "
                f"{p.FG_MUTED}ctx {compact_ctx(ctx)}/{compact_ctx(max_ctx)} · KV ~{kv}"
            )
            return self._stream_line(stamp, "PROFILE", p.FG_CYAN, msg)

        if body.startswith(("Skill installed", "Skill passed dry-run", "Skill outcome:", "Strategist repaired")):
            return self._stream_line(stamp, "SKILL", p.FG_TEAL, f"{p.FG_TEAL}◆ {p.FG_TEXT}{body}")

        # 12. Rejections, Warnings & Errors (strip huge raw JSON dumps and verbose prefixes from live screen)
        if body.startswith(("Model plan rejected;", "Rejected actor response;")):
            clean = re.sub(r"\s*\(raw response:.*$", "", body)
            clean = clean.replace("Rejected actor response; requesting one correction: ", "Retry: ")
            clean = clean.replace("Model plan rejected; no action executed: ", "Blocked: ")
            return self._stream_line(stamp, "REJECT", p.FG_AMBER, f"{p.FG_AMBER}! {p.FG_TEXT}{clean}")

        if level in ("ERROR", "CRITICAL") or body.startswith(("ERROR", "Traceback")):
            return self._stream_line(stamp, "ERROR", p.FG_RED_BOLD, f"{p.FG_RED}✗ {body}")

        if level in ("WARNING", "WARN") or body.startswith("WARNING"):
            clean = re.sub(r"^\[[A-Z-]+\]\s*", "", body)
            return self._stream_line(stamp, "WARN", p.FG_AMBER, f"{p.FG_AMBER}! {p.FG_TEXT}{clean}")

        # Generic info deduplication (prevents identical back-to-back spam)
        if body == self._last_info_dedup:
            return None
        self._last_info_dedup = body
        clean = re.sub(r"^\[([A-Z-]+)\]\s*", "", body)
        return self._stream_line(stamp, "INFO", p.FG_MUTED, f"{p.FG_TEXT}{clean}")

    def _flush_pending_critic(self):
        if self._pending_critic is None:
            return
        pc = self._pending_critic
        self._pending_critic = None
        rendered = self._format_critic_and_reward(pc["stamp"], pc["task"], pc["passed"], pc["category"], pc["mode"])
        if rendered:
            self.write(rendered)

    def emit(self, line):
        """Emits a log line, fusing consecutive `[CRITIC]` + `[RL-REWARD]` pairs into a single compact row."""
        _, stamp, _, body = self._parse_line_parts(line)
        critic_match = CRITIC_RE.match(body)
        if critic_match:
            self._flush_pending_critic()
            verdict, task_id, category, mode = critic_match.groups()
            self._pending_critic = {
                "stamp": stamp,
                "passed": verdict == "PASS",
                "task": task_id,
                "category": category,
                "mode": mode,
                "mono": time.monotonic(),
            }
            return

        reward_match = REWARD_RE.match(body)
        if reward_match and self._pending_critic is not None:
            task_id, total_s, details = reward_match.groups()
            if self._pending_critic["task"] == task_id:
                pc = self._pending_critic
                self._pending_critic = None
                fused = self._format_critic_and_reward(
                    stamp or pc["stamp"],
                    task_id,
                    pc["passed"],
                    pc["category"],
                    pc["mode"],
                    float(total_s),
                    details,
                )
                if fused:
                    self.write(fused)
                return

        self._flush_pending_critic()
        formatted = self.format(line)
        if formatted:
            self.write(formatted)

    def tail(self):
        position, inode = None, None
        while True:
            try:
                if self._pending_critic and (time.monotonic() - self._pending_critic["mono"]) > 0.25:
                    self._flush_pending_critic()
                info = self.log.stat()
                if inode != info.st_ino or (position is not None and info.st_size < position):
                    with self.log.open("rb") as handle:
                        lines = handle.read().decode("utf-8", "replace").splitlines()
                    if position is None:
                        for line in lines[-35:]:
                            self.emit(line)
                    else:
                        for line in lines:
                            self.emit(line)
                    self._flush_pending_critic()
                    position, inode = info.st_size, info.st_ino
                elif info.st_size > position:
                    with self.log.open("rb") as handle:
                        handle.seek(position)
                        chunk = handle.read(info.st_size - position)
                    position = info.st_size
                    for line in chunk.decode("utf-8", "replace").splitlines():
                        self.emit(line)
            except FileNotFoundError:
                pass
            time.sleep(0.2)

    def journal(self):
        important = re.compile(
            r"context shift|truncated = 1|too large for model|out of memory|level=ERROR|"
            r"panic|failed to load|CUDA error",
            re.IGNORECASE,
        )
        try:
            process = subprocess.Popen(
                ["journalctl", "-u", "ollama", "-f", "-n", "0", "-o", "cat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return
        for line in process.stdout:
            if important.search(line):
                msg = f"{self.p.FG_AMBER}! {self.p.FG_TEXT}{line.strip()[:200]}"
                self.write(self._stream_line(time.strftime("%H:%M:%S"), "OLLAMA", self.p.FG_AMBER, msg))

    def run_once(self, tail_lines=30):
        """Renders the HUD and recent log stream once (useful for snapshots and non-interactive diagnostics)."""
        if self.log and self.log.is_file():
            lines = self.log.read_text(encoding="utf-8", errors="replace").splitlines()
            buffered = []
            orig_write = self.write
            self.write = buffered.append
            try:
                for line in lines[-tail_lines:]:
                    self.emit(line)
                self._flush_pending_critic()
            finally:
                self.write = orig_write
        else:
            buffered = []
        cols, _ = self._columns_rows()
        for row in self.status(columns=cols):
            self.write(row)
        for item in buffered:
            self.write(item)

    def run(self):
        p = self.p
        if self.tty:
            # Clear screen, hide cursor flicker during redraws, set terminal title
            sys.stdout.write("\033[2J\033]0;Willow AI / Ollama\007")
            sys.stdout.flush()
        self.draw_status(force=True)
        if self.tty:
            sys.stdout.write(f"\033[{self.panel + 1};1H")
            sys.stdout.flush()
        log_hint = str(self.log) if self.log else "no world open"
        self.write(
            self._stream_line(
                time.strftime("%H:%M:%S"),
                "CONSOLE",
                p.FG_BRAND,
                f"{p.FG_MUTED}Streaming {p.FG_TEXT}{log_hint}{p.FG_MUTED}  (Ctrl+C closes window)",
            )
        )
        if self.log:
            threading.Thread(target=self.tail, daemon=True).start()
        threading.Thread(target=self.journal, daemon=True).start()
        last_periodic = time.monotonic()
        try:
            while True:
                time.sleep(0.5)
                now = time.monotonic()
                cols, rows = self._columns_rows()
                resized = self._region != (self.panel, rows, cols)
                if self._dirty or resized or (now - last_periodic) >= 2.0:
                    self.draw_status()
                    last_periodic = now
        except KeyboardInterrupt:
            pass
        finally:
            if self.tty:
                sys.stdout.write("\033[r" + p.RESET + "\n")
                sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Willow AI / Ollama live telemetry console")
    parser.add_argument("--log", help="Path to <world>/localai/brain.log")
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA, help="Ollama API base URL")
    parser.add_argument("--once", action="store_true", help="Render HUD + recent log tail once and exit")
    parser.add_argument("--color", dest="color", action="store_true", default=None, help="Force ANSI colors")
    parser.add_argument("--no-color", dest="color", action="store_false", help="Disable ANSI colors")
    args = parser.parse_args()
    if os.name != "nt":
        os.environ.setdefault("TERM", "xterm-256color")
    console = Console(args.log, ollama=args.ollama, color=args.color)
    if args.once:
        console.run_once()
    else:
        console.run()


if __name__ == "__main__":
    main()
