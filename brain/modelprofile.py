#!/usr/bin/env python3
"""Per-model runtime profile derived from Ollama /api/show metadata plus user overrides.

Run directly to print a profile as JSON (used by the in-game settings screen):
    python3 modelprofile.py --model llama3:8b [--settings '{"contextLength": 8192}']
"""

import argparse
import ctypes
import functools
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

DEFAULT_MODEL = "gemma4:12b-it-qat"
DEFAULT_CONTEXT_CAP = 32768
DEFAULT_ACTOR_TOKENS = 1024
DEFAULT_PLAN_TOKENS = 4096
DEFAULT_PROMPT_BYTES = 20000
DEFAULT_ACTOR_TIMEOUT = 240
DEFAULT_PLAN_TIMEOUT = 90
DEFAULT_PLAN_MAX_AGE = 240

MIN_CONTEXT = 4096
MAX_CONTEXT = 262144
KV_BUDGET_BYTES = 8 * 1024 ** 3
BYTES_PER_TOKEN = 3.5
TEMPLATE_TOKENS = 256
PROMPT_BYTES_CAP = 131072
PROMPT_BYTES_MIN = 6000
SETTING_KEYS = {"contextLength", "actorTokens", "planTokens", "temperature", "think", "vision", "strategistTemperature", "roles", "bounded_repetition"}

DECISION_LIMITS = {
    "target": 96,
    "goal": 240,
    "say": 240,
    "memory": 240,
    "waypoint": 48,
    "plan": 8,
    "code": 4096,
}

GAMEPLAY_LIMITS = {
    "goto_max_range": 24,
    "goto_min_distance": 2,
    "danger_food": 14,
    "danger_health": 8.0,
    "danger_hostile_distance": 6.0,
    "stockpile_plank_equiv": 128,
    "stockpile_cobble": 192,
    "skill_loop_consecutive": 5,
    "skill_loop_no_progress": 3,
    "lessons_cap": 2000,
    "decisions_cap": 20000,
    "response_limit_bytes": 131072,
    "request_limit_bytes": 262144,
}

PRESETS = {
    "low_vram": {
        "name": "low_vram",
        "description": "GPUs with <= 8 GiB VRAM running alongside Minecraft",
        "vram_max_mib": 8192,
        "minecraft_reserve_mib": 2560,
        "safety_margin_mib": 768,
        "max_context": 8192,
        "target_actor_latency_s": 2.5,
        "target_plan_latency_s": 12.0,
        "planner_mode": "disabled",
        "num_parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attention": True,
        "vision_default": False,
    },
    "medium_vram": {
        "name": "medium_vram",
        "description": "GPUs with 10-16 GiB VRAM running alongside Minecraft + Iris shaders",
        "vram_max_mib": 16384,
        "minecraft_reserve_mib": 3200,
        "safety_margin_mib": 1024,
        "max_context": 16384,
        "target_actor_latency_s": 2.5,
        "target_plan_latency_s": 15.0,
        "planner_mode": "idle_only",
        "num_parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attention": True,
        "vision_default": False,
    },
    "high_vram": {
        "name": "high_vram",
        "description": "GPUs with >= 20 GiB VRAM running alongside Minecraft + shaders",
        "vram_max_mib": 999999,
        "minecraft_reserve_mib": 4096,
        "safety_margin_mib": 1536,
        "max_context": 32768,
        "target_actor_latency_s": 2.0,
        "target_plan_latency_s": 15.0,
        "planner_mode": "idle_only",
        "num_parallel": 1,
        "kv_cache_type": "f16",
        "flash_attention": True,
        "vision_default": True,
    },
    "rtx4070ti_5800x3d": {
        "name": "rtx4070ti_5800x3d",
        "description": "Tuned preset for AMD Ryzen 7 5800X3D (8C/16T) + NVIDIA RTX 4070 Ti (12 GiB) + 62.7 GiB RAM",
        "vram_max_mib": 12282,
        "minecraft_reserve_mib": 3200,
        "safety_margin_mib": 1024,
        "max_context": 16384,
        "num_thread": 6,
        "target_actor_latency_s": 2.5,
        "target_plan_latency_s": 15.0,
        "planner_mode": "idle_only",
        "num_parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attention": True,
        "vision_default": False,
    },
}


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        numbers = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return max(numbers) if numbers else None
    return None


def _round_tokens(value, low=256, high=16384, step=64):
    clamped = max(low, min(high, int(round(value))))
    return max(low, min(high, ((clamped + step // 2) // step) * step))


@functools.lru_cache(maxsize=1)
def detect_hardware():
    """Detect real CPU, RAM, storage, and GPU VRAM using NVML ctypes with CLI/sysfs fallbacks."""
    hw = {
        "os_release": "",
        "kernel": "",
        "cpu_model": "unknown",
        "cpu_cores": 8,
        "cpu_threads": os.cpu_count() or 16,
        "cpu_max_mhz": 0.0,
        "num_thread": 6,
        "ram_total_mib": 16384,
        "ram_available_mib": 8192,
        "swap_total_mib": 0,
        "gpu_name": "unknown",
        "vram_total_mib": 12282,
        "vram_used_mib": 1315,
        "vram_free_mib": 10967,
        "gpu_source": "default_fallback",
        "strata_nvme_available": os.path.isdir("/mnt/LINUX-NVME/STRATA"),
    }
    try:
        hw["kernel"] = os.uname().release
    except OSError:
        pass
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("PRETTY_NAME="):
                    hw["os_release"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    try:
        cores = None
        model_name = None
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if model_name is None and line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                elif cores is None and line.startswith("cpu cores"):
                    cores = int(line.split(":", 1)[1].strip())
                if model_name and cores:
                    break
        if model_name:
            hw["cpu_model"] = model_name
        if cores and cores > 0:
            hw["cpu_cores"] = cores
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", encoding="utf-8") as handle:
            hw["cpu_max_mhz"] = round(int(handle.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    # Reserve at least 2 physical cores (4 threads on SMT) for Minecraft's main/render/chunk threads.
    hw["num_thread"] = max(2, min(8, hw["cpu_cores"] - 2))

    try:
        mem = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        mem[key] = int(val) // 1024
        if "MemTotal" in mem:
            hw["ram_total_mib"] = mem["MemTotal"]
        if "MemAvailable" in mem:
            hw["ram_available_mib"] = mem["MemAvailable"]
        if "SwapTotal" in mem:
            hw["swap_total_mib"] = mem["SwapTotal"]
    except OSError:
        pass

    # 1. Primary GPU detection via libnvidia-ml.so.1 (works without external binaries in PATH)
    nvml_ok = False
    for soname in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            nvml = ctypes.CDLL(soname)
            if nvml.nvmlInit_v2() == 0:
                try:
                    handle = ctypes.c_void_p()
                    if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                        name_buf = ctypes.create_string_buffer(96)
                        if nvml.nvmlDeviceGetName(handle, name_buf, 96) == 0:
                            hw["gpu_name"] = name_buf.value.decode("utf-8", errors="replace")

                        class _NvmlMemory(ctypes.Structure):
                            _fields_ = [
                                ("total", ctypes.c_ulonglong),
                                ("free", ctypes.c_ulonglong),
                                ("used", ctypes.c_ulonglong),
                            ]

                        mem_info = _NvmlMemory()
                        if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem_info)) == 0 and mem_info.total > 0:
                            hw["vram_total_mib"] = int(mem_info.total // (1024 ** 2))
                            hw["vram_used_mib"] = int(mem_info.used // (1024 ** 2))
                            hw["vram_free_mib"] = int(mem_info.free // (1024 ** 2))
                            hw["gpu_source"] = "nvml"
                            nvml_ok = True
                finally:
                    nvml.nvmlShutdown()
            if nvml_ok:
                break
        except Exception:
            continue

    # 2. Fallback: nvidia-smi CLI
    if not nvml_ok:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=3,
                text=True,
            ).strip()
            first = out.splitlines()[0] if out else ""
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                hw["gpu_name"] = parts[0]
                hw["vram_total_mib"] = int(float(parts[1]))
                hw["vram_used_mib"] = int(float(parts[2]))
                hw["vram_free_mib"] = int(float(parts[3]))
                hw["gpu_source"] = "nvidia-smi"
                nvml_ok = True
        except Exception:
            pass

    # 3. Fallback: Linux DRM sysfs (AMD / Intel / generic DRM)
    if not nvml_ok:
        try:
            for card in sorted(os.listdir("/sys/class/drm")):
                base = f"/sys/class/drm/{card}/device"
                total_path = f"{base}/mem_info_vram_total"
                used_path = f"{base}/mem_info_vram_used"
                if os.path.isfile(total_path):
                    with open(total_path, encoding="utf-8") as handle:
                        total_b = int(handle.read().strip())
                    used_b = 0
                    if os.path.isfile(used_path):
                        with open(used_path, encoding="utf-8") as handle:
                            used_b = int(handle.read().strip())
                    if total_b > 0:
                        hw["vram_total_mib"] = total_b // (1024 ** 2)
                        hw["vram_used_mib"] = used_b // (1024 ** 2)
                        hw["vram_free_mib"] = max(0, hw["vram_total_mib"] - hw["vram_used_mib"])
                        hw["gpu_source"] = "sysfs_drm"
                        break
        except Exception:
            pass

    hw["preset"] = select_preset(hw)
    return hw


def select_preset(hw):
    """Select the hardware preset from detected GPU and CPU specs."""
    if not isinstance(hw, dict):
        return "rtx4070ti_5800x3d"
    if hw.get("preset") in PRESETS:
        return hw["preset"]
    gpu_lower = str(hw.get("gpu_name") or "").lower()
    cpu_lower = str(hw.get("cpu_model") or "").lower()
    vram_total_mib = int(hw.get("vram_total_mib") or 12282)
    if "4070 ti" in gpu_lower and "5800x3d" in cpu_lower:
        return "rtx4070ti_5800x3d"
    if vram_total_mib <= 8192:
        return "low_vram"
    if vram_total_mib <= 16384:
        return "medium_vram"
    return "high_vram"


def estimate_weights_mib(model, show=None):
    """Estimate model weight footprint in MiB from parameter count and quantization when /api/tags is unavailable."""
    name = (model or "").lower()
    if "strata" in name:
        return 11480
    details = (show or {}).get("details") or {}
    param_str = str(details.get("parameter_size") or "")
    quant_str = str(details.get("quantization_level") or "").upper()
    match = re.search(r"(\d+(?:\.\d+)?)\s*([BM])", param_str, re.IGNORECASE)
    if not match:
        match = re.search(r":(\d+(?:\.\d+)?)([bm])", name, re.IGNORECASE)
    params_b = 8.0
    if match:
        val = float(match.group(1))
        unit = match.group(2).upper()
        params_b = val if unit == "B" else val / 1000.0
    elif "e2b" in name:
        params_b = 5.1
    bytes_per_param = 0.62
    if "Q8" in quant_str:
        bytes_per_param = 1.05
    elif "F16" in quant_str or "BF16" in quant_str:
        bytes_per_param = 2.05
    elif "Q2" in quant_str:
        bytes_per_param = 0.35
    elif "Q4" in quant_str or "QAT" in name.upper():
        bytes_per_param = 0.58
    return int(round(params_b * 1024 * bytes_per_param))


def fetch_model_weights_mib(model, show=None, ollama="http://127.0.0.1:11434", timeout=4):
    """Return (weights_mib, ps_vram_mib, source) from Ollama /api/tags and /api/ps, or estimate."""
    weights_mib = None
    ps_vram_mib = 0
    source = "estimated"
    if "strata" in (model or "").lower():
        return 11480, 0, "strata_static"
    try:
        with urllib.request.urlopen(ollama + "/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read(1_048_576))
        target = model.lower()
        for entry in tags.get("models", []):
            ename = str(entry.get("name", "")).lower()
            if ename == target or ename.split(":")[0] == target.split(":")[0]:
                size_b = entry.get("size")
                if isinstance(size_b, int) and size_b > 0:
                    weights_mib = int(round(size_b / (1024 ** 2)))
                    source = "ollama_tags"
                    if ename == target:
                        break
    except Exception:
        pass
    try:
        with urllib.request.urlopen(ollama + "/api/ps", timeout=timeout) as resp:
            ps = json.loads(resp.read(262_144))
        target = model.lower()
        total_ps_vram_mib = 0
        for entry in ps.get("models", []):
            ename = str(entry.get("name", "")).lower()
            vram_b = entry.get("size_vram") or entry.get("size") or 0
            if isinstance(vram_b, int) and vram_b > 0:
                total_ps_vram_mib += int(round(vram_b / (1024 ** 2)))
            if ename == target or ename.split(":")[0] == target.split(":")[0]:
                if weights_mib is None and isinstance(entry.get("size"), int) and entry["size"] > 0:
                    weights_mib = int(round(entry["size"] / (1024 ** 2)))
                    source = "ollama_ps"
        if total_ps_vram_mib > 0:
            ps_vram_mib = total_ps_vram_mib
    except Exception:
        pass
    if weights_mib is None:
        weights_mib = estimate_weights_mib(model, show)
    return weights_mib, ps_vram_mib, source


def derive_timeouts(prompt_tokens, actor_tokens, plan_tokens, prompt_tps, gen_tps, is_strata=False):
    """Derive all timeouts and intervals from measured prompt_tps and gen_tps with safety margins."""
    eff_prompt_tps = max(float(prompt_tps or 400.0), 25.0)
    eff_gen_tps = max(float(gen_tps or 45.0), 5.0)
    # Worst-case prompt + output evaluation times in seconds
    actor_eval_s = (prompt_tokens / eff_prompt_tps) + (actor_tokens / eff_gen_tps)
    plan_eval_s = (prompt_tokens / eff_prompt_tps) + (plan_tokens / eff_gen_tps)
    # Safety factor 3.0x for GPU contention with Minecraft shaders + 12s cold weight load buffer
    cold_buffer_s = 20.0 if is_strata else 12.0
    actor_timeout = int(max(18, min(DEFAULT_ACTOR_TIMEOUT, math.ceil(actor_eval_s * 3.0 + cold_buffer_s))))
    plan_timeout = int(max(25, min(DEFAULT_PLAN_TIMEOUT, math.ceil(plan_eval_s * 3.0 + cold_buffer_s))))
    plan_max_age = int(max(60, min(DEFAULT_PLAN_MAX_AGE, math.ceil(plan_timeout * 2.0))))
    # Multi-step plans chain up to DECISION_LIMITS["plan"] (8) in-game actions (~8-15s each)
    steps_deadline = int(max(60, min(300, math.ceil(DECISION_LIMITS["plan"] * max(12.0, actor_eval_s * 3.0)))))
    next_plan_interval = int(max(20, min(90, math.ceil(plan_eval_s * 3.5 + 12.0))))
    skill_hold_seconds = int(max(25, min(90, math.ceil(actor_eval_s * 8.0 + 15.0))))
    strata_startup_timeout = 45 if is_strata else 30
    keep_alive = "5m" if is_strata else "15m"
    return {
        "actor_timeout": actor_timeout,
        "plan_timeout": plan_timeout,
        "plan_max_age": plan_max_age,
        "steps_deadline": steps_deadline,
        "next_plan_interval": next_plan_interval,
        "skill_hold_seconds": skill_hold_seconds,
        "strata_startup_timeout": strata_startup_timeout,
        "keep_alive": keep_alive,
        "expected_actor_latency_s": round(actor_eval_s, 2),
        "expected_plan_latency_s": round(plan_eval_s, 2),
        "formula": "ceil((prompt_tokens/prompt_tps + output_tokens/gen_tps) * 3.0 + cold_buffer_s)",
    }


def derive_gameplay_limits(
    context,
    actor_tokens,
    plan_tokens,
    prompt_bytes,
    bytes_per_token,
    ram_total_mib=16384,
    vision_enabled=False,
    expected_actor_latency_s=2.5,
    hw_mode=False,
):
    """Derive HTTP buffer limits, SQLite retention caps, stockpile limits, and survival/loop guards from hardware & latency."""
    if not hw_mode:
        return dict(GAMEPLAY_LIMITS)
    max_out_tokens = max(int(actor_tokens or 512), int(plan_tokens or 1024))
    bpt_val = bytes_per_token if (bytes_per_token is not None and bytes_per_token > 0) else BYTES_PER_TOKEN
    bpt = max(2.5, float(bpt_val))
    # HTTP response limit: 4x max output bytes + 64 KiB Ollama JSON metadata envelope, rounded to 32 KiB
    raw_resp = int(max_out_tokens * bpt * 4.0) + 65536
    response_limit_bytes = max(65536, min(262144, ((raw_resp + 32767) // 32768) * 32768))
    # HTTP request limit: 3x prompt budget + vision attachment reserve (256 KiB if vision, else 64 KiB), rounded to 32 KiB
    raw_req = int(prompt_bytes * 3.0) + (262144 if vision_enabled else 65536)
    request_limit_bytes = max(131072, min(524288, ((raw_req + 32767) // 32768) * 32768))
    # SQLite retention caps scaled to system RAM (32 GiB baseline = 1.0x, 62.7 GiB = 2.0x -> 2000 lessons, 20000 decisions)
    ram_scale = max(0.5, min(2.0, float(ram_total_mib or 16384) / 32768.0))
    lessons_cap = int(round(10 * ram_scale)) * 100
    decisions_cap = lessons_cap * 10
    # Survival thresholds derived from expected actor decision latency and mob closing speed (~2.5 m/s)
    latency_s = max(1.0, float(expected_actor_latency_s or 2.5))
    danger_hostile_distance = round(max(6.0, min(8.0, 4.0 + latency_s * 0.6)), 1)
    danger_health = round(max(8.0, min(10.0, 6.0 + latency_s * 0.5)), 1)
    danger_food = int(max(12, min(14, round(10.0 + latency_s * 0.5))))
    # Goto range derived from 16-block observation cube diagonal (ceil(16 * sqrt(2)) + 1 = 24) and reach distance (2)
    obs_scan_radius = 16
    goto_max_range = int(math.ceil(obs_scan_radius * math.sqrt(2))) + 1
    goto_min_distance = 2
    # Stockpile caps derived from 36-slot inventory (max 2 stacks wood = 128 plank-equiv, 3 stacks cobble = 192)
    stockpile_plank_equiv = 2 * 64
    stockpile_cobble = 3 * 64
    # Skill loop guards scaled to decision cadence so slow models abort loops sooner
    skill_loop_consecutive = max(3, min(5, int(round(12.0 / max(2.0, latency_s)))))
    skill_loop_no_progress = max(2, min(3, skill_loop_consecutive - 2))
    return {
        "goto_max_range": goto_max_range,
        "goto_min_distance": goto_min_distance,
        "danger_food": danger_food,
        "danger_health": danger_health,
        "danger_hostile_distance": danger_hostile_distance,
        "stockpile_plank_equiv": stockpile_plank_equiv,
        "stockpile_cobble": stockpile_cobble,
        "skill_loop_consecutive": skill_loop_consecutive,
        "skill_loop_no_progress": skill_loop_no_progress,
        "lessons_cap": lessons_cap,
        "decisions_cap": decisions_cap,
        "response_limit_bytes": response_limit_bytes,
        "request_limit_bytes": request_limit_bytes,
    }


def modelfile_parameters(text):
    values = {}
    for line in (text or "").splitlines():
        match = re.fullmatch(r"\s*([a-z_]+)\s+(\S.*?)\s*", line)
        if not match:
            continue
        key, raw = match.groups()
        try:
            values[key] = json.loads(raw)
        except ValueError:
            values[key] = raw
    return values


def validate_settings(settings):
    if settings is None:
        return {}
    if not isinstance(settings, dict) or set(settings) - SETTING_KEYS:
        raise ValueError("Unknown model setting")
    clean = {}
    for key, value in settings.items():
        if value is None or value == "":
            continue
        if key == "think":
            if value not in ("auto", "on", "off"):
                raise ValueError("think must be auto, on or off")
            clean[key] = value
        elif key == "vision":
            if value not in ("auto", "on", "off"):
                raise ValueError("vision must be auto, on or off")
            clean[key] = value
        elif key == "temperature":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
                raise ValueError("temperature must be between 0 and 2")
            clean[key] = float(value)
        elif key == "strategistTemperature":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
                raise ValueError("strategistTemperature must be between 0 and 2")
            clean[key] = float(value)
        elif key == "bounded_repetition":
            clean[key] = bool(value)
        elif key == "roles":
            if not isinstance(value, dict):
                raise ValueError("roles must be a dictionary")
            clean_roles = {}
            for role_name in ("actor", "strategist"):
                r_val = value.get(role_name)
                if isinstance(r_val, dict):
                    c_role = {}
                    if "temperature" in r_val and r_val["temperature"] is not None:
                        t = r_val["temperature"]
                        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 2:
                            raise ValueError(f"role {role_name} temperature must be between 0 and 2")
                        c_role["temperature"] = float(t)
                    if "num_predict" in r_val and r_val["num_predict"] is not None:
                        np = r_val["num_predict"]
                        if isinstance(np, bool) or not isinstance(np, int) or not 64 <= np <= 16384:
                            raise ValueError(f"role {role_name} num_predict must be an integer between 64 and 16384")
                        c_role["num_predict"] = np
                    if "think" in r_val and r_val["think"] is not None:
                        if r_val["think"] not in ("auto", "on", "off", True, False):
                            raise ValueError(f"role {role_name} think must be auto, on, off, or boolean")
                        c_role["think"] = r_val["think"]
                    if "timeout" in r_val and r_val["timeout"] is not None:
                        to = r_val["timeout"]
                        if isinstance(to, bool) or not isinstance(to, (int, float)) or to <= 0:
                            raise ValueError(f"role {role_name} timeout must be positive")
                        c_role["timeout"] = int(to)
                    clean_roles[role_name] = c_role
            clean[key] = clean_roles
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            low, high = (MIN_CONTEXT, MAX_CONTEXT) if key == "contextLength" else (256, 16384)
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            clean[key] = value
    return clean


def build(model, show, settings=None, hardware=None, benchmark=None):
    """Return a JSON-serializable profile. Raises ValueError for unusable models."""
    settings = validate_settings(settings)
    if not isinstance(show, dict):
        raise ValueError("Invalid model metadata")
    if show.get("remote_host") or show.get("remote_model"):
        raise ValueError("Remote models are not allowed for Willow")
    capabilities = [c for c in show.get("capabilities", []) if isinstance(c, str)]
    is_strata = "strata" in model.lower()
    is_lfm = (
        "lfm" in model.lower()
        or "liquid" in model.lower()
        or "lfm" in str(show.get("details", {}).get("family", "")).lower()
        or "lfm" in str(show.get("model_info", {}).get("general.architecture", "")).lower()
        or "liquid" in str(show.get("model_info", {}).get("general.architecture", "")).lower()
        or "tri-hybrid" in str(show.get("model_info", {}).get("general.architecture", "")).lower()
    )
    if is_strata and not capabilities:
        capabilities = ["completion", "vision", "tools", "thinking"]
    if capabilities and not any(c in capabilities for c in ("completion", "vision", "audio", "multimodal", "tools")) and not is_lfm and not is_strata:
        raise ValueError("Selected model does not support text generation or multimodal interaction")
    info = show.get("model_info") or {}
    details = show.get("details") or {}
    arch = str(info.get("general.architecture") or details.get("family") or ("qwen35" if is_strata else "unknown"))

    def field(name):
        return _number(info.get(f"{arch}.{name}"))

    trained = int(field("context_length") or 8192)
    layers = int(field("block_count") or 0)
    heads = field("attention.head_count") or 0
    raw_kv_heads = info.get(f"{arch}.attention.head_count_kv")
    if isinstance(raw_kv_heads, list) and field("attention.sliding_window"):
        non_zero = [v for v in raw_kv_heads if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
        kv_heads = min(non_zero) if non_zero else (field("attention.head_count_kv") or heads)
    elif raw_kv_heads is None and arch == "gemma4" and heads:
        kv_heads = max(2, int(heads // 4))
    else:
        kv_heads = field("attention.head_count_kv") or heads
    embedding = int(field("embedding_length") or 0)
    key_length = field("attention.key_length") or (embedding / heads if heads else 0)
    value_length = field("attention.value_length") or key_length
    full_layers = layers
    interval = field("full_attention_interval")
    if interval and interval > 1:
        full_layers = max(1, layers // int(interval))
    elif is_lfm:
        # LFM3 tri-hybrid: gated short convolutions carry 0 KV cache; sliding-window has
        # small local cache; full attention occurs in ~1/4 to 1/6 of layers.
        full_layers = max(1, layers // 4) if layers else 4
    pattern = info.get(f"{arch}.attention.sliding_window_pattern")
    if isinstance(pattern, list) and pattern:
        full_layers = sum(1 for sliding in pattern if not sliding) or full_layers
    elif field("attention.sliding_window") and layers:
        # Typical interleaved local/global attention keeps a full cache in about one layer in six.
        full_layers = max(1, layers // 6)
    kv_per_token = int(full_layers * kv_heads * (key_length + value_length) * 2)
    experts = field("expert_count")
    active_experts = field("expert_used_count")

    hw_mode = hardware is not None or benchmark is not None
    hw = hardware or {}
    bench = benchmark or {}
    preset_name = select_preset(hw) if hw_mode else "default"
    preset = PRESETS.get(preset_name, PRESETS["rtx4070ti_5800x3d"])
    sources = {}

    vram_total_mib = int(hw.get("vram_total_mib") or 12282)
    vram_used_mib = int(hw.get("vram_used_mib") or 1315)
    weights_mib = int(hw.get("model_weights_mib") or bench.get("model_weights_mib") or estimate_weights_mib(model, show))
    ps_vram_mib = int(hw["ps_vram_mib"] or 0) if "ps_vram_mib" in hw else int(bench.get("ps_vram_mib") or 0)
    # vram_used_mib and ps_vram_mib are sampled together before any load (see autotune), so the
    # non-model share is simply their difference. A load that is still in progress cannot inflate it.
    non_model_vram_used = max(0, vram_used_mib - ps_vram_mib)
    minecraft_reserve_mib = max(int(preset.get("minecraft_reserve_mib", 3200)), non_model_vram_used)
    safety_margin_mib = int(preset.get("safety_margin_mib", 1024))
    if is_strata:
        # STRATA hybrid engine streams MoE experts from RAM/NVMe and manages its own 16k KV window
        free_kv_budget_mib = max(2560, vram_total_mib - minecraft_reserve_mib - safety_margin_mib)
    else:
        free_kv_budget_mib = max(256, vram_total_mib - minecraft_reserve_mib - weights_mib - safety_margin_mib)

    # Only discount KV cache bytes per token when OLLAMA_KV_CACHE_TYPE (or explicit hw["kv_cache_type"]) is active.
    kv_cache_type = str(hw.get("kv_cache_type") or os.environ.get("OLLAMA_KV_CACHE_TYPE") or "f16").lower()
    if hw_mode and kv_cache_type in ("q8_0", "q8"):
        eff_kv_per_token = kv_per_token // 2
    elif hw_mode and kv_cache_type in ("q4_0", "q4"):
        eff_kv_per_token = kv_per_token // 4
    else:
        eff_kv_per_token = kv_per_token

    if hw_mode:
        kv_budget_bytes = free_kv_budget_mib * (1024 ** 2)
        context_cap = min(DEFAULT_CONTEXT_CAP, int(preset.get("max_context", DEFAULT_CONTEXT_CAP)))
        if is_lfm:
            context_cap = min(context_cap, 16384)
        sources["context"] = (
            f"vram_budget(total={vram_total_mib}MiB - mc={minecraft_reserve_mib}MiB - weights={weights_mib}MiB "
            f"- margin={safety_margin_mib}MiB => kv_budget={free_kv_budget_mib}MiB, kv_type={kv_cache_type}, cap={context_cap})"
        )
    else:
        kv_budget_bytes = KV_BUDGET_BYTES
        context_cap = 16384 if is_lfm else DEFAULT_CONTEXT_CAP
        sources["context"] = "model_metadata_cap"

    measured_prompt_tps = float(bench.get("prompt_tps") or (960.0 if "12b" in model.lower() else 1800.0))
    measured_gen_tps = float(bench.get("gen_tps") or (48.5 if "12b" in model.lower() else (230.0 if is_lfm else 85.0)))
    raw_bpt = bench.get("bytes_per_token")
    try:
        measured_bpt = float(raw_bpt) if raw_bpt is not None else BYTES_PER_TOKEN
    except (ValueError, TypeError):
        measured_bpt = BYTES_PER_TOKEN
    if measured_bpt <= 0:
        measured_bpt = BYTES_PER_TOKEN
    measured_bpt = max(2.5, min(5.5, measured_bpt))
    sources["bytes_per_token"] = "measured_ollama_eval" if "bytes_per_token" in bench else "default_3.5"
    sources["throughput"] = "measured_benchmark" if "gen_tps" in bench else "architecture_estimate"

    auto_context = min(trained, context_cap)
    while eff_kv_per_token and auto_context > MIN_CONTEXT and eff_kv_per_token * auto_context > kv_budget_bytes:
        auto_context //= 2

    if "contextLength" in settings:
        context = min(settings["contextLength"], trained, MAX_CONTEXT)
        context = max(context, min(MIN_CONTEXT, trained))
        sources["context"] = "user_settings"
    elif hw_mode:
        # Choose ONE shared num_ctx for both roles (start at 16384, computed as max(prompt tokens + output cap)
        # plus about 20% headroom, rounded to a power of two, capped at the trained context and hardware budget).
        est_prompt_tokens = int(DEFAULT_PROMPT_BYTES / measured_bpt)
        needed_tokens = int((est_prompt_tokens + max(DEFAULT_ACTOR_TOKENS, DEFAULT_PLAN_TOKENS)) * 1.2)
        power2_ctx = 1 << (needed_tokens - 1).bit_length()
        # Start at 16384 (or preset max_context if higher such as high_vram 32k)
        base_ctx = max(16384, int(preset.get("max_context", 16384)))
        derived_ctx = max(base_ctx, power2_ctx)
        context = max(MIN_CONTEXT, min(derived_ctx, auto_context, trained, MAX_CONTEXT))
        while eff_kv_per_token and context > MIN_CONTEXT and eff_kv_per_token * context > kv_budget_bytes:
            context //= 2
        sources["context"] = f"shared_role_context(needed={needed_tokens}tok -> power2={power2_ctx}, clamped={context})"
    else:
        context = auto_context
        sources["context"] = "model_metadata_cap"

    thinking_supported = "thinking" in capabilities
    think_mode = settings.get("think", "auto")
    if hw_mode and think_mode == "auto":
        # On real hardware with 1-3s target latency, disable actor chain-of-thought unless explicitly turned on.
        # Measured on RTX 4070 Ti: think=True takes 24-68s (500-6000 tokens); think=False takes 0.8-2.6s (65-125 tokens).
        thinking = False
        sources["think"] = "hardware_latency_target(auto->False for 1-3s actor response)"
    else:
        thinking = thinking_supported and think_mode != "off"
        sources["think"] = "user_settings" if "think" in settings else "model_capability"
    think_value = None
    if thinking_supported:
        think_value = thinking
        if thinking and arch.startswith("gptoss"):
            think_value = "medium"

    # Per-role thinking is decided here so that each role's output budget matches its mode.
    role_overrides = settings.get("roles") or {}
    actor_override = role_overrides.get("actor") or {}
    strat_override = role_overrides.get("strategist") or {}
    if "think" in actor_override:
        actor_thinking = thinking_supported and actor_override["think"] in ("on", True)
    elif "think" in settings and settings["think"] != "auto":
        actor_thinking = thinking_supported and settings["think"] == "on"
    elif hw_mode:
        actor_thinking = False
    else:
        actor_thinking = thinking
    if "think" in strat_override:
        strat_thinking = thinking_supported and strat_override["think"] not in ("off", False)
    elif "think" in settings and settings["think"] != "auto":
        strat_thinking = thinking_supported and settings["think"] == "on"
    else:
        strat_thinking = thinking_supported

    if hw_mode:
        target_actor_s = float(preset.get("target_actor_latency_s", 2.5))
        target_plan_s = float(preset.get("target_plan_latency_s", 15.0))
        # Actor JSON schema responses are 60-140 tokens (~260-520 bytes).
        # Size actor_tokens to gen_tps * target_actor_s * 3.5x safety margin so done_reason=="length" never happens.
        if actor_thinking:
            auto_actor = _round_tokens(measured_gen_tps * max(target_actor_s * 8.0, 20.0), low=1536, high=min(4096, context // 2))
        else:
            auto_actor = _round_tokens(measured_gen_tps * target_actor_s * 3.5, low=384, high=min(1024, context // 4))
        if strat_thinking:
            # Reasoning traces for skill code routinely exceed 1500 tokens; a smaller cap just forces the
            # think=False fallback on every call.
            auto_plan = _round_tokens(measured_gen_tps * max(target_plan_s * 2.0, 30.0), low=2048, high=min(4096, context // 2))
        else:
            auto_plan = _round_tokens(measured_gen_tps * target_plan_s * 1.2, low=768, high=min(2048, context // 4))
        sources["actor_tokens"] = f"latency_formula(gen_tps={measured_gen_tps:.1f} * target={target_actor_s}s, think={actor_thinking})"
        sources["plan_tokens"] = f"latency_formula(gen_tps={measured_gen_tps:.1f} * target={target_plan_s}s, think={strat_thinking})"
    elif is_lfm:
        auto_actor = min(1024, max(512, context // 16))
        auto_plan = min(2048, max(1024, context // 8))
        sources["actor_tokens"] = "lfm_default"
        sources["plan_tokens"] = "lfm_default"
    elif thinking:
        auto_actor = auto_plan = min(8192, max(2048, context // 4))
        sources["actor_tokens"] = "thinking_default"
        sources["plan_tokens"] = "thinking_default"
    else:
        # Non-reasoning models answer directly; tight caps stop runaway JSON strings.
        auto_actor = min(1024, context // 8)
        auto_plan = min(4096, context // 4)
        sources["actor_tokens"] = "non_reasoning_default"
        sources["plan_tokens"] = "non_reasoning_default"

    actor_tokens = min(settings.get("actorTokens", auto_actor), context // 2)
    plan_tokens = min(settings.get("planTokens", auto_plan), context // 2)
    if "actorTokens" in settings:
        sources["actor_tokens"] = "user_settings"
    if "planTokens" in settings:
        sources["plan_tokens"] = "user_settings"

    # Decompose output reserve into pure JSON output tokens + thinking reserve + safety margin:
    # prompt_tokens = context - output_tokens - thinking_reserve - safety_margin
    any_thinking = thinking or actor_thinking or strat_thinking
    total_output_reserve = max(actor_tokens, plan_tokens)
    json_output_tokens = min(1024, total_output_reserve) if any_thinking else total_output_reserve
    thinking_reserve_tokens = max(0, total_output_reserve - json_output_tokens) if any_thinking else 0
    safety_margin_tokens = TEMPLATE_TOKENS if not hw_mode else max(TEMPLATE_TOKENS, min(1024, context // 32))
    prompt_tokens = context - json_output_tokens - thinking_reserve_tokens - safety_margin_tokens
    prompt_bytes = int(min(PROMPT_BYTES_CAP, prompt_tokens * measured_bpt))
    if prompt_bytes < PROMPT_BYTES_MIN:
        raise ValueError(f"Context {context} leaves too little room for Willow's observation; "
                         "raise the context length or lower the output limits")
    sources["prompt_budget"] = (
        f"token_formula(ctx={context} - out={json_output_tokens} - think_rsv={thinking_reserve_tokens} "
        f"- margin={safety_margin_tokens} = {prompt_tokens} tok * {measured_bpt:.2f} B/tok = {prompt_bytes} B)"
    )

    # Scale observation trimming stages and compact-prompt threshold to the real token budget
    # Baseline reference: 5700 tokens (~20000 bytes at 3.5 B/tok) = scale 1.0
    trim_scale = round(max(0.5, min(3.0, prompt_tokens / 5700.0)), 2)
    compact_prompt_threshold_tokens = max(1800, min(4096, int(context * 0.25)))

    defaults = modelfile_parameters(show.get("parameters"))
    num_thread = int(hw.get("num_thread") or preset.get("num_thread", 6))
    options = {"num_ctx": context, "num_thread": num_thread}
    sources["num_thread"] = f"cpu_cores({hw.get('cpu_cores', 8)}C -> {num_thread} threads, leaving 2C for Minecraft)"
    repeat_last_n = defaults.get("repeat_last_n")
    if isinstance(repeat_last_n, (int, float)) and repeat_last_n < 0:
        # llama-server strictly requires 0 <= repeat_last_n <= 2147483647.
        # Older Modelfiles frequently used -1, causing HTTP 400 Bad Request in modern Ollama.
        options["repeat_last_n"] = 64
    top_k = defaults.get("top_k")
    if isinstance(top_k, (int, float)) and top_k < 0:
        options["top_k"] = 0
    if "temperature" in settings:
        options["temperature"] = settings["temperature"]
        sources["temperature"] = "user_settings"
    elif hw_mode:
        # Deterministic low temperature (0.2) prevents malformed JSON and erratic coordinate hallucinations.
        options["temperature"] = 0.2
        sources["temperature"] = "hardware_autotune_deterministic(0.2)"
    elif "temperature" not in defaults:
        # LFM models excel with structured outputs at low temperature; default others to 0.6.
        options["temperature"] = 0.2 if (is_lfm or is_strata) else 0.6
        sources["temperature"] = "architecture_default"
    elif is_strata:
        options["temperature"] = 0.2
        sources["temperature"] = "strata_default"
    else:
        sources["temperature"] = "modelfile_default"

    multimodal_caps = {
        "text": "completion" in capabilities or is_lfm or is_strata,
        "vision": "vision" in capabilities or any(v in model.lower() for v in ("vl", "vision", "omni", "strata")),
        "audio": "audio" in capabilities or any(a in model.lower() for a in ("audio", "omni")),
        "omni": "omni" in model.lower() or ("vision" in capabilities and "audio" in capabilities) or is_strata,
        "decision": any(d in model.lower() for d in ("d1", "decision", "strata")),
    }
    vision_setting = settings.get("vision", "auto")
    if vision_setting == "on":
        vision_enabled = bool(multimodal_caps["vision"])
    elif vision_setting == "off":
        vision_enabled = False
    else:
        # Enable direct image input when the model supports vision and VRAM budget has >= 512 MiB
        # for KV + vision (in addition to the 1024 MiB safety_margin_mib already reserved).
        vision_enabled = bool(multimodal_caps["vision"]) and (
            not hw_mode or (preset_name != "low_vram" and free_kv_budget_mib >= 512)
        )

    timeouts = derive_timeouts(
        prompt_tokens=min(prompt_tokens, 6000),
        actor_tokens=actor_tokens,
        plan_tokens=plan_tokens,
        prompt_tps=measured_prompt_tps,
        gen_tps=measured_gen_tps,
        is_strata=is_strata,
    )
    sources["timeouts"] = timeouts["formula"]

    gameplay_limits = derive_gameplay_limits(
        context=context,
        actor_tokens=actor_tokens,
        plan_tokens=plan_tokens,
        prompt_bytes=prompt_bytes,
        bytes_per_token=measured_bpt,
        ram_total_mib=int(hw.get("ram_total_mib") or 64205),
        vision_enabled=vision_enabled,
        expected_actor_latency_s=timeouts.get("expected_actor_latency_s", 2.5),
        hw_mode=hw_mode,
    )
    sources["gameplay_limits"] = "hardware_and_latency_derived" if hw_mode else "default_constants"

    planner_mode = preset.get("planner_mode", "idle_only")
    if (preset_name == "low_vram" and free_kv_budget_mib < 512) or is_strata:
        planner_mode = "disabled"
    sources["planner_mode"] = f"single_gpu_policy({planner_mode}, num_parallel=1)"

    # Configure Role-Aware parameters (actor vs strategist); thinking per role was decided above.
    actor_think_value = None
    if thinking_supported:
        actor_think_value = actor_thinking
        if actor_thinking and arch.startswith("gptoss"):
            actor_think_value = "medium"

    strat_think_value = None
    if thinking_supported:
        strat_think_value = strat_thinking
        if strat_thinking and arch.startswith("gptoss"):
            strat_think_value = "medium"

    # Temperature per role
    actor_temp = actor_override.get("temperature", options.get("temperature"))
    # Strategist temperature: default 0.3 for skill writing, or override
    strat_temp = strat_override.get(
        "temperature",
        settings.get("strategistTemperature", 0.3 if not is_strata else 0.2)
    )

    actor_output = int(actor_override.get("num_predict", actor_tokens))
    strat_output = int(strat_override.get("num_predict", plan_tokens))

    actor_timeout = int(actor_override.get("timeout", timeouts["actor_timeout"]))
    strat_timeout = int(strat_override.get("timeout", timeouts["plan_timeout"]))

    # Token/byte calculations per role
    actor_safety = TEMPLATE_TOKENS if not hw_mode else max(TEMPLATE_TOKENS, min(1024, context // 32))
    actor_prompt_tokens = max(512, context - actor_output - actor_safety)
    actor_prompt_bytes = int(min(PROMPT_BYTES_CAP, actor_prompt_tokens * measured_bpt))

    strat_safety = TEMPLATE_TOKENS if not hw_mode else max(TEMPLATE_TOKENS, min(1024, context // 32))
    strat_json_output = min(1024, strat_output) if strat_thinking else strat_output
    strat_think_rsv = max(0, strat_output - strat_json_output) if strat_thinking else 0
    strat_prompt_tokens = max(512, context - strat_json_output - strat_think_rsv - strat_safety)
    strat_prompt_bytes = int(min(PROMPT_BYTES_CAP, strat_prompt_tokens * measured_bpt))

    roles = {
        "actor": {
            "num_predict": actor_output,
            "think": actor_thinking,
            "think_value": actor_think_value,
            "temperature": actor_temp,
            "timeout": actor_timeout,
            "prompt_tokens": actor_prompt_tokens,
            "prompt_bytes": actor_prompt_bytes,
            "num_ctx": context,
        },
        "strategist": {
            "num_predict": strat_output,
            "think": strat_thinking,
            "think_value": strat_think_value,
            "temperature": strat_temp,
            "timeout": strat_timeout,
            "prompt_tokens": strat_prompt_tokens,
            "prompt_bytes": strat_prompt_bytes,
            "num_ctx": context,
        },
    }

    return {
        "model": model,
        "roles": roles,
        "architecture": arch,
        "family": details.get("family", arch),
        "parameter_size": details.get("parameter_size", "125B" if is_strata else ""),
        "quantization": details.get("quantization_level", "Q2_0" if is_strata else ""),
        "capabilities": capabilities,
        "multimodal_capabilities": multimodal_caps,
        "vision_enabled": vision_enabled,
        "is_lfm": is_lfm,
        "is_strata": is_strata,
        "trained_context": trained,
        "embedding_length": embedding,
        "layers": layers,
        "moe_experts": int(experts) if experts else 0,
        "moe_active_experts": int(active_experts) if active_experts else 0,
        "kv_bytes_per_token": kv_per_token,
        "kv_cache_type": kv_cache_type,
        "kv_cache_mib": round(kv_per_token * context / 1024 ** 2),
        "effective_kv_cache_mib": round(eff_kv_per_token * context / 1024 ** 2),
        "vram_total_mib": vram_total_mib,
        "vram_used_mib": vram_used_mib,
        "minecraft_reserve_mib": minecraft_reserve_mib,
        "model_weights_mib": weights_mib,
        "safety_margin_mib": safety_margin_mib,
        "free_kv_budget_mib": free_kv_budget_mib,
        "context": context,
        "auto_context": auto_context,
        "thinking_supported": thinking_supported,
        "thinking": thinking,
        "think_value": think_value,
        "actor_tokens": actor_tokens,
        "plan_tokens": plan_tokens,
        "json_output_tokens": json_output_tokens,
        "thinking_reserve_tokens": thinking_reserve_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "prompt_tokens": prompt_tokens,
        "bytes_per_token": round(measured_bpt, 3),
        "prompt_bytes": prompt_bytes,
        "trim_scale": trim_scale,
        "compact_prompt_threshold_tokens": compact_prompt_threshold_tokens,
        "prompt_tps": round(measured_prompt_tps, 1),
        "gen_tps": round(measured_gen_tps, 1),
        "timeouts": timeouts,
        "planner_mode": planner_mode,
        "decision_limits": dict(DECISION_LIMITS),
        "gameplay_limits": gameplay_limits,
        "preset": preset_name,
        "sources": sources,
        "model_defaults": {k: v for k, v in defaults.items()
                           if k in ("temperature", "top_k", "top_p", "min_p", "presence_penalty", "repeat_penalty")},
        "options": options,
        "settings": settings,
    }


def warm_load(model, ollama="http://127.0.0.1:11434", context=16384, num_thread=6, timeout=120):
    """Load the model with the runner options the brain will use, so the benchmark measures warm speed
    and the first decision does not pay for a second load. Returns the load time in seconds or None."""
    if "strata" in (model or "").lower():
        return None
    body = {"model": model, "messages": [], "keep_alive": "15m",
            "options": {"num_ctx": context, "num_thread": num_thread}}
    started = time.monotonic()
    try:
        req = urllib.request.Request(ollama + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(65536)
        return round(time.monotonic() - started, 2)
    except Exception:
        return None


def benchmark_model(model, ollama="http://127.0.0.1:11434", show=None, context=16384, num_thread=6, timeout=30):
    """Run a short, realistic structured-output benchmark against Ollama to measure prompt_tps, gen_tps, and B/tok."""
    if "strata" in (model or "").lower():
        return {
            "prompt_tps": 420.0,
            "gen_tps": 56.5,
            "bytes_per_token": 3.8,
            "model_weights_mib": 11480,
            "ps_vram_mib": 11480,
            "source": "strata_measured_baseline",
        }
    weights_mib, ps_vram_mib, weight_src = fetch_model_weights_mib(model, show=show, ollama=ollama, timeout=min(timeout, 4))
    probe_obs = {
        "position": [-120, 64, 240],
        "health": 20.0,
        "food": 18,
        "dimension": "minecraft:overworld",
        "time_of_day": "day",
        "inventory": {"minecraft:oak_log": 2, "minecraft:wooden_pickaxe": 1},
        "hotbar": ["minecraft:wooden_pickaxe", "minecraft:oak_log"],
        "nearby_blocks": ["minecraft:oak_log at -118,64,241 (2m)", "minecraft:stone at -120,63,240 (1m)"],
        "nearby_entities": [],
        "recent_actions": ["mine oak_log: Broken"],
        "task_memory": {"goal": "Gather wood and craft planks"},
    }
    sys_msg = (
        "You are Willow, an autonomous Minecraft survival player. Return ONLY valid JSON matching the schema with "
        "keys action, target, count, x, y, z, goal, say, memory, waypoint, plan. Keep strings short and plan empty."
    )
    user_msg = json.dumps(probe_obs, separators=(",", ":"))
    prompt_byte_len = len(sys_msg.encode("utf-8")) + len(user_msg.encode("utf-8"))
    step_schema = {
        "type": "object",
        "required": ["action", "target", "count", "x", "y", "z"],
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["mine", "craft", "goto", "explore", "idle"]},
            "target": {"type": "string", "maxLength": 96},
            "count": {"type": "integer", "minimum": 1, "maximum": 64},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "z": {"type": "integer"},
        },
    }
    schema = {
        "type": "object",
        "required": ["action", "target", "count", "x", "y", "z", "goal", "say", "memory", "waypoint", "plan"],
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["mine", "craft", "goto", "explore", "idle"]},
            "target": {"type": "string", "maxLength": 96},
            "count": {"type": "integer", "minimum": 1, "maximum": 64},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "z": {"type": "integer"},
            "goal": {"type": "string", "maxLength": 240},
            "say": {"type": "string", "maxLength": 240},
            "memory": {"type": "string", "maxLength": 240},
            "waypoint": {"type": "string", "maxLength": 48},
            "plan": {"type": "array", "maxItems": 2, "items": step_schema},
        },
    }
    caps = (show or {}).get("capabilities") or []
    body = {
        "model": model,
        "stream": False,
        "keep_alive": "15m",
        "format": schema,
        "options": {"num_ctx": context, "num_predict": 192, "num_thread": num_thread, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    if "thinking" in caps:
        body["think"] = False
    started = time.monotonic()
    try:
        req = urllib.request.Request(
            ollama + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read(262_144))
        elapsed = max(time.monotonic() - started, 0.001)
        p_tok = int(data.get("prompt_eval_count") or 0)
        p_dur = int(data.get("prompt_eval_duration") or 0)
        g_tok = int(data.get("eval_count") or 0)
        g_dur = int(data.get("eval_duration") or 0)
        prompt_tps = (p_tok / (p_dur / 1e9)) if (p_tok > 0 and p_dur > 0) else 960.0
        gen_tps = (g_tok / (g_dur / 1e9)) if (g_tok > 0 and g_dur > 0) else 48.5
        bpt = (prompt_byte_len / p_tok) if p_tok > 20 else BYTES_PER_TOKEN
        if bpt <= 0:
            bpt = BYTES_PER_TOKEN
        _, live_ps_vram, _ = fetch_model_weights_mib(model, show=show, ollama=ollama, timeout=2)
        return {
            "prompt_tps": round(prompt_tps, 1),
            "gen_tps": round(gen_tps, 1),
            "bytes_per_token": round(max(2.5, min(5.5, bpt)), 3),
            "prompt_eval_count": p_tok,
            "eval_count": g_tok,
            "done_reason": data.get("done_reason", "stop"),
            "elapsed_s": round(elapsed, 2),
            "model_weights_mib": weights_mib,
            "ps_vram_mib": live_ps_vram or ps_vram_mib,
            "weights_source": weight_src,
            "source": "ollama_live_benchmark",
        }
    except Exception as exc:
        return {
            "prompt_tps": 960.0 if "12b" in model.lower() else 1800.0,
            "gen_tps": 48.5 if "12b" in model.lower() else 85.0,
            "bytes_per_token": BYTES_PER_TOKEN,
            "model_weights_mib": weights_mib,
            "ps_vram_mib": ps_vram_mib,
            "weights_source": weight_src,
            "source": f"fallback({str(exc)[:80]})",
        }


def autotune(model, ollama="http://127.0.0.1:11434", settings=None, show=None, run_benchmark=True, save_path=None):
    """Detect hardware, fetch model metadata, run a quick live benchmark, and return/save the tuned profile."""
    hw = detect_hardware()
    if show is None:
        show = fetch_show(model, ollama=ollama, timeout=10)
    weights_mib, ps_vram_mib, weight_src = fetch_model_weights_mib(model, show=show, ollama=ollama, timeout=4)
    hw["model_weights_mib"] = weights_mib
    hw["ps_vram_mib"] = ps_vram_mib
    hw["weights_source"] = weight_src
    prelim = build(model, show, settings=settings, hardware=hw)
    if run_benchmark:
        load_s = warm_load(model, ollama=ollama, context=prelim["context"],
                           num_thread=prelim["options"]["num_thread"], timeout=120)
        bench = benchmark_model(
            model,
            ollama=ollama,
            show=show,
            context=prelim["context"],
            num_thread=prelim["options"]["num_thread"],
            timeout=40,
        )
        bench["load_s"] = load_s
        # The VRAM sample taken before loading stays authoritative for the Minecraft share; only the
        # free-memory figure is refreshed for display.
        hw_after = detect_hardware()
        hw["vram_free_after_load_mib"] = hw_after["vram_free_mib"]
    else:
        bench = {
            "model_weights_mib": weights_mib,
            "ps_vram_mib": ps_vram_mib,
            "source": "benchmark_skipped",
        }
    profile = build(model, show, settings=settings, hardware=hw, benchmark=bench)
    profile["hardware"] = hw
    profile["benchmark"] = bench
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(profile, handle, indent=2)
        except OSError:
            pass
    return profile


def format_startup_log(profile):
    """Return a single comprehensive startup log line showing every final value and where it came from."""
    hw = profile.get("hardware") or {}
    bench = profile.get("benchmark") or {}
    timeouts = profile.get("timeouts") or {}
    limits = profile.get("gameplay_limits") or {}
    sources = profile.get("sources") or {}
    opts = profile.get("options") or {}
    roles = profile.get("roles") or {}
    actor_role = roles.get("actor") or {}
    strat_role = roles.get("strategist") or {}
    return (
        f"[AUTOTUNE] preset={profile.get('preset')} "
        f"gpu={hw.get('gpu_name', 'unknown')}({profile.get('vram_total_mib')}MiB total, "
        f"mc_rsv={profile.get('minecraft_reserve_mib')}MiB, weights={profile.get('model_weights_mib')}MiB, "
        f"kv_eff={profile.get('effective_kv_cache_mib')}MiB/{profile.get('free_kv_budget_mib')}MiB[{profile.get('kv_cache_type', 'f16')}], "
        f"margin={profile.get('safety_margin_mib')}MiB, src={hw.get('gpu_source', 'profile')}) "
        f"cpu={hw.get('cpu_model', 'unknown')}({hw.get('cpu_cores', 8)}C/{hw.get('cpu_threads', 16)}T -> "
        f"num_thread={opts.get('num_thread')}, src={sources.get('num_thread', 'default')}) "
        f"model={profile.get('model')} ctx={profile.get('context')}(trained={profile.get('trained_context')}, src={sources.get('context')}) "
        f"think=actor:{actor_role.get('think_value', profile.get('think_value'))}/"
        f"strategist:{strat_role.get('think_value', profile.get('think_value'))}(src={sources.get('think')}) "
        f"temp={opts.get('temperature', 'model_default')}(src={sources.get('temperature')}) "
        f"actor_tok={actor_role.get('num_predict', profile.get('actor_tokens'))}(src={sources.get('actor_tokens')}) "
        f"plan_tok={strat_role.get('num_predict', profile.get('plan_tokens'))}(src={sources.get('plan_tokens')}) "
        f"prompt_budget={profile.get('prompt_tokens')}tok/{profile.get('prompt_bytes')}B@"
        f"{profile.get('bytes_per_token')}B/tok(trim_scale={profile.get('trim_scale')}, "
        f"compact_below={profile.get('compact_prompt_threshold_tokens')}tok, src={sources.get('prompt_budget')}) "
        f"speed=(prompt_tps={profile.get('prompt_tps')}, gen_tps={profile.get('gen_tps')}, load={bench.get('load_s')}s, src={bench.get('source', sources.get('throughput'))}) "
        f"timeouts=(actor={timeouts.get('actor_timeout')}s, plan={timeouts.get('plan_timeout')}s, "
        f"plan_max_age={timeouts.get('plan_max_age')}s, steps_deadline={timeouts.get('steps_deadline')}s, "
        f"next_plan={timeouts.get('next_plan_interval')}s, skill_hold={timeouts.get('skill_hold_seconds')}s, "
        f"keep_alive={timeouts.get('keep_alive')}, src={timeouts.get('formula')}) "
        f"limits=(req={limits.get('request_limit_bytes')}B, resp={limits.get('response_limit_bytes')}B, "
        f"lessons={limits.get('lessons_cap')}, decisions={limits.get('decisions_cap')}, "
        f"goto={limits.get('goto_min_distance')}-{limits.get('goto_max_range')}m, src={sources.get('gameplay_limits')}) "
        f"planner_mode={profile.get('planner_mode')} vision={profile.get('vision_enabled')}"
    )


def fetch_show(model, ollama="http://127.0.0.1:11434", timeout=10):
    if "strata" in model.lower():
        # Query Strata endpoint if available, else return pre-configured Strata metadata
        strata_url = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
        try:
            req = urllib.request.Request(f"{strata_url}/v1/models", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=min(timeout, 2)) as resp:
                data = json.loads(resp.read().decode())
                model_name = data.get("data", [{}])[0].get("id", "qwen3.8-flash-next-q2_0")
        except Exception:
            model_name = "qwen3.8-flash-next-q2_0"
        return {
            "model": model,
            "details": {"family": "qwen35", "parameter_size": "125B", "quantization_level": "Q2_0"},
            "model_info": {
                "general.architecture": "qwen35",
                "qwen35.context_length": 32768,
                "qwen35.block_count": 64,
                "qwen35.attention.head_count": 64,
                "qwen35.attention.head_count_kv": 8,
                "qwen35.embedding_length": 5120,
            },
            "capabilities": ["completion", "vision", "tools", "thinking"],
            "parameters": "temperature 0.2\ntop_p 0.95",
        }
    request = urllib.request.Request(ollama + "/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(4_194_305)
    if len(raw) > 4_194_304:
        raise OSError("Ollama model metadata exceeds size limit")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings", default="{}")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    try:
        parsed_settings = json.loads(args.settings)
        if args.legacy:
            result = build(args.model, fetch_show(args.model, args.ollama), parsed_settings)
        else:
            result = autotune(
                args.model,
                ollama=args.ollama,
                settings=parsed_settings,
                run_benchmark=bool(args.autotune and not args.no_benchmark),
            )
    except (OSError, ValueError) as exc:
        result = {"error": str(exc)[:300]}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
