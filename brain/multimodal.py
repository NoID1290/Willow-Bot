#!/usr/bin/env python3
"""Concurrent multimodal sensory pipeline for Willow and LFM3 local AI.

Coordinates concurrent Sight, Sound, Omni, and Decision streams running simultaneously.
Tracks what Willow is seeing, hearing, and doing in real-time, providing
immediate reflex responses, active tool grounding, and multimodal contexts
for the entire LFM3 family (lfm3:1.7b, lfm3-vl:2b, lfm3-audio:2b, lfm3-omni:2b, d1-2b).
"""

import copy
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid

LOG = logging.getLogger("localai.multimodal")

LFM_FAMILY_MODELS = [
    "lfm3:1.7b",
    "lfm3-vl:2b",
    "lfm3-audio:2b",
    "lfm3-omni:2b",
    "d1-2b:latest",
    "d1-omni:2b",
]


def warm_lfm_family(ollama_url="http://127.0.0.1:11434", timeout=5):
    """Pre-warms the full LFM3 family in Ollama concurrently with keep_alive."""
    def _warm_one(model_name):
        try:
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=json.dumps({"model": model_name, "keep_alive": "24h"}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            LOG.info("Pre-warmed LFM3 family model: %s", model_name)
        except Exception as exc:
            LOG.debug("Could not warm LFM3 model %s: %s", model_name, exc)

    threads = [threading.Thread(target=_warm_one, args=(m,), daemon=True) for m in LFM_FAMILY_MODELS]
    for t in threads:
        t.start()
    return threads


class MultimodalState:
    """Thread-safe synchronized store for concurrent sensory streams."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seeing = {}
        self.listening = {}
        self.doing = {}
        self.omni_synthesis = ""
        self.emergency_reflex = None
        self.emergency_time = 0.0
        self.last_update = 0.0

    def update_sight(self, sight_data):
        with self.lock:
            self.seeing = copy.deepcopy(sight_data)
            self.last_update = time.monotonic()

    def update_hearing(self, hearing_data, emergency=None):
        with self.lock:
            self.listening = copy.deepcopy(hearing_data)
            if emergency:
                self.emergency_reflex = emergency
                self.emergency_time = time.monotonic()
            self.last_update = time.monotonic()

    def update_doing(self, doing_data):
        with self.lock:
            self.doing = copy.deepcopy(doing_data)
            self.last_update = time.monotonic()

    def update_omni(self, synthesis_text):
        with self.lock:
            self.omni_synthesis = str(synthesis_text or "")
            self.last_update = time.monotonic()

    def pop_emergency_reflex(self, max_age=3.0):
        """Reflex computed by the audio worker, or None if it is older than max_age seconds."""
        with self.lock:
            ref = self.emergency_reflex
            self.emergency_reflex = None
            if ref is not None and time.monotonic() - self.emergency_time > max_age:
                return None
            return ref

    def snapshot(self):
        with self.lock:
            return {
                "seeing": copy.deepcopy(self.seeing),
                "listening": copy.deepcopy(self.listening),
                "doing": copy.deepcopy(self.doing),
                "omni_synthesis": self.omni_synthesis,
                "updated_at": self.last_update,
            }


class VisionProcessor(threading.Thread):
    """Concurrent worker processing Willow's visual stream with LFM3-VL."""

    def __init__(self, state: MultimodalState, model_profile: dict = None, ollama: str = "http://127.0.0.1:11434", enabled: bool = False, is_lfm: bool = False):
        super().__init__(name="VisionStreamWorker", daemon=True)
        self.state = state
        self.profile = model_profile or {}
        self.ollama = ollama
        self.enabled = enabled
        self.is_lfm = is_lfm
        self.queue = queue.Queue(maxsize=16)
        self.running = True
        self.last_inference = 0.0
        self.vl_model = "lfm3-vl:2b"

    def submit(self, observation):
        try:
            self.queue.put_nowait(observation)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(observation)
            except (queue.Empty, queue.Full):
                pass

    def run(self):
        while self.running:
            try:
                obs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                sight = obs.get("sight", {})
                if not sight:
                    sight = self._synthesize_sight(obs)

                # Process visual hazards immediately (0ms rule check)
                processed = self._analyze_vision(sight, obs)

                # Query vision model or synthesize visual insight
                now = time.monotonic()
                if self.is_lfm and (now - self.last_inference > 2.5):
                    insight = self._query_lfm_vision(sight, obs)
                    if insight:
                        processed["vision_insight"] = insight
                        processed["lfm_vision_insight"] = insight
                        self.last_inference = now
                elif not self.is_lfm:
                    summary = sight.get("visual_summary", "")
                    looking = sight.get("looking_at", {}).get("id", "air")
                    insight = f"Looking at {looking}. {summary}".strip()
                    if insight:
                        processed["vision_insight"] = insight

                self.state.update_sight(processed)
            except Exception as exc:
                LOG.debug("Vision stream processing error: %s", exc)

    def _synthesize_sight(self, obs):
        entities = obs.get("entities", [])
        return {
            "facing": "Unknown",
            "elevation": "level",
            "visible_entities": entities[:6],
            "visual_summary": f"{len(entities)} entity/entities in immediate surroundings.",
        }

    def _analyze_vision(self, sight, obs):
        analyzed = dict(sight)
        looking_at = sight.get("looking_at", {})
        hazards = []
        if looking_at.get("id") == "minecraft:lava":
            hazards.append("Lava directly in crosshairs")

        visible_entities = sight.get("visible_entities", [])
        threats = [e for e in visible_entities if e.get("hostile") and e.get("distance", 99) < 8.0]
        if threats:
            closest = min(threats, key=lambda e: e.get("distance", 99))
            hazards.append(f"Hostile {closest.get('type', 'mob')} visible at {closest.get('distance')}m ({closest.get('bearing')})")

        analyzed["visual_hazards"] = hazards
        return analyzed

    def _query_lfm_vision(self, sight, obs):
        summary = sight.get("visual_summary", "")
        looking = sight.get("looking_at", {}).get("id", "air")
        entities = [e.get("type", "mob") for e in sight.get("visible_entities", [])[:3]]
        fallback = f"Gaze on {looking}. Entities: {', '.join(entities) if entities else 'none'}. {summary}".strip()
        try:
            prompt = f"Vision check: facing {sight.get('facing', 'unknown')}, looking at {looking}. Entities: {', '.join(entities) if entities else 'none'}. {summary}"
            req = urllib.request.Request(
                f"{self.ollama}/api/generate",
                data=json.dumps({
                    "model": self.vl_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"num_predict": 48, "temperature": 0.2}
                }).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                res = data.get("response", "").strip()[:180]
                return res if res else fallback
        except Exception:
            return fallback if fallback else None

    def stop(self):
        self.running = False


class AudioProcessor(threading.Thread):
    """Concurrent worker processing Willow's auditory stream with LFM3-Audio."""

    def __init__(self, state: MultimodalState, model_profile: dict = None, ollama: str = "http://127.0.0.1:11434", enabled: bool = False, is_lfm: bool = False):
        super().__init__(name="AudioStreamWorker", daemon=True)
        self.state = state
        self.profile = model_profile or {}
        self.ollama = ollama
        self.enabled = enabled
        self.is_lfm = is_lfm
        self.queue = queue.Queue(maxsize=16)
        self.running = True
        self.last_inference = 0.0
        self.audio_model = "lfm3-audio:2b"

    def submit(self, observation):
        try:
            self.queue.put_nowait(observation)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(observation)
            except (queue.Empty, queue.Full):
                pass

    def run(self):
        while self.running:
            try:
                obs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                hearing = dict(obs.get("hearing", {}))
                emergency = self._evaluate_audio_threats(hearing, obs)

                # Query audio insight if alerts/sounds present and enabled
                now = time.monotonic()
                if self.is_lfm and (hearing.get("active_alerts") or hearing.get("recent_sounds")) and now - self.last_inference > 2.5:
                    insight = self._query_lfm_audio(hearing)
                    if insight:
                        hearing["audio_insight"] = insight
                        hearing["lfm_audio_insight"] = insight
                        self.last_inference = now
                elif not self.is_lfm:
                    alerts = hearing.get("active_alerts", [])
                    recent = hearing.get("recent_sounds", [])
                    all_sounds = (alerts + recent)[:3]
                    if all_sounds:
                        hearing["audio_insight"] = ", ".join(f"{s.get('sound')} at {s.get('distance')}m" for s in all_sounds)

                self.state.update_hearing(hearing, emergency=emergency)
            except Exception as exc:
                LOG.debug("Audio stream processing error: %s", exc)

    def _evaluate_audio_threats(self, hearing, obs):
        alerts = hearing.get("active_alerts", [])
        recent = hearing.get("recent_sounds", [])

        # Check for critical audio cue: creeper fuse or explosion
        for sound in alerts + recent:
            name = str(sound.get("sound", "")).lower()
            dist = float(sound.get("distance", 99.0))

            if ("creeper.primed" in name or "explode" in name) and dist < 6.0:
                bearing = sound.get("bearing", "behind")
                LOG.warning("AUDIO EMERGENCY: Creeper fuse primed %.1fm %s! Triggering reflex.", dist, bearing)
                return {
                    "action": "escape",
                    "target": "",
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "goal": f"Dodge: Creeper hissing {bearing} ({dist:.1f}m)!",
                    "say": "Creeper!",
                    "memory": f"Heard creeper primed at {dist:.1f}m",
                    "waypoint": "",
                    "reflex": True,
                }
        return None

    def _query_lfm_audio(self, hearing):
        alerts = hearing.get("active_alerts", [])
        recent = hearing.get("recent_sounds", [])
        all_sounds = (alerts + recent)[:3]
        if not all_sounds:
            return None
        sounds_str = ", ".join(f"{s.get('sound')} at {s.get('distance')}m ({s.get('bearing')})" for s in all_sounds)
        fallback = f"Auditory cues: {sounds_str}."
        try:
            prompt = f"Analyze auditory environment: cues detected: {sounds_str}. Assess danger level."
            req = urllib.request.Request(
                f"{self.ollama}/api/generate",
                data=json.dumps({
                    "model": self.audio_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"num_predict": 48, "temperature": 0.2}
                }).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                res = data.get("response", "").strip()[:180]
                return res if res else fallback
        except Exception:
            return fallback

    def stop(self):
        self.running = False


class OmniProcessor(threading.Thread):
    """Concurrent worker fusing cross-modal perception with LFM3-Omni."""

    def __init__(self, state: MultimodalState, model_profile: dict = None, ollama: str = "http://127.0.0.1:11434", enabled: bool = False, is_lfm: bool = False):
        super().__init__(name="OmniStreamWorker", daemon=True)
        self.state = state
        self.profile = model_profile or {}
        self.ollama = ollama
        self.enabled = enabled
        self.is_lfm = is_lfm
        self.queue = queue.Queue(maxsize=8)
        self.running = True
        self.last_synthesis = 0.0
        self.omni_model = "lfm3-omni:2b"

    def submit(self, observation):
        try:
            self.queue.put_nowait(observation)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(observation)
            except (queue.Empty, queue.Full):
                pass

    def run(self):
        while self.running:
            try:
                obs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            now = time.monotonic()
            if now - self.last_synthesis < 3.5:
                continue

            try:
                if self.is_lfm:
                    synthesis = self._fuse_modalities(obs)
                else:
                    snap = self.state.snapshot()
                    looking = snap.get("seeing", {}).get("looking_at", {}).get("id", "none")
                    doing = snap.get("doing", {}).get("action", "idle")
                    alerts = len(snap.get("listening", {}).get("active_alerts", []))
                    hazards = snap.get("seeing", {}).get("visual_hazards", [])
                    if hazards:
                        synthesis = f"Hazards present ({hazards[0]}). Gaze: {looking}, action: {doing}."
                    elif looking and looking not in ("none", "minecraft:air"):
                        synthesis = f"Gaze on {looking.split(':')[-1]}. Action: {doing}."
                    else:
                        synthesis = f"Awareness: action {doing}, alerts: {alerts}."
                if synthesis:
                    self.state.update_omni(synthesis)
                    self.last_synthesis = now
            except Exception as exc:
                LOG.debug("Omni stream processing error: %s", exc)

    def _fuse_modalities(self, obs):
        snap = self.state.snapshot()
        looking = snap.get("seeing", {}).get("looking_at", {}).get("id", "none")
        doing = snap.get("doing", {}).get("action", "idle")
        alerts = len(snap.get("listening", {}).get("active_alerts", []))
        hazards = snap.get("seeing", {}).get("visual_hazards", [])
        if hazards:
            fallback = f"Hazards present ({hazards[0]}). Gaze: {looking}, action: {doing}."
        elif looking and looking not in ("none", "minecraft:air"):
            fallback = f"Gaze on {looking.split(':')[-1]}. Action: {doing}. Alerts: {alerts}."
        else:
            fallback = f"Awareness: action {doing}, alerts: {alerts}."
        try:
            prompt = f"Cross-modal fusion: Gaze: {looking}, Doing: {doing}, Acoustic alerts: {alerts}. Synthesize situational awareness in one sentence."
            req = urllib.request.Request(
                f"{self.ollama}/api/generate",
                data=json.dumps({
                    "model": self.omni_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"num_predict": 48, "temperature": 0.2}
                }).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                res = data.get("response", "").strip()[:180]
                return res if res else fallback
        except Exception:
            return fallback

    def stop(self):
        self.running = False


class ConcurrentPipeline:
    """Master manager for simultaneous Seeing, Listening, Omni, and Deciding streams."""

    def __init__(self, model: str = "", profile: dict = None, ollama: str = "http://127.0.0.1:11434"):
        self.model = model
        self.profile = profile or {}
        self.ollama = ollama
        is_lfm = bool(model and any(k in str(model).lower() for k in ("lfm", "d1", "ensemble")))
        # Only spawn separate auxiliary model calls when VRAM >= 24 GiB; on <=16 GiB GPUs, loading 6 models
        # (25.6 GiB total) evicts the main actor model and causes severe VRAM thrashing.
        allow_multi_model = bool(is_lfm and int(self.profile.get("vram_total_mib") or 0) >= 24576)
        self.enabled = True
        self.is_lfm = is_lfm
        self.state = MultimodalState()
        self.vision = VisionProcessor(self.state, self.profile, self.ollama, enabled=self.enabled, is_lfm=allow_multi_model)
        self.audio = AudioProcessor(self.state, self.profile, self.ollama, enabled=self.enabled, is_lfm=allow_multi_model)
        self.omni = OmniProcessor(self.state, self.profile, self.ollama, enabled=self.enabled, is_lfm=allow_multi_model)
        self.vision.start()
        self.audio.start()
        self.omni.start()

        if allow_multi_model:
            warm_lfm_family(self.ollama)

    def ingest(self, observation):
        """Dispatches sensory inputs concurrently to vision, audio, and omni streams."""
        if not isinstance(observation, dict):
            return

        doing = observation.get("doing")
        if doing:
            self.state.update_doing(doing)

        # Dispatch to concurrent background streams
        self.vision.submit(observation)
        self.audio.submit(observation)
        self.omni.submit(observation)

    def check_reflex(self, observation):
        """Checks for urgent reflex actions triggered by hearing or vision."""
        hearing = observation.get("hearing", {})
        alerts = hearing.get("active_alerts", [])
        for alert in alerts:
            s_name = str(alert.get("sound", "")).lower()
            dist = float(alert.get("distance", 99.0))
            if "creeper.primed" in s_name and dist < 5.0:
                bearing = alert.get("bearing", "behind")
                return {
                    "action": "escape",
                    "target": "",
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "goal": f"Emergency dodge: Creeper primed {bearing}!",
                    "say": "Creeper!",
                    "memory": "Evaded primed creeper",
                    "waypoint": "",
                    "reflex": True,
                }

        return self.state.pop_emergency_reflex()

    def enrich(self, context):
        """Enriches the context dictionary with live concurrent perception streams."""
        snap = self.state.snapshot()
        obs_sight = context.get("sight")
        obs_hearing = context.get("hearing")
        obs_doing = context.get("doing")

        seeing = obs_sight if obs_sight else snap["seeing"]
        listening = obs_hearing if obs_hearing else snap["listening"]
        doing = obs_doing if obs_doing else snap["doing"]

        enriched = dict(context)
        enriched["seeing"] = seeing
        enriched["listening"] = listening
        enriched["doing"] = doing
        if snap.get("omni_synthesis"):
            enriched["omni_synthesis"] = snap["omni_synthesis"]
            enriched["lfm_omni_synthesis"] = snap["omni_synthesis"]
        return enriched

    def close(self):
        """Cleanly halts background streams."""
        self.vision.stop()
        self.audio.stop()
        self.omni.stop()
        self.vision.join(timeout=0.5)
        self.audio.join(timeout=0.5)
        self.omni.join(timeout=0.5)

