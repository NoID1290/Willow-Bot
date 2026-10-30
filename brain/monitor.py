#!/usr/bin/env python3
"""Willow / Ollama live console: server status, token speed and Willow's brain log.

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

OLLAMA = "http://127.0.0.1:11434"
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
RED, YELLOW, GREEN, CYAN, MAGENTA = "\033[31m", "\033[33m", "\033[32m", "\033[36m", "\033[35m"
ROLE = re.compile(r"role=(\w+) prompt_tokens=(\S+) generated_tokens=(\S+) elapsed=(\S+?)s"
                  r"(?: prompt_tps=(\S+) gen_tps=(\S+))?")
DECISION = re.compile(r"decision=\S+ action=(\S+) target=(\S*) source=(\S+) elapsed=(\S+)")
LOCK = threading.Lock()


def get(path, timeout=2):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as response:
        return json.loads(response.read(1_048_576))


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class Console:
    def __init__(self, log):
        self.log = Path(log) if log else None
        self.tty = sys.stdout.isatty()
        self.last = "waiting for Willow's first model call..."
        self.calls = []
        self.status_lines = []
        self.panel = 0

    def write(self, line):
        with LOCK:
            sys.stdout.write(line + RESET + "\n")
            sys.stdout.flush()

    def status(self):
        lines = []
        profile = self.log.parent / "model-profile.json" if self.log else None
        try:
            version = get("/api/version").get("version", "?")
            lines.append(f"{BOLD}{GREEN}Ollama {version} running{RESET}")
            loaded = get("/api/ps").get("models", [])
            if not loaded:
                if profile and profile.is_file():
                    lines.append(f"{YELLOW}  loading/warming model into VRAM... (first load takes ~15-20s){RESET}")
                else:
                    lines.append(f"{DIM}  no model loaded (Willow loads one on its next decision){RESET}")
            for model in loaded[:3]:
                size, vram = model.get("size", 0), model.get("size_vram", 0)
                gpu = f"{100 * vram / size:.0f}% GPU" if size else "?"
                context = model.get("context_length")
                lines.append(f"  {CYAN}{model.get('name')}{RESET}  {human(size)}  {gpu}"
                             + (f"  ctx {context}" if context else "")
                             + f"  until {str(model.get('expires_at', ''))[11:19]}")
        except (OSError, ValueError):
            lines.append(f"{BOLD}{RED}Ollama is not answering on {OLLAMA}{RESET}")
        if profile and profile.is_file():
            try:
                p = json.loads(profile.read_text())
                moe = f" MoE {p['moe_experts']}x" if p.get("moe_experts") else ""
                lines.append(f"  Willow: {p['model']} [{p['architecture']} {p['parameter_size']} {p['quantization']}{moe}]"
                             f" ctx {p['context']}/{p['trained_context']}"
                             f" think {'on' if p['thinking'] else 'off'}"
                             f" out {p['actor_tokens']}/{p['plan_tokens']} KV~{p['kv_cache_mib']}MiB")
            except (OSError, ValueError, KeyError):
                pass
        recent = [c for c in self.calls if c[0] > time.time() - 60]
        speeds = [c[1] for c in recent if c[1]]
        average = f"{sum(speeds) / len(speeds):.1f} tok/s" if speeds else "-"
        lines.append(f"  Last call: {self.last}")
        lines.append(f"  Last minute: {len(recent)} model calls, average generation {average}")
        return lines

    def draw_status(self):
        lines = self.status()
        if not self.tty:
            if lines != self.status_lines:
                self.write("\n".join(lines))
            self.status_lines = lines
            return
        columns, rows = shutil.get_terminal_size((100, 30))
        panel = len(lines) + 1
        with LOCK:
            out = ["\0337"]  # save cursor
            if panel != self.panel:
                out.append(f"\033[{panel + 1};{rows}r")  # scroll region below the panel
                self.panel = panel
            for index, line in enumerate(lines + [DIM + "─" * columns]):
                out.append(f"\033[{index + 1};1H\033[2K{line[:columns + 40]}{RESET}")
            out.append("\0338")  # restore cursor
            sys.stdout.write("".join(out))
            sys.stdout.flush()
        self.status_lines = lines

    def format(self, line):
        stamp = line[11:19] if re.match(r"\d{4}-\d\d-\d\d ", line) else ""
        body = line[24:] if stamp else line
        if '"POST /decide' in body:
            return None
        match = ROLE.search(body)
        if match:
            role, prompt, generated, elapsed, prompt_tps, gen_tps = match.groups()
            speed = float(gen_tps) if gen_tps not in (None, "-") else None
            try:
                when = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                when = time.time()
            self.calls.append((when, speed))
            self.calls = self.calls[-200:]
            self.last = (f"{role} prompt {prompt} tok"
                         + (f" @ {prompt_tps} tok/s" if prompt_tps and prompt_tps != "-" else "")
                         + f", generated {generated} tok" + (f" @ {gen_tps} tok/s" if speed else "")
                         + f" in {elapsed}s")
            return f"{DIM}{stamp}{RESET} {MAGENTA}[{role}]{RESET} {self.last.split(' ', 1)[1]}"
        match = DECISION.search(body)
        if match:
            action, target, source, elapsed = match.groups()
            return f"{DIM}{stamp}{RESET} {GREEN}→ {action} {target}{RESET} {DIM}({source}, {elapsed}){RESET}"
        if "Strategist goal=" in body:
            return f"{DIM}{stamp}{RESET} {CYAN}★ {body.split('Strategist ', 1)[1]}"
        if "Model profile" in body:
            return f"{DIM}{stamp}{RESET} {BOLD}{body}"
        if " ERROR " in line or body.startswith("ERROR") or body.startswith("Traceback"):
            return f"{DIM}{stamp}{RESET} {RED}{body}"
        if " WARNING " in line or body.startswith("WARNING"):
            return f"{DIM}{stamp}{RESET} {YELLOW}{body}"
        if line.startswith((" ", "\t")):
            return f"{DIM}{line}"
        return f"{DIM}{stamp}{RESET} {body}"

    def tail(self):
        position, inode = None, None
        while True:
            try:
                info = self.log.stat()
                if inode != info.st_ino or (position is not None and info.st_size < position):
                    with self.log.open("rb") as handle:
                        lines = handle.read().decode("utf-8", "replace").splitlines()
                    if position is None:
                        for line in lines[-25:]:
                            self.emit(line)
                    else:
                        for line in lines:
                            self.emit(line)
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
            time.sleep(0.3)

    def emit(self, line):
        formatted = self.format(line)
        if formatted:
            self.write(formatted)

    def journal(self):
        important = re.compile(r"context shift|truncated = 1|too large for model|out of memory|level=ERROR|"
                               r"panic|failed to load|CUDA error", re.IGNORECASE)
        try:
            process = subprocess.Popen(["journalctl", "-u", "ollama", "-f", "-n", "0", "-o", "cat"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        for line in process.stdout:
            if important.search(line):
                self.write(f"{YELLOW}[ollama] {line.strip()[:220]}")

    def run(self):
        if self.tty:
            sys.stdout.write("\033[2J\033]0;Willow AI / Ollama\007")
        self.draw_status()
        if self.tty:
            sys.stdout.write(f"\033[{self.panel + 1};1H")
        self.write(f"{DIM}Willow brain log: {self.log or 'no world open'}   (Ctrl+C closes this window)")
        if self.log:
            threading.Thread(target=self.tail, daemon=True).start()
        threading.Thread(target=self.journal, daemon=True).start()
        try:
            while True:
                time.sleep(2)
                self.draw_status()
        except KeyboardInterrupt:
            pass
        finally:
            if self.tty:
                sys.stdout.write("\033[r" + RESET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    args = parser.parse_args()
    if os.name != "nt":
        os.environ.setdefault("TERM", "xterm-256color")
    Console(args.log).run()


if __name__ == "__main__":
    main()
