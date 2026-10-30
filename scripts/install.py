#!/usr/bin/env python3
"""Install only the three required JARs, backing up conflicting versions."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

from dependencies import ROOT, FILES, main as dependencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minecraft", type=Path, default=Path.home() / ".minecraft")
    args = parser.parse_args()
    dependencies()
    jar = ROOT / "build/libs/minecraft-local-ai-0.1.0.jar"
    if not jar.is_file():
        raise SystemExit("Build first: bash scripts/build.sh")
    mods = args.minecraft / "mods"
    mods.mkdir(parents=True, exist_ok=True)
    sources = [jar] + [ROOT / "libs" / entry[0] for entry in FILES]
    for source in sources:
        destination = mods / source.name
        prefix = ("minecraft-local-ai-" if source == jar else
                  "fabric-carpet-" if source.name.startswith("fabric-carpet-") else "fabric-api-")
        for existing in mods.glob(prefix + "*.jar"):
            if existing.name == source.name and hashlib.sha256(existing.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest():
                continue
            backup = args.minecraft / "config/localai/backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(existing), backup / existing.name)
            print(f"Backed up {existing.name} to {backup}")
        shutil.copy2(source, destination)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError(f"Installation verification failed: {destination}")
        print(f"Installed {destination}")


if __name__ == "__main__":
    main()
