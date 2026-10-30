#!/usr/bin/env python3
"""Fetch pinned mod dependencies and verify upstream SHA-512 digests."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("fabric-carpet-26.3+v260915.jar",
     "https://cdn.modrinth.com/data/TQTTVgYE/versions/yt9oDFOj/fabric-carpet-26.3%2Bv260915.jar",
     "7a479ba69eb5049cb94365d5044bd58c9f970c50b3484861fe3675f1917f7c99c1b1f50a5e01bdd9ded940d76a995539009a3656e05be2f513f20ec14de991a7"),
    ("fabric-api-0.161.0+26.3.jar",
     "https://cdn.modrinth.com/data/P7dR8mSH/versions/bNnaTiuM/fabric-api-0.161.0%2B26.3.jar",
     "ed6b2586d6fde11fde8472f5a527c51e99b67026e46f94d4bfd85e7e28ce5ee299173ee16ad576ceb51f39f98d30a811086a6deb1a86a524859cc16e12da109d"),
]


def main():
    directory = ROOT / "libs"
    directory.mkdir(exist_ok=True)
    for name, url, digest in FILES:
        destination = directory / name
        if destination.exists() and hashlib.sha512(destination.read_bytes()).hexdigest() == digest:
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000 or hashlib.sha512(data).hexdigest() != digest:
            raise RuntimeError(f"Dependency checksum mismatch: {name}")
        temporary = destination.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
