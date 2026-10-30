#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${JAVA_HOME:-}" ]]; then
    export JAVA_HOME="$HOME/.minecraft/runtime/java-runtime-epsilon/linux/java-runtime-epsilon"
fi
if [[ ! -x "$JAVA_HOME/bin/javac" ]]; then
    printf 'A Java 25 JDK is required. Set JAVA_HOME to its directory.\n' >&2
    exit 1
fi
python3 scripts/dependencies.py
if [[ -x tools/gradle-9.6.0/bin/gradle ]]; then
    tools/gradle-9.6.0/bin/gradle --no-daemon --console=plain build "$@"
else
    ./gradlew --no-daemon --console=plain build "$@"
fi
