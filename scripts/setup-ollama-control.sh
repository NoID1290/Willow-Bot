#!/usr/bin/env bash
# One-time setup: lets the Minecraft mod start/stop the system Ollama service
# without a password prompt. Only start/stop of ollama.service is allowed, and
# only for the user who runs this script.
#   bash scripts/setup-ollama-control.sh            install the rule
#   bash scripts/setup-ollama-control.sh --remove   remove it
set -euo pipefail

RULE=/etc/polkit-1/rules.d/49-localai-ollama.rules
TARGET_USER="${SUDO_USER:-$(id -un)}"

if [[ ! "$TARGET_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "Unexpected user name: $TARGET_USER" >&2
    exit 1
fi

if [[ "${1:-}" == "--remove" ]]; then
    sudo rm -f "$RULE"
    echo "Removed $RULE"
    exit 0
fi

sudo tee "$RULE" >/dev/null <<EOF
// Installed by minecraft-local-ai: Minecraft may start/stop Ollama for $TARGET_USER.
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        action.lookup("unit") == "ollama.service" &&
        (action.lookup("verb") == "start" || action.lookup("verb") == "stop") &&
        subject.user == "$TARGET_USER") {
        return polkit.Result.YES;
    }
});
EOF
sudo chmod 644 "$RULE"
echo "Installed $RULE for $TARGET_USER"

if systemctl --no-ask-password stop ollama.service && systemctl --no-ask-password start ollama.service; then
    echo "Verified: $TARGET_USER can stop/start ollama.service without a password."
else
    echo "Rule installed, but systemctl still needs authorization; polkit may need a moment to reload." >&2
    exit 1
fi
