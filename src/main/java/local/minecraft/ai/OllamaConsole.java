package local.minecraft.ai;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import net.fabricmc.loader.api.FabricLoader;

/** Opens a desktop terminal running monitor.py: Ollama status, token speed and Willow's live brain log. */
public final class OllamaConsole {
    private OllamaConsole() {}

    static List<String> terminalCommand(String terminal, List<String> command) {
        var line = new ArrayList<String>();
        switch (terminal) {
            case "konsole" -> line.addAll(List.of("konsole", "-p", "tabtitle=Willow AI / Ollama", "-e"));
            case "gnome-terminal", "kgx" -> line.addAll(List.of(terminal, "--title=Willow AI / Ollama", "--"));
            case "xfce4-terminal", "mate-terminal" -> line.addAll(List.of(terminal, "--title=Willow AI / Ollama", "-x"));
            case "kitty", "wezterm" -> line.addAll(terminal.equals("kitty")
                    ? List.of("kitty", "--title", "Willow AI / Ollama") : List.of("wezterm", "start", "--"));
            default -> line.addAll(List.of(terminal, "-e"));
        }
        line.addAll(command);
        return line;
    }

    static String findTerminal() {
        String path = System.getenv("PATH");
        if (path == null) return null;
        for (String name : List.of("konsole", "gnome-terminal", "kgx", "xfce4-terminal", "mate-terminal",
                "alacritty", "kitty", "wezterm", "foot", "xterm", "x-terminal-emulator")) {
            for (String directory : path.split(File.pathSeparator)) {
                if (!directory.isEmpty() && Files.isExecutable(Path.of(directory, name))) return name;
            }
        }
        return null;
    }

    /** Returns a user-facing message; brainLog may be null when no world is open. */
    public static String open(Path brainLog) {
        try {
            Path runtime = FabricLoader.getInstance().getConfigDir().resolve("localai/runtime");
            BrainBridge.extractScripts(runtime);
            String terminal = findTerminal();
            if (terminal == null) return "No terminal emulator found (tried konsole, gnome-terminal, xterm...).";
            var command = new ArrayList<>(List.of("python3", runtime.resolve("monitor.py").toString()));
            if (brainLog != null) command.addAll(List.of("--log", brainLog.toString()));
            new ProcessBuilder(terminalCommand(terminal, command))
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                    .redirectError(ProcessBuilder.Redirect.DISCARD).start();
            return "Opened the Willow/Ollama console in " + terminal + ".";
        } catch (IOException | RuntimeException e) {
            LocalAiMod.LOG.error("Cannot open Ollama console", e);
            return "Could not open the console: " + LocalAiMod.rootMessage(e);
        }
    }
}
