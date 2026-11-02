package local.minecraft.ai;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Ties the local Ollama server to Minecraft: started on demand, unloaded and stopped when the game quits. */
public final class OllamaService {
    private static final OllamaService DEFAULT = new OllamaService(URI.create("http://127.0.0.1:11434"),
            List.of("systemctl", "--no-ask-password"), "ollama.service");

    private final URI base;
    private final List<String> systemctl;
    private final String unit;
    private final AtomicBoolean stopped = new AtomicBoolean();
    private final Object startLock = new Object();

    OllamaService(URI base, List<String> systemctl, String unit) {
        this.base = base;
        this.systemctl = List.copyOf(systemctl);
        this.unit = unit;
    }

    public static OllamaService get() { return DEFAULT; }

    boolean reachable() {
        try (var http = client()) {
            var response = http.send(HttpRequest.newBuilder(base.resolve("/api/version"))
                    .timeout(Duration.ofSeconds(2)).GET().build(), HttpResponse.BodyHandlers.discarding());
            return response.statusCode() == 200;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /** Starts the Ollama service if it is not answering; returns once it answers or the wait expires. */
    public boolean ensureRunning() {
        if (reachable()) return true;
        synchronized (startLock) {
            if (reachable()) return true;
            stopped.set(false);
            LocalAiMod.LOG.info("Ollama is not running; starting {}", unit);
            if (!runSystemctl("start", 20)) {
                LocalAiMod.LOG.warn("Could not start {}. Run scripts/setup-ollama-control.sh once, "
                        + "or start Ollama manually.", unit);
            }
            for (int i = 0; i < 60; i++) {
                if (reachable()) return true;
                try {
                    Thread.sleep(250);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return false;
                }
            }
            return reachable();
        }
    }

    public static final List<String> LFM_FAMILY_MODELS = List.of(
            "lfm3:1.7b", "lfm3-vl:2b", "lfm3-audio:2b", "lfm3-omni:2b", "d1-2b:latest", "d1-omni:2b"
    );

    /** Preloads model weights into GPU VRAM in background. */
    public void preload(String model) {
        preload(model, 0);
    }

    /**
     * Preloads the model with the context size the brain will use. Ollama restarts the runner whenever
     * num_ctx changes, so loading with an unknown context would only add a second load; in that case the
     * brain's own startup benchmark performs the single warm load.
     */
    public void preload(String model, int numCtx) {
        preload(model, numCtx, 0);
    }

    public void preload(String model, int numCtx, int numThread) {
        if (model == null || model.isBlank()) return;
        if (model.toLowerCase(java.util.Locale.ROOT).contains("strata")) return;
        if (numCtx > 0) {
            Thread.startVirtualThread(() -> {
                try (var http = client()) {
                    JsonObject request = new JsonObject();
                    request.addProperty("model", model);
                    request.addProperty("keep_alive", "15m");
                    JsonObject options = new JsonObject();
                    options.addProperty("num_ctx", numCtx);
                    // Ollama's runner also restarts when num_thread differs from the brain's requests.
                    if (numThread > 0) options.addProperty("num_thread", numThread);
                    request.add("options", options);
                    http.send(HttpRequest.newBuilder(base.resolve("/api/generate"))
                            .timeout(Duration.ofSeconds(120)).header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(request.toString())).build(),
                            HttpResponse.BodyHandlers.discarding());
                } catch (Exception ignored) {}
            });
        }
        if (isLfmFamily(model)) {
            preloadLfmFamily();
        }
    }

    public static boolean isLfmFamily(String model) {
        if (model == null) return false;
        String lower = model.toLowerCase(java.util.Locale.ROOT);
        return lower.contains("lfm") || lower.contains("d1");
    }

    /** Concurrently preloads the entire LFM3 family so all models run simultaneously. */
    public void preloadLfmFamily() {
        for (String familyModel : LFM_FAMILY_MODELS) {
            Thread.startVirtualThread(() -> {
                try (var http = client()) {
                    JsonObject request = new JsonObject();
                    request.addProperty("model", familyModel);
                    request.addProperty("keep_alive", "24h");
                    http.send(HttpRequest.newBuilder(base.resolve("/api/generate"))
                            .timeout(Duration.ofSeconds(120)).header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(request.toString())).build(),
                            HttpResponse.BodyHandlers.discarding());
                } catch (Exception ignored) {}
            });
        }
    }

    /** Frees every loaded model, then stops the service. Safe to call repeatedly; only the first call acts. */
    public void shutdownOnExit() {
        if (!stopped.compareAndSet(false, true)) return;
        if (Boolean.getBoolean("localai.testing") || Boolean.getBoolean("localai.clientSmoke")
                || Boolean.getBoolean("localai.clientCameraSmoke")) {
            if (!Boolean.getBoolean("localai.ollamaLifecycleTest")) return;
        }
        if (!stopOnExitEnabled(configFile())) {
            log("Leaving Ollama running (stopOllamaOnExit is false)");
            return;
        }
        int unloaded = unloadAll();
        boolean stoppedService = runSystemctl("stop", 10);
        log(stoppedService
                ? "Stopped Ollama (" + unloaded + " model(s) unloaded first)"
                : "Unloaded " + unloaded + " Ollama model(s); could not stop " + unit
                        + " (run scripts/setup-ollama-control.sh once to allow it)");
    }

    public int unloadAll() {
        int count = 0;
        // Unload any active STRATA processes
        try {
            var strataReq = HttpRequest.newBuilder(URI.create("http://127.0.0.1:8080/unload"))
                    .timeout(Duration.ofSeconds(2)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{}")).build();
            try (var http = client()) {
                http.send(strataReq, HttpResponse.BodyHandlers.discarding());
            }
        } catch (Exception ignored) {}
        try {
            new ProcessBuilder("pkill", "-f", "serve/server.py").start().waitFor(2, TimeUnit.SECONDS);
            new ProcessBuilder("pkill", "-f", "engine/strata").start().waitFor(2, TimeUnit.SECONDS);
            new ProcessBuilder("pkill", "-f", "engine/strata-vision").start().waitFor(2, TimeUnit.SECONDS);
        } catch (Exception ignored) {}

        try (var http = client()) {
            var response = http.send(HttpRequest.newBuilder(base.resolve("/api/ps"))
                    .timeout(Duration.ofSeconds(3)).GET().build(), HttpResponse.BodyHandlers.ofInputStream());
            List<String> models = new ArrayList<>();
            try (var body = response.body()) {
                if (response.statusCode() != 200) return 0;
                byte[] bytes = body.readNBytes(1_048_577);
                if (bytes.length > 1_048_576) return 0;
                JsonObject json = JsonParser.parseString(new String(bytes, StandardCharsets.UTF_8)).getAsJsonObject();
                if (!json.has("models") || !json.get("models").isJsonArray()) return 0;
                for (var value : json.getAsJsonArray("models")) {
                    var model = value.getAsJsonObject();
                    String name = model.has("model") ? model.get("model").getAsString()
                            : model.has("name") ? model.get("name").getAsString() : null;
                    if (OllamaModels.validName(name)) models.add(name);
                }
            }
            for (String name : models) {
                JsonObject request = new JsonObject();
                request.addProperty("model", name);
                request.addProperty("keep_alive", 0);
                var unload = http.send(HttpRequest.newBuilder(base.resolve("/api/generate"))
                        .timeout(Duration.ofSeconds(5)).header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(request.toString())).build(),
                        HttpResponse.BodyHandlers.discarding());
                if (unload.statusCode() == 200) count++;
            }
        } catch (IOException | RuntimeException e) {
            // Ollama already down or unreadable: nothing left to unload.
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return count;
    }

    boolean runSystemctl(String verb, int timeoutSeconds) {
        var command = new ArrayList<>(systemctl);
        command.add(verb);
        command.add(unit);
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return false;
            }
            return process.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    static boolean stopOnExitEnabled(Path config) {
        try {
            if (!Files.isRegularFile(config)) return true;
            JsonObject json = JsonParser.parseString(Files.readString(config)).getAsJsonObject();
            return !json.has("stopOllamaOnExit") || json.get("stopOllamaOnExit").getAsBoolean();
        } catch (IOException | RuntimeException e) {
            return true;
        }
    }

    private static Path configFile() {
        try {
            return net.fabricmc.loader.api.FabricLoader.getInstance().getConfigDir().resolve("localai/config.json");
        } catch (RuntimeException e) {
            return Path.of("config/localai/config.json");
        }
    }

    private static HttpClient client() {
        return HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    }

    private static void log(String message) {
        // Shutdown hooks may run after Log4j has stopped, so mirror to stderr.
        LocalAiMod.LOG.info(message);
        System.err.println("[localai] " + message);
    }
}
