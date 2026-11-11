package local.minecraft.ai;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

final class BrainBridge implements AutoCloseable {
    static final class DecisionRejectedException extends IOException {
        DecisionRejectedException(String detail) { super(detail); }
    }
    private final Path script;
    private final Path directory;
    private final String model;
    private final String settings;
    private final String token = UUID.randomUUID().toString() + UUID.randomUUID();
    private final ExecutorService executor = Executors.newSingleThreadExecutor(Thread.ofVirtual().factory());
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    private volatile Process process;
    private volatile boolean closed;
    private volatile Path portFile;
    private static String pythonBinary;

    static synchronized String getPythonBinary() {
        if (pythonBinary != null) return pythonBinary;
        String env = System.getenv("LOCALAI_PYTHON");
        if (env != null && !env.isBlank() && canRun(env)) {
            pythonBinary = env;
            return pythonBinary;
        }
        if (canRun("python3")) {
            pythonBinary = "python3";
            return pythonBinary;
        }
        if (canRun("python")) {
            pythonBinary = "python";
            return pythonBinary;
        }
        pythonBinary = "python3";
        return pythonBinary;
    }

    private static boolean canRun(String binary) {
        try {
            Process p = new ProcessBuilder(binary, "--version").start();
            return p.waitFor(2, java.util.concurrent.TimeUnit.SECONDS) && p.exitValue() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    BrainBridge(Path script, Path directory, String model) {
        this(script, directory, model, "{}");
    }

    BrainBridge(Path script, Path directory, String model, String settings) {
        this.script = script;
        this.directory = directory;
        this.model = model;
        this.settings = settings;
    }

    private static final String[] BRAIN_RESOURCES = new String[]{
            "server.py",
            "experience.py",
            "modelprofile.py",
            "monitor.py",
            "memory.py",
            "multimodal.py",
            "orchestrator.py",
            "hardware_opt/__init__.py",
            "hardware_opt/llm_client.py",
            "hardware_opt/telemetry_formatter.py",
            "hardware_opt/memory_pool.py",
            "rl_rewards/__init__.py",
            "rl_rewards/telemetry_snapshot.py",
            "rl_rewards/reward_calculator.py",
            "rl_rewards/q_table.py",
            "rl_rewards/reflexion_log.py",
            "curriculum/__init__.py",
            "curriculum/advancement_dag.json",
            "curriculum/mechanics_registry.py",
            "curriculum/curriculum_dag.py",
            "curriculum/curriculum_manager.py",
            "execution/__init__.py",
            "execution/pathfinder.py",
            "execution/bot_actions.py",
            "execution/sandbox.py",
            "skills/__init__.py",
            "skills/legacy_skills.py",
            "skills/skill_store.py",
            "skills/critic.py",
            "skills/code_agent.py"
    };

    static void extractScripts(Path directory) throws IOException {
        Files.createDirectories(directory);
        Files.deleteIfExists(directory.resolve("skills.py"));
        for (String name : BRAIN_RESOURCES) {
            try (var resource = BrainBridge.class.getResourceAsStream("/localai/brain/" + name)) {
                if (resource == null) throw new IOException("Packaged brain/" + name + " missing");
                Path target = directory.resolve(name);
                Files.createDirectories(target.getParent());
                Files.copy(resource, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private synchronized void cleanupStartupFailure() {
        if (process != null) {
            process.destroy();
            try {
                if (!process.waitFor(1, java.util.concurrent.TimeUnit.SECONDS)) process.destroyForcibly();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
            process = null;
        }
        cleanupStalePortFiles();
    }

    private void cleanupStalePortFiles() {
        try {
            if (portFile != null) {
                Files.deleteIfExists(portFile);
                portFile = null;
            }
            if (Files.isDirectory(directory)) {
                try (var stream = Files.newDirectoryStream(directory, "port-*.txt")) {
                    for (Path p : stream) {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException ignored) {}
                    }
                }
            }
        } catch (IOException ignored) {}
    }

    private synchronized Path ensureRunning() throws IOException {
        if (closed) throw new IOException("Brain bridge is closed");
        if (process != null && process.isAlive()) return portFile;
        if (process != null) LocalAiMod.LOG.error("Brain process exited with code {}; restarting", process.exitValue());
        if (!OllamaService.get().ensureRunning()) LocalAiMod.LOG.warn("Ollama is not answering; Willow's brain will retry");
        try {
            cleanupStalePortFiles();
            Files.createDirectories(directory);
            portFile = directory.resolve("port-" + UUID.randomUUID() + ".txt");
            ProcessBuilder builder = new ProcessBuilder(getPythonBinary(), script.toString(),
                    "--database", directory.resolve("brain.sqlite3").toString(),
                    "--port-file", portFile.toString(), "--model", model, "--settings", settings);
            builder.environment().put("LOCALAI_TOKEN", token);
            builder.redirectErrorStream(true);
            builder.redirectOutput(directory.resolve("brain-startup.log").toFile());
            process = builder.start();
            return portFile;
        } catch (Exception e) {
            cleanupStartupFailure();
            throw e;
        }
    }

    CompletableFuture<JsonObject> decide(JsonObject observation) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                CompletableFuture<String> snapFuture = SightPerception.getPendingSnapshot();
                if (snapFuture != null) {
                    try {
                        String b64 = snapFuture.get(100, TimeUnit.MILLISECONDS);
                        if (b64 != null && observation.has("sight") && observation.get("sight").isJsonObject()) {
                            observation.getAsJsonObject("sight").addProperty("snapshot_png_base64", b64);
                        }
                    } catch (Exception ignored) {
                    }
                }
                String body = observation.toString();
                Path file = ensureRunning();
                for (int i = 0; !Files.exists(file) && i < 100; i++) {
                    if (Thread.currentThread().isInterrupted()) throw new InterruptedException();
                    if (process != null && !process.isAlive()) {
                        cleanupStartupFailure();
                        throw new IOException("Brain process exited with code " + process.exitValue()
                                + " during startup. See " + directory.resolve("brain-startup.log"));
                    }
                    Thread.sleep(50);
                }
                if (!Files.exists(file)) {
                    cleanupStartupFailure();
                    throw new IOException("Brain failed to create port file within 5 seconds. See "
                            + directory.resolve("brain-startup.log"));
                }
                int port = Integer.parseInt(Files.readString(file).trim());
                HttpRequest request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/decide"))
                        .timeout(Duration.ofSeconds(250))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + token)
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build();
                var response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
                try (var input = response.body()) {
                    byte[] bytes = input.readNBytes(65537);
                    return decodeResponse(response.statusCode(), bytes);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new CompletionException(e);
            } catch (IOException | IllegalArgumentException | IllegalStateException e) {
                throw new CompletionException(e);
            }
        }, executor);
    }

    static JsonObject decodeResponse(int status, byte[] bytes) throws IOException {
        if (bytes.length > 65536) throw new IOException("Oversized planner response");
        if (status == 422) {
            JsonObject rejection = JsonParser.parseString(new String(bytes,
                    java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject();
            if (rejection.has("error") && rejection.get("error").getAsString().equals("decision_rejected")
                    && rejection.has("detail") && rejection.get("detail").isJsonPrimitive()
                    && rejection.get("detail").getAsJsonPrimitive().isString()) {
                throw new DecisionRejectedException(rejection.get("detail").getAsString());
            }
        }
        if (status != 200) throw new IOException("Planner HTTP " + status);
        return JsonParser.parseString(new String(bytes, java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject();
    }

    @Override
    public synchronized void close() {
        closed = true;
        executor.shutdownNow();
        http.shutdownNow();
        if (process != null) {
            process.destroy();
            try {
                if (!process.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) process.destroyForcibly();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
            process = null;
        }
        cleanupStalePortFiles();
    }
}
