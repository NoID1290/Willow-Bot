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
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;

public final class OllamaModels {
    private OllamaModels() {}

    public static CompletableFuture<List<String>> installed() {
        var result = new CompletableFuture<List<String>>();
        Thread.startVirtualThread(() -> {
            OllamaService.get().ensureRunning();
            List<String> models = new ArrayList<>();
            try (var http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build()) {
                try {
                    var request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:11434/api/tags"))
                            .timeout(Duration.ofSeconds(8)).GET().build();
                    var response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
                    try (var body = response.body()) {
                        if (response.statusCode() == 200) {
                            byte[] bytes = body.readNBytes(1_048_577);
                            if (bytes.length <= 1_048_576) {
                                models.addAll(parse(JsonParser.parseString(new String(bytes,
                                        java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject()));
                            }
                        }
                    }
                } catch (IOException | RuntimeException e) {
                    LocalAiMod.LOG.warn("Cannot list installed Ollama models: {}", e.getMessage());
                }

                // Check STRATA inference engine
                checkStrata(http, models);

                if (!models.isEmpty()) {
                    result.complete(models);
                } else {
                    result.completeExceptionally(new IOException("No local models found in Ollama or STRATA"));
                }
            } catch (Exception e) {
                LocalAiMod.LOG.error("Failed to query local models", e);
                result.completeExceptionally(e);
            }
        });
        return result;
    }

    static void checkStrata(HttpClient http, List<String> models) {
        boolean strataAvailable = false;
        // 1. Try querying STRATA HTTP OpenAI models endpoint
        try {
            var req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:8080/v1/models"))
                    .timeout(Duration.ofSeconds(2)).GET().build();
            var resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200 && resp.body() != null) {
                var json = JsonParser.parseString(resp.body()).getAsJsonObject();
                if (json.has("data") && json.get("data").isJsonArray()) {
                    for (var elem : json.getAsJsonArray("data")) {
                        if (elem.isJsonObject() && elem.getAsJsonObject().has("id")) {
                            String id = elem.getAsJsonObject().get("id").getAsString();
                            String mName = "strata:" + id;
                            if (!models.contains(mName)) models.add(mName);
                            strataAvailable = true;
                        }
                    }
                }
            }
        } catch (Exception ignored) {}

        // 2. Try STRATA /health endpoint
        if (!strataAvailable) {
            try {
                var req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:8080/health"))
                        .timeout(Duration.ofSeconds(1)).GET().build();
                var resp = http.send(req, HttpResponse.BodyHandlers.discarding());
                if (resp.statusCode() == 200) {
                    strataAvailable = true;
                }
            } catch (Exception ignored) {}
        }

        // 3. Check filesystem for local STRATA installation
        if (!strataAvailable) {
            if (Files.isDirectory(Path.of("/mnt/LINUX-NVME/STRATA/Strata-main"))
                    || Files.isDirectory(Path.of(System.getProperty("user.home"), "STRATA"))
                    || Files.isDirectory(Path.of(System.getProperty("user.home"), ".strata"))) {
                strataAvailable = true;
            }
        }

        if (strataAvailable && !models.contains("strata:latest")) {
            models.add("strata:latest");
        }
    }

    public static String resolveModel(List<String> models, String requested) {
        if (models == null || models.isEmpty() || requested == null || requested.isBlank()) {
            return requested;
        }
        if (models.contains(requested)) {
            return requested;
        }
        String reqLower = requested.toLowerCase(java.util.Locale.ROOT).trim();
        // Exact case-insensitive match
        for (String m : models) {
            if (m.equalsIgnoreCase(reqLower)) return m;
        }
        // Match base name without tag
        for (String m : models) {
            String mBase = m.contains(":") ? m.substring(0, m.indexOf(':')) : m;
            if (mBase.equalsIgnoreCase(reqLower)) return m;
        }
        // Prefix match
        for (String m : models) {
            if (m.toLowerCase(java.util.Locale.ROOT).startsWith(reqLower)) return m;
        }
        // Substring match
        for (String m : models) {
            if (m.toLowerCase(java.util.Locale.ROOT).contains(reqLower)) return m;
        }
        return requested;
    }

    static List<String> parse(JsonObject response) throws IOException {
        if (!response.has("models") || !response.get("models").isJsonArray()) {
            throw new IOException("Ollama response is missing its model list");
        }
        var names = new TreeSet<String>((a, b) -> {
            String aLower = a.toLowerCase(java.util.Locale.ROOT);
            String bLower = b.toLowerCase(java.util.Locale.ROOT);
            boolean aFam = aLower.contains("lfm3-family") || aLower.contains("lfm3-all");
            boolean bFam = bLower.contains("lfm3-family") || bLower.contains("lfm3-all");
            if (aFam && !bFam) return -1;
            if (!aFam && bFam) return 1;
            boolean aStrata = aLower.contains("strata");
            boolean bStrata = bLower.contains("strata");
            if (aStrata && !bStrata) return -1;
            if (!aStrata && bStrata) return 1;
            boolean aLfm = aLower.contains("lfm") || aLower.contains("d1");
            boolean bLfm = bLower.contains("lfm") || bLower.contains("d1");
            if (aLfm && !bLfm) return -1;
            if (!aLfm && bLfm) return 1;
            return a.compareToIgnoreCase(b);
        });
        boolean hasAnyLfm = false;
        for (var value : response.getAsJsonArray("models")) {
            var model = value.getAsJsonObject();
            String name = model.get("name").getAsString();
            boolean isLfm = name.toLowerCase(java.util.Locale.ROOT).contains("lfm")
                    || name.toLowerCase(java.util.Locale.ROOT).contains("d1");
            if (isLfm) hasAnyLfm = true;
            if (model.has("capabilities")) {
                boolean hasSupportedCap = model.getAsJsonArray("capabilities").asList().stream()
                        .anyMatch(cap -> {
                            String c = cap.getAsString();
                            return c.equals("completion") || c.equals("vision") || c.equals("audio")
                                     || c.equals("multimodal") || c.equals("tools");
                        });
                if (!hasSupportedCap && !isLfm) continue;
            }
            if (validName(name) && !model.has("remote_host") && !model.has("remote_model")) names.add(name);
        }
        if (hasAnyLfm && !names.contains("lfm3-family:latest")) {
            names.add("lfm3-family:latest");
        }
        return List.copyOf(names);
    }

    public static boolean validName(String name) {
        if (name == null || name.isBlank()) return false;
        String lower = name.toLowerCase(java.util.Locale.ROOT);
        if (lower.equals("strata") || lower.startsWith("strata:") || lower.startsWith("strata/")) return true;
        return name.matches("[a-zA-Z0-9][a-zA-Z0-9_./:-]{0,199}")
                && !lower.contains("cloud");
    }
}
