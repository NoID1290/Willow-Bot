package local.minecraft.ai;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import net.fabricmc.loader.api.FabricLoader;

/** Asks the brain's own modelprofile.py for the effective settings, so the GUI and brain never disagree. */
public final class ModelProfiles {
    private ModelProfiles() {}

    public static CompletableFuture<JsonObject> describe(String model, ModelSettings settings) {
        var result = new CompletableFuture<JsonObject>();
        Thread.startVirtualThread(() -> {
            try {
                if (!OllamaModels.validName(model)) throw new IOException("Invalid model name");
                settings.validate();
                OllamaService.get().ensureRunning();
                Path runtime = FabricLoader.getInstance().getConfigDir().resolve("localai/runtime");
                BrainBridge.extractScripts(runtime);
                Process process = new ProcessBuilder("python3", runtime.resolve("modelprofile.py").toString(),
                        "--model", model, "--settings", settings.toJson())
                        .redirectError(ProcessBuilder.Redirect.DISCARD).start();
                byte[] output;
                try (var stdout = process.getInputStream()) {
                    output = stdout.readNBytes(262_145);
                }
                if (!process.waitFor(20, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                    throw new IOException("Model profile timed out");
                }
                if (output.length > 262_144) throw new IOException("Model profile output too large");
                JsonObject json = JsonParser.parseString(new String(output, StandardCharsets.UTF_8)).getAsJsonObject();
                if (json.has("error")) throw new IOException(json.get("error").getAsString());
                result.complete(json);
            } catch (IOException | RuntimeException e) {
                result.completeExceptionally(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                result.completeExceptionally(e);
            }
        });
        return result;
    }
}
