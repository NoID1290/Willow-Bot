package local.minecraft.ai.client;

import java.util.List;
import local.minecraft.ai.LocalAiMod;
import local.minecraft.ai.OllamaModels;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;

public final class ModelPickerScreen extends Screen {
    private final MinecraftServer server;
    private List<String> models = List.of();
    private String selected = LocalAiMod.instance().selectedModel();
    private String message = "Loading installed local models from Ollama...";
    private int page;
    private boolean busy;
    private boolean requested;
    private long smokeStarted;

    @Override
    public void tick() {
        if (!Boolean.getBoolean("localai.clientSmoke") || server != null) return;
        if (smokeStarted == 0) smokeStarted = System.nanoTime();
        if (!busy && requested) {
            if (models.isEmpty() || children().size() < 6) throw new IllegalStateException("Client model picker failed: " + message);
            org.slf4j.LoggerFactory.getLogger("localai").info("CLIENT PICKER PASS: {} local models, {} widgets",
                    models.size(), children().size());
            minecraft.stop();
        } else if (System.nanoTime() - smokeStarted > java.time.Duration.ofSeconds(20).toNanos()) {
            throw new IllegalStateException("Client model picker timed out");
        }
    }

    public ModelPickerScreen(MinecraftServer server) {
        super(Component.literal("Choose Willow's AI"));
        this.server = server;
    }

    @Override
    protected void init() {
        rebuild();
        if (!requested) refresh();
    }

    private void refresh() {
        requested = true;
        busy = true;
        message = "Loading installed local models from Ollama...";
        rebuild();
        OllamaModels.installed().whenComplete((names, error) -> minecraft.execute(() -> {
            if (minecraft.gui.screen() != this) return;
            busy = false;
            if (error != null) {
                message = "Ollama unavailable: " + LocalAiMod.rootMessage(error);
            } else {
                models = names;
                page = 0;
                if (!models.contains(selected) || selected.equals("gemma4:12b-it-qat")) {
                    selected = models.stream().filter(m -> m.toLowerCase().contains("lfm") || m.toLowerCase().contains("strata")).findFirst()
                            .orElse(models.isEmpty() ? "" : models.getFirst());
                }
                message = models.isEmpty() ? "No local models installed. Download a model in Ollama, then Refresh."
                        : "Choose a model. Start enables cinematic spectator follow.";
            }
            rebuild();
        }));
    }

    private int pageSize() {
        return Math.clamp((height - 170) / 24, 1, 8);
    }

    private void rebuild() {
        clearWidgets();
        int left = width / 2 - 150;
        int size = pageSize();
        page = Math.clamp(page, 0, Math.max(0, (models.size() - 1) / size));
        for (int row = 0; row < size && page * size + row < models.size(); row++) {
            String model = models.get(page * size + row);
            String label = model;
            if (model.contains("lfm3-family") || model.contains("lfm3-all")) {
                label = "★ " + model + " [Omni+VL+Audio]";
            } else if (model.toLowerCase(java.util.Locale.ROOT).contains("strata")) {
                label = "⚡ " + model + " [STRATA Engine]";
            } else if (model.toLowerCase(java.util.Locale.ROOT).contains("lfm")) {
                label = "✦ " + model;
            } else if (model.toLowerCase(java.util.Locale.ROOT).contains("qwen")) {
                label = "◈ " + model;
            }
            var button = addRenderableWidget(Button.builder(Component.literal(
                    (model.equals(selected) ? "> " : "") + label), b -> {
                        selected = model;
                        rebuild();
                    }).bounds(left, 65 + row * 24, 300, 20).build());
            button.active = !busy;
        }
        int controls = 70 + size * 24;
        var previous = addRenderableWidget(Button.builder(Component.literal("<"), b -> {
            page--;
            rebuild();
        }).bounds(left, controls, 35, 20).build());
        previous.active = !busy && page > 0;
        var next = addRenderableWidget(Button.builder(Component.literal(">"), b -> {
            page++;
            rebuild();
        }).bounds(left + 40, controls, 35, 20).build());
        next.active = !busy && (page + 1) * size < models.size();
        var settings = addRenderableWidget(Button.builder(Component.literal("Settings..."), b ->
                minecraft.gui.setScreen(new WillowSettingsScreen(server, this, selected)))
                .bounds(left + 80, controls, 115, 20).build());
        settings.active = !busy && models.contains(selected);
        var refresh = addRenderableWidget(Button.builder(Component.literal("Refresh"), b -> refresh())
                .bounds(left + 200, controls, 100, 20).build());
        refresh.active = !busy;
        var start = addRenderableWidget(Button.builder(Component.literal("Start Willow"), b -> start())
                .bounds(left, controls + 26, 145, 20).build());
        start.active = !busy && models.contains(selected);
        addRenderableWidget(Button.builder(Component.literal("Not now"), b -> onClose())
                .bounds(left + 155, controls + 26, 145, 20).build());
    }

    private void start() {
        busy = true;
        message = "Starting Willow with " + selected + "...";
        rebuild();
        LocalAiMod.instance().startSelected(server, selected).whenComplete((status, error) -> minecraft.execute(() -> {
            if (minecraft.getSingleplayerServer() != server) return;
            if (error != null) {
                busy = false;
                message = LocalAiMod.rootMessage(error);
                if (minecraft.gui.screen() == this) rebuild();
                else if (minecraft.player != null) minecraft.gui.chatListener().handleSystemMessage(Component.literal(message), false);
            } else if (minecraft.gui.screen() == this) {
                minecraft.gui.setScreen(null);
            }
        }));
    }

    @Override
    public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float partialTick) {
        super.extractRenderState(graphics, mouseX, mouseY, partialTick);
        graphics.centeredText(font, title, width / 2, 18, 0xFFFFFFFF);
        graphics.textWithWordWrap(font, Component.literal(message), width / 2 - 150, 35, 300, 0xFFDDDDDD);
        graphics.centeredText(font, "/STOP restores your player. /START opens this picker.", width / 2, height - 23, 0xFFAAAAAA);
    }

    @Override
    public boolean isPauseScreen() { return false; }

    @Override
    public void onClose() { minecraft.gui.setScreen(null); }
}
