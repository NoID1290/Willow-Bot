package local.minecraft.ai.client;

import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import local.minecraft.ai.LocalAiMod;
import local.minecraft.ai.ModelProfiles;
import local.minecraft.ai.ModelSettings;
import local.minecraft.ai.OllamaConsole;
import local.minecraft.ai.OllamaModels;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.EditBox;
import net.minecraft.client.gui.components.Tooltip;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;

/** Per-model brain settings: shows what was detected from Ollama and lets the player override it. */
public final class WillowSettingsScreen extends Screen {
    private static final String[] THINK = {"auto", "on", "off"};
    private final MinecraftServer server;
    private final Screen parent;
    private List<String> models = List.of();
    private String model;
    private ModelSettings settings;
    private JsonObject profile;
    private String message = "Reading installed models from Ollama...";
    private boolean busy;
    private long previewRequest;
    private EditBox context;
    private EditBox actor;
    private EditBox plan;
    private EditBox temperature;

    public WillowSettingsScreen(MinecraftServer server, Screen parent, String model) {
        super(Component.literal("Willow AI settings"));
        this.server = server;
        this.parent = parent;
        this.model = model;
        this.settings = LocalAiMod.instance().modelSettings(model);
    }

    String model() { return model; }
    JsonObject profile() { return profile; }
    boolean busy() { return busy; }

    @Override
    protected void init() {
        boolean first = context == null;
        rebuild();
        if (first) {
            OllamaModels.installed().whenComplete((names, error) -> minecraft.execute(() -> {
                if (error != null) {
                    message = "Ollama unavailable: " + LocalAiMod.rootMessage(error);
                } else {
                    models = names;
                    if (!models.contains(model) && !models.isEmpty()) select(models.getFirst());
                }
                preview();
            }));
        }
    }

    private void select(String next) {
        model = next;
        settings = LocalAiMod.instance().modelSettings(next);
        profile = null;
        rebuild();
        preview();
    }

    private static String text(Number value) {
        return value == null ? "" : value instanceof Double d && d == Math.rint(d) ? String.valueOf(d) : value.toString();
    }

    private EditBox field(int x, int y, String label, String value, String tip) {
        var box = new EditBox(font, x, y, 60, 18, Component.literal(label));
        box.setMaxLength(7);
        box.setValue(value);
        box.setTooltip(Tooltip.create(Component.literal(tip)));
        return addRenderableWidget(box);
    }

    private void rebuild() {
        String contextValue = context == null ? text(settings.contextLength) : context.getValue();
        String actorValue = actor == null ? text(settings.actorTokens) : actor.getValue();
        String planValue = plan == null ? text(settings.planTokens) : plan.getValue();
        String temperatureValue = temperature == null ? text(settings.temperature) : temperature.getValue();
        clearWidgets();
        int center = width / 2;
        int top = top();
        addRenderableWidget(Button.builder(Component.literal("<"), b -> cycle(-1))
                .bounds(center - 210, top, 20, 20).build()).active = !busy && models.size() > 1;
        String running = model.equals(LocalAiMod.instance().runningModel()) ? "  (Willow uses this)" : "";
        addRenderableWidget(Button.builder(Component.literal(model + running), b -> cycle(1))
                .bounds(center - 186, top, 372, 20).build()).active = !busy && models.size() > 1;
        addRenderableWidget(Button.builder(Component.literal(">"), b -> cycle(1))
                .bounds(center + 190, top, 20, 20).build()).active = !busy && models.size() > 1;

        int x = center + 100;
        int y = top + 28;
        context = field(x, y, "Context length", contextValue,
                "Tokens of context (num_ctx). Blank = auto: the model's trained length, capped at 32768.");
        actor = field(x, y + 22, "Actor output", actorValue,
                "Max tokens per action decision. Thinking models need room to reason; others answer in ~100.");
        plan = field(x, y + 44, "Strategist output", planValue,
                "Max tokens for the background strategist that writes goals and skill code.");
        temperature = field(x, y + 66, "Temperature", temperatureValue,
                "Sampling temperature 0-2. Blank = the model's own recommended default.");
        String think = settings.think == null ? "auto" : settings.think;
        var thinking = addRenderableWidget(Button.builder(Component.literal("Thinking: " + think), b -> {
            int index = java.util.Arrays.asList(THINK).indexOf(settings.think == null ? "auto" : settings.think);
            settings.think = THINK[(index + 1) % THINK.length];
            rebuild();
            preview();
        }).bounds(center + 5, y + 88, 100, 20).build());
        thinking.setTooltip(Tooltip.create(Component.literal(
                "auto/on: reason before acting when the model supports it. off: faster, less careful.")));
        String vis = settings.vision == null ? "auto" : settings.vision;
        var visionBtn = addRenderableWidget(Button.builder(Component.literal("Vision: " + vis), b -> {
            int index = java.util.Arrays.asList(THINK).indexOf(settings.vision == null ? "auto" : settings.vision);
            settings.vision = THINK[(index + 1) % THINK.length];
            rebuild();
            preview();
        }).bounds(center + 110, y + 88, 100, 20).build());
        visionBtn.setTooltip(Tooltip.create(Component.literal(
                "auto/on: send first-person camera snapshots when the model supports vision. off: text-only.")));
        for (var box : List.of(context, actor, plan, temperature)) box.setResponder(value -> previewSoon());

        int buttons = y + 156;
        addRenderableWidget(Button.builder(Component.literal("Reset to auto"), b -> {
            settings = new ModelSettings();
            context = actor = plan = temperature = null;
            rebuild();
            preview();
        }).bounds(center - 210, buttons, 136, 20).build()).active = !busy;
        String saveLabel = model.equals(LocalAiMod.instance().runningModel()) ? "Save & apply" : "Save";
        addRenderableWidget(Button.builder(Component.literal(saveLabel), b -> save())
                .bounds(center - 68, buttons, 136, 20).build()).active = !busy && server != null;
        addRenderableWidget(Button.builder(Component.literal("Done"), b -> onClose())
                .bounds(center + 74, buttons, 136, 20).build());
        addRenderableWidget(Button.builder(Component.literal("Open Ollama console + Willow log"), b ->
                message = OllamaConsole.open(LocalAiMod.instance().brainLog()))
                .bounds(center - 210, buttons + 24, 420, 20).build());
    }

    private int top() {
        return Math.max(22, Math.min(height / 2 - 118, height - 232));
    }

    private void cycle(int direction) {
        if (models.isEmpty()) return;
        int index = Math.max(0, models.indexOf(model));
        select(models.get(Math.floorMod(index + direction, models.size())));
    }

    private ModelSettings parse() {
        var parsed = new ModelSettings();
        parsed.contextLength = integer(context.getValue(), "Context length");
        parsed.actorTokens = integer(actor.getValue(), "Actor output");
        parsed.planTokens = integer(plan.getValue(), "Strategist output");
        String temp = temperature.getValue().trim();
        try {
            parsed.temperature = temp.isEmpty() ? null : Double.valueOf(temp);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Temperature must be a number");
        }
        parsed.think = settings.think;
        parsed.vision = settings.vision;
        parsed.validate();
        return parsed;
    }

    private static Integer integer(String value, String label) {
        value = value.trim();
        if (value.isEmpty()) return null;
        try {
            return Integer.valueOf(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(label + " must be a whole number");
        }
    }

    private int previewDelay;

    private void previewSoon() {
        previewDelay = 12;
    }

    @Override
    public void tick() {
        if (previewDelay > 0 && --previewDelay == 0) preview();
    }

    private void preview() {
        ModelSettings parsed;
        try {
            parsed = parse();
        } catch (IllegalArgumentException e) {
            message = e.getMessage();
            return;
        }
        long request = ++previewRequest;
        message = "Reading " + model + " architecture from Ollama...";
        String wanted = model;
        ModelProfiles.describe(wanted, parsed).whenComplete((json, error) -> minecraft.execute(() -> {
            if (request != previewRequest || !wanted.equals(model)) return;
            if (error != null) {
                profile = null;
                message = LocalAiMod.rootMessage(error);
            } else {
                profile = json;
                message = "Blank fields use the auto values shown on the left.";
            }
        }));
    }

    private void save() {
        ModelSettings parsed;
        try {
            parsed = parse();
        } catch (IllegalArgumentException e) {
            message = e.getMessage();
            return;
        }
        busy = true;
        rebuild();
        String wanted = model;
        // Check against the real model first so an impossible combination is never saved.
        ModelProfiles.describe(wanted, parsed)
                .thenCompose(json -> LocalAiMod.instance().saveModelSettings(server, wanted, parsed))
                .whenComplete((result, error) -> minecraft.execute(() -> {
                    busy = false;
                    message = error != null ? "Not saved: " + LocalAiMod.rootMessage(error) : result;
                    if (error == null) settings = parsed;
                    if (minecraft.gui.screen() == this) rebuild();
                    preview();
                }));
    }

    private List<String> describe() {
        var lines = new ArrayList<String>();
        if (profile == null) return lines;
        var p = profile;
        lines.add("§e" + p.get("architecture").getAsString() + "§r  " + p.get("parameter_size").getAsString()
                + "  " + p.get("quantization").getAsString());
        int experts = p.get("moe_experts").getAsInt();
        lines.add(experts > 0 ? "MoE: " + experts + " experts, " + p.get("moe_active_experts").getAsInt() + " active"
                : "Dense model, " + p.get("layers").getAsInt() + " layers");
        lines.add("Trained context: " + p.get("trained_context").getAsInt());
        lines.add("Embedding size: " + p.get("embedding_length").getAsInt());
        var caps = new ArrayList<String>();
        p.getAsJsonArray("capabilities").forEach(c -> caps.add(c.getAsString()));
        lines.add("Can: " + String.join(", ", caps));
        lines.add("§aUsing context " + p.get("context").getAsInt() + "§r (auto " + p.get("auto_context").getAsInt() + ")");
        lines.add("KV cache ~" + p.get("kv_cache_mib").getAsInt() + " MiB");
        lines.add("Thinking: " + (p.get("thinking").getAsBoolean() ? "on" : "off")
                + (p.get("thinking_supported").getAsBoolean() ? "" : " (unsupported)")
                + "  Vision: " + (p.has("vision_enabled") && p.get("vision_enabled").getAsBoolean() ? "on" : "off"));
        lines.add("Output: actor " + p.get("actor_tokens").getAsInt() + ", strategist " + p.get("plan_tokens").getAsInt());
        lines.add("Prompt budget: " + p.get("prompt_bytes").getAsInt() + " bytes");
        var options = p.getAsJsonObject("options");
        var defaults = p.getAsJsonObject("model_defaults");
        lines.add("Temperature: " + (options.has("temperature") ? options.get("temperature").getAsString()
                : defaults.has("temperature") ? defaults.get("temperature").getAsString() + " (model)" : "default"));
        return lines;
    }

    @Override
    public void extractRenderState(GuiGraphicsExtractor graphics, int mouseX, int mouseY, float partialTick) {
        super.extractRenderState(graphics, mouseX, mouseY, partialTick);
        int center = width / 2;
        int top = top();
        graphics.centeredText(font, title, center, top - 16, 0xFFFFFFFF);
        int y = top + 28;
        graphics.text(font, "Detected from Ollama", center - 210, y - 2, 0xFFFFFF80, true);
        int line = y + 10;
        for (String text : describe()) {
            graphics.text(font, Component.literal(font.plainSubstrByWidth(text, 205)), center - 210, line, 0xFFDDDDDD, true);
            line += 10;
        }
        String[] labels = {"Context length", "Actor output", "Strategist output", "Temperature"};
        String[] autos = profile == null ? new String[]{"", "", "", ""} : new String[]{
                "auto " + profile.get("auto_context").getAsInt(),
                "now " + profile.get("actor_tokens").getAsInt(),
                "now " + profile.get("plan_tokens").getAsInt(), "auto"};
        for (int i = 0; i < labels.length; i++) {
            graphics.text(font, labels[i], center + 5, y + 5 + i * 22, 0xFFFFFFFF, true);
            if (!autos[i].isEmpty()) graphics.text(font, "§7" + autos[i], center + 165, y + 5 + i * 22, 0xFFAAAAAA, true);
        }
        graphics.textWithWordWrap(font, Component.literal(message), center - 210, y + 126, 420, 0xFFDDDDDD);
    }

    @Override
    public boolean isPauseScreen() { return false; }

    @Override
    public void onClose() { minecraft.gui.setScreen(parent); }
}
