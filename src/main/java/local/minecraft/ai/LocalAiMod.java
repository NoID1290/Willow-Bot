package local.minecraft.ai;

import carpet.patches.EntityPlayerMPFake;
import carpet.patches.FakeClientConnection;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.authlib.GameProfile;
import com.mojang.brigadier.arguments.StringArgumentType;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.message.v1.ServerMessageEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.server.permissions.Permissions;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.storage.LevelResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class LocalAiMod implements ModInitializer {
    private static LocalAiMod instance;
    public static LocalAiMod instance() { return instance; }
    static final Logger LOG = LoggerFactory.getLogger("localai");
    static final String NAME = "Willow_AI";
    static final UUID PLAYER_ID = UUID.nameUUIDFromBytes(("OfflinePlayer:" + NAME).getBytes(StandardCharsets.UTF_8));
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private Config config;
    private volatile MinecraftServer server;
    private ServerPlayer player;
    private Actions actions;
    private volatile BrainBridge bridge;
    private CompletableFuture<JsonObject> pending;
    private final List<String> chat = new ArrayList<>();
    private int sentChat;
    private long nextThink;
    private int failures;
    private int rejectedPlans;
    private volatile boolean paused;
    private boolean discardPending;
    private boolean died;
    private volatile String status = "Not in a world";
    private volatile boolean selectingModel;
    private volatile long selectionRequest;
    private volatile String selectedModel = "gemma4:12b-it-qat";
    private long lifecycle;
    private boolean starting;
    private final CameraDirector director = new CameraDirector();

    static final class Config {
        boolean enabled = true;
        boolean allowDedicatedServer = false;
        String model = "gemma4:12b-it-qat";
        int thinkIntervalTicks = 0;
        boolean chooseModelOnStart = true;
        boolean cinematicCamera = true;
        boolean stopOllamaOnExit = true;
        java.util.concurrent.ConcurrentSkipListMap<String, ModelSettings> modelSettings =
                new java.util.concurrent.ConcurrentSkipListMap<>();
    }

    @Override
    public void onInitialize() {
        instance = this;
        ServerLifecycleEvents.SERVER_STARTED.register(s -> {
            if (!Boolean.getBoolean("localai.testing")) started(s);
        });
        ServerLifecycleEvents.SERVER_STOPPING.register(s -> shutdown());
        ServerLifecycleEvents.SERVER_STOPPED.register(s -> {
            if (s.isDedicatedServer()) OllamaService.get().shutdownOnExit();
        });
        Runtime.getRuntime().addShutdownHook(new Thread(() -> OllamaService.get().shutdownOnExit(),
                "localai-ollama-shutdown"));
        ServerTickEvents.END_SERVER_TICK.register(this::tick);
        ServerLivingEntityEvents.AFTER_DEATH.register((entity, damageSource) -> {
            if (entity == player) {
                died = true;
                discardPending = pending != null;
                if (pending != null) pending.cancel(true);
                if (actions != null) actions.stop("Died from " + damageSource.getMsgId(), true);
            }
        });
        ServerMessageEvents.CHAT_MESSAGE.register((message, sender, params) -> {
            MinecraftServer s = server;
            if (s != null) {
                s.execute(() -> {
                    if (player == null || bridge == null || paused) return;
                    String text = message.signedContent();
                    if (text == null || text.isBlank()) return;
                    String lower = text.toLowerCase();
                    if (lower.contains("willow") || lower.contains("@willow")) {
                        queueChatMessage(text, sender.getGameProfile().name(), null);
                    }
                });
            }
        });
        ServerMessageEvents.COMMAND_MESSAGE.register((message, source, params) -> {
            MinecraftServer s = server;
            if (s != null) {
                s.execute(() -> {
                    if (player == null || bridge == null || paused) return;
                    String text = message.signedContent();
                    if (text == null || text.isBlank()) return;
                    String senderName = source.getTextName();
                    if (NAME.equalsIgnoreCase(senderName)) return;
                    if (params.targetName().isPresent()) {
                        String target = params.targetName().get().getString();
                        if (!NAME.equalsIgnoreCase(target)) return;
                    }
                    queueChatMessage(text, senderName, null);
                });
            }
        });
        CommandRegistrationCallback.EVENT.register((dispatcher, registry, environment) -> {
                dispatcher.register(Commands.literal("brain")
                        // Login command-tree serialization probes permissions without a server.
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .then(Commands.literal("status").executes(context -> {
                            context.getSource().sendSuccess(() -> Component.literal(status), false);
                            return 1;
                        }))
                        .then(Commands.literal("pause").executes(context -> {
                            paused = true;
                            discardPending = true;
                            if (pending != null) pending.cancel(true);
                            if (actions != null) actions.stop("Paused by player");
                            status = "Willow paused; /brain resume to continue";
                            context.getSource().sendSuccess(() -> Component.literal(status), false);
                            return 1;
                        }))
                        .then(Commands.literal("resume").executes(context -> {
                            if (bridge == null) {
                                context.getSource().sendFailure(Component.literal("Willow is stopped. Use /START."));
                                return 0;
                            }
                            paused = false;
                            nextThink = 0;
                            status = "Willow resumed";
                            context.getSource().sendSuccess(() -> Component.literal(status), false);
                            return 1;
                        }))
                        .then(Commands.literal("start")
                                .executes(context -> requestStart(context.getSource()))
                                .then(Commands.argument("model", StringArgumentType.greedyString())
                                        .executes(context -> startCommand(context.getSource(),
                                                StringArgumentType.getString(context, "model")))))
                        .then(Commands.literal("stop").executes(context -> stopCommand(context.getSource())))
                        .then(Commands.literal("model").executes(context -> requestStart(context.getSource())))
                        .then(Commands.literal("console").executes(context -> consoleCommand(context.getSource())))
                        .then(Commands.literal("settings").executes(context -> settingsCommand(context.getSource())))
                        .then(Commands.literal("camera")
                                .then(Commands.literal("on").executes(context -> cameraCommand(context.getSource(), true)))
                                .then(Commands.literal("off").executes(context -> cameraCommand(context.getSource(), false))))
                        .then(Commands.literal("say")
                                .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                                    String message = StringArgumentType.getString(context, "message");
                                    return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                                }))));
                dispatcher.register(Commands.literal("START")
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .executes(context -> requestStart(context.getSource()))
                        .then(Commands.argument("model", StringArgumentType.greedyString()).executes(
                                context -> startCommand(context.getSource(), StringArgumentType.getString(context, "model")))));
                dispatcher.register(Commands.literal("STOP")
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .executes(context -> stopCommand(context.getSource())));
                dispatcher.register(Commands.literal("SAY")
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                            String message = StringArgumentType.getString(context, "message");
                            return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                        })));
                dispatcher.register(Commands.literal("willow")
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .then(Commands.literal("say")
                                .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                                    String message = StringArgumentType.getString(context, "message");
                                    return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                                })))
                        .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                            String message = StringArgumentType.getString(context, "message");
                            return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                        })));
                dispatcher.register(Commands.literal("WILLOW")
                        .requires(source -> environment == Commands.CommandSelection.INTEGRATED
                                || source.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))
                        .then(Commands.literal("say")
                                .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                                    String message = StringArgumentType.getString(context, "message");
                                    return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                                })))
                        .then(Commands.argument("message", StringArgumentType.greedyString()).executes(context -> {
                            String message = StringArgumentType.getString(context, "message");
                            return queueChatMessage(message, context.getSource().getTextName(), context.getSource()) ? 1 : 0;
                        })));
                // Opening a desktop window only makes sense on the singleplayer host's own machine.
                if (environment == Commands.CommandSelection.INTEGRATED) {
                    dispatcher.register(Commands.literal("OLLAMA").executes(context -> consoleCommand(context.getSource())));
                }
        });
    }

    boolean queueChatMessage(String message, String senderName, CommandSourceStack source) {
        if (bridge == null || player == null) {
            if (source != null) {
                source.sendFailure(Component.literal("Willow is stopped. Use /START to start Willow."));
            }
            return false;
        }
        if (paused) {
            if (source != null) {
                source.sendFailure(Component.literal("Willow is paused. Use /brain resume to continue."));
            }
            return false;
        }
        if (message == null || message.isBlank()) return false;
        if (message.length() > 160) {
            message = message.substring(0, 160);
        }
        if (chat.size() >= 4) {
            if (source != null) {
                source.sendFailure(Component.literal("Willow's message queue is full (max 4 messages). Please wait."));
            }
            return false;
        }
        String entry = (senderName != null && !senderName.isBlank()) ? senderName + ": " + message : message;
        chat.add(entry);
        nextThink = 0;
        LOG.info("Message queued for Willow by {}: \"{}\"", senderName != null ? senderName : "player", message);
        if (source != null) {
            source.sendSuccess(() -> Component.literal("Message queued for Willow."), false);
        }
        if (actions != null && actions.busy()) {
            actions.stop("Listening to player");
            LOG.info("Interrupted current action to read player message immediately");
        }
        if (pending != null) {
            discardPending = true;
            pending.cancel(true);
            LOG.info("Discarding pending pre-chat plan to process player message immediately");
        }
        return true;
    }

    void started(MinecraftServer minecraftServer) {
        server = minecraftServer;
        try {
            Path root = FabricLoader.getInstance().getConfigDir().resolve("localai");
            Files.createDirectories(root);
            Path file = root.resolve("config.json");
            if (Files.exists(file)) {
                config = GSON.fromJson(Files.readString(file), Config.class);
            } else {
                config = new Config();
                Files.writeString(file, GSON.toJson(config));
            }
            if (config == null || config.model == null || config.model.isBlank()
                    || config.model.contains("cloud") || config.thinkIntervalTicks < 0
                    || config.thinkIntervalTicks > 1200) throw new IOException("Invalid localai/config.json");
            selectedModel = config.model;
            if (config.modelSettings == null) config.modelSettings = new java.util.concurrent.ConcurrentSkipListMap<>();
            config.modelSettings.values().removeIf(java.util.Objects::isNull);
            for (var entry : config.modelSettings.entrySet()) {
                try {
                    entry.getValue().validate();
                } catch (IllegalArgumentException e) {
                    LOG.warn("Ignoring invalid settings for {}: {}", entry.getKey(), e.getMessage());
                    entry.setValue(new ModelSettings());
                }
            }
            director.load(server);
            if (!config.enabled || (server.isDedicatedServer() && !config.allowDedicatedServer)) {
                status = "Willow disabled by configuration";
                return;
            }
            if (!server.isDedicatedServer() && config.chooseModelOnStart) {
                requestSelection();
            } else {
                startSelected(server, config.model);
            }
        } catch (IOException | RuntimeException e) {
            LOG.error("Cannot start local AI player", e);
            status = "Willow startup failed: " + e.getMessage();
            if (bridge != null) bridge.close();
            bridge = null;
        }
    }

    public boolean wantsSelection(MinecraftServer current) {
        return server == current && selectingModel;
    }

    public long selectionRequest() { return selectionRequest; }
    public String selectedModel() { return selectedModel; }

    private void requestSelection() {
        selectingModel = true;
        selectionRequest++;
        status = "Choose Willow's local model, or /START <model>";
    }

    private int requestStart(net.minecraft.commands.CommandSourceStack source) {
        if (server == null || config == null) {
            source.sendFailure(Component.literal("Willow's controller is unavailable."));
            return 0;
        }
        if (bridge != null || starting) {
            source.sendFailure(Component.literal("Willow is already running or starting. /STOP before changing models."));
            return 0;
        }
        if (server.isDedicatedServer()) return startCommand(source, config.model);
        requestSelection();
        source.sendSuccess(() -> Component.literal("Model picker requested. You can also use /START <model>."), false);
        return 1;
    }

    private int startCommand(net.minecraft.commands.CommandSourceStack source, String model) {
        startSelected(source.getServer(), model).whenComplete((message, error) -> source.getServer().execute(() -> {
            if (error != null) source.sendFailure(Component.literal("Willow could not start: " + rootMessage(error)));
            else source.sendSuccess(() -> Component.literal(message), false);
        }));
        return 1;
    }

    public CompletableFuture<String> startSelected(MinecraftServer current, String model) {
        var result = new CompletableFuture<String>();
        current.execute(() -> {
            if (server != current || config == null || (current.isDedicatedServer() && !config.allowDedicatedServer)) {
                result.completeExceptionally(new IllegalStateException("Willow is unavailable or dedicated-server use is disabled"));
                return;
            }
            if (bridge != null || starting || !OllamaModels.validName(model)) {
                result.completeExceptionally(new IllegalStateException("Already running/starting, or invalid local model name"));
                return;
            }
            starting = true;
            long request = ++lifecycle;
            OllamaModels.installed().whenComplete((models, error) -> current.execute(() -> {
                if (server != current || lifecycle != request) {
                    result.completeExceptionally(new IllegalStateException("Start cancelled"));
                    return;
                }
                starting = false;
                try {
                    if (error != null) throw new IOException("Cannot read local Ollama models: " + rootMessage(error), error);
                    String resolved = OllamaModels.resolveModel(models, model);
                    if (!models.contains(resolved)) throw new IOException("Model is not installed locally: " + model);
                    launch(resolved);
                    selectingModel = false;
                    result.complete(status);
                } catch (IOException | RuntimeException e) {
                    LOG.error("Cannot start Willow", e);
                    stopAi("Willow startup failed: " + e.getMessage());
                    result.completeExceptionally(e);
                }
            }));
        });
        return result;
    }

    void launch(String model) throws IOException {
        Path root = FabricLoader.getInstance().getConfigDir().resolve("localai");
        BrainBridge.extractScripts(root.resolve("runtime"));
        config.model = model;
        saveConfig();
        selectedModel = model;
        bridge = newBridge(model);
        int[] runner = savedProfileRunnerOptions(model);
        OllamaService.get().preload(model, runner[0], runner[1]);
        SightPerception.setSnapshotEnabled(true);
        player = spawn(server);
        AcousticSensor.setPlayer(player);
        actions = new Actions(player);
        died = false;
        paused = false;
        discardPending = false;
        failures = 0;
        rejectedPlans = 0;
        nextThink = server.getTickCount() + 40L;
        status = "Willow starting with " + model;
        if (OllamaService.isLfmFamily(model)) {
            server.getPlayerList().broadcastSystemMessage(
                    Component.literal("Willow is starting with " + model + ". Running LFM3 Multimodal Family concurrently (Vision, Audio, Omni & Planning active simultaneously)..."), false);
        } else if (model.toLowerCase(java.util.Locale.ROOT).contains("strata")) {
            server.getPlayerList().broadcastSystemMessage(
                    Component.literal("Willow is starting with " + model + ". High-performance STRATA Engine active with Sight & Acoustic perception!"), false);
        } else {
            server.getPlayerList().broadcastSystemMessage(
                    Component.literal("Willow is starting with " + model + ". Sensory systems active (Sight & Acoustic perception enabled). Pre-warming model..."), false);
        }
    }

    private void saveConfig() throws IOException {
        Path file = FabricLoader.getInstance().getConfigDir().resolve("localai/config.json");
        Path temporary = file.resolveSibling("config.json.tmp");
        Files.writeString(temporary, GSON.toJson(config));
        Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
    }

    /** {num_ctx, num_thread} the brain used for this model in this world last time; zeros when unknown. */
    private int[] savedProfileRunnerOptions(String model) {
        Path profile = server.getWorldPath(LevelResource.ROOT).resolve("localai/model-profile.json");
        try {
            if (!Files.isRegularFile(profile)) return new int[]{0, 0};
            JsonObject saved = JsonParser.parseString(Files.readString(profile)).getAsJsonObject();
            if (!saved.has("model") || !model.equals(saved.get("model").getAsString())) return new int[]{0, 0};
            JsonObject options = saved.has("options") ? saved.getAsJsonObject("options") : new JsonObject();
            int numCtx = options.has("num_ctx") ? options.get("num_ctx").getAsInt()
                    : saved.has("context") ? saved.get("context").getAsInt() : 0;
            int numThread = options.has("num_thread") ? options.get("num_thread").getAsInt() : 0;
            return new int[]{numCtx, numThread};
        } catch (IOException | RuntimeException e) {
            LOG.debug("No usable model profile for preload", e);
            return new int[]{0, 0};
        }
    }

    private int stopCommand(net.minecraft.commands.CommandSourceStack source) {
        stopAi("Willow stopped. Use /START to choose a model and reconnect.");
        source.sendSuccess(() -> Component.literal(status), false);
        return 1;
    }

    private int consoleCommand(net.minecraft.commands.CommandSourceStack source) {
        if (server == null || server.isDedicatedServer()) {
            source.sendFailure(Component.literal("The Ollama console opens on the singleplayer host only."));
            return 0;
        }
        String message = OllamaConsole.open(brainLog());
        source.sendSuccess(() -> Component.literal(message), false);
        return 1;
    }

    private int settingsCommand(net.minecraft.commands.CommandSourceStack source) {
        if (server == null || server.isDedicatedServer()) {
            source.sendFailure(Component.literal("The settings screen is for the singleplayer host. Edit config/localai/config.json."));
            return 0;
        }
        settingsRequest++;
        source.sendSuccess(() -> Component.literal("Opening Willow AI settings."), false);
        return 1;
    }

    private volatile long settingsRequest;

    public long settingsRequest() { return settingsRequest; }

    public Path brainLog() {
        MinecraftServer current = server;
        return current == null ? null : current.getWorldPath(LevelResource.ROOT).resolve("localai/brain.log");
    }

    /** The model Willow is currently thinking with, or null when stopped. */
    public String runningModel() {
        return bridge != null ? selectedModel : null;
    }

    public ModelSettings modelSettings(String model) {
        Config current = config;
        ModelSettings settings = current == null ? null : current.modelSettings.get(model);
        return settings == null ? new ModelSettings() : settings.copy();
    }

    /** Saves overrides for a model; when Willow runs that model, restarts only its brain process to apply them. */
    public CompletableFuture<String> saveModelSettings(MinecraftServer current, String model, ModelSettings settings) {
        var result = new CompletableFuture<String>();
        ModelSettings copy = settings.copy();
        current.execute(() -> {
            try {
                if (server != current || config == null) throw new IllegalStateException("World is not running");
                if (!OllamaModels.validName(model)) throw new IllegalArgumentException("Invalid model name");
                copy.validate();
                if (copy.isEmpty()) config.modelSettings.remove(model);
                else config.modelSettings.put(model, copy);
                saveConfig();
                if (bridge != null && model.equals(selectedModel)) {
                    restartBrain();
                    result.complete("Saved. Willow's brain restarted with the new " + model + " settings.");
                } else {
                    result.complete("Saved settings for " + model + ".");
                }
            } catch (IOException | RuntimeException e) {
                LOG.error("Cannot save model settings", e);
                result.completeExceptionally(e);
            }
        });
        return result;
    }

    private BrainBridge newBridge(String model) {
        Path root = FabricLoader.getInstance().getConfigDir().resolve("localai");
        return new BrainBridge(root.resolve("runtime/server.py"), server.getWorldPath(LevelResource.ROOT).resolve("localai"),
                model, modelSettings(model).toJson());
    }

    private void restartBrain() throws IOException {
        if (pending != null) pending.cancel(true);
        pending = null;
        discardPending = false;
        if (bridge != null) bridge.close();
        BrainBridge.extractScripts(FabricLoader.getInstance().getConfigDir().resolve("localai/runtime"));
        bridge = newBridge(selectedModel);
        if (actions != null) actions.stop("Brain settings changed");
        failures = 0;
        rejectedPlans = 0;
        nextThink = server.getTickCount() + 20L;
        status = "Willow brain restarted with new settings for " + selectedModel;
        LOG.info(status);
    }

    private int cameraCommand(net.minecraft.commands.CommandSourceStack source, boolean enabled) {
        if (config == null || server == null || server.isDedicatedServer()) {
            source.sendFailure(Component.literal("Automatic cinematic viewing is available to the singleplayer host."));
            return 0;
        }
        config.cinematicCamera = enabled;
        try {
            saveConfig();
            if (!enabled) director.restore(server);
        } catch (IOException e) {
            LOG.error("Cannot update cinematic camera", e);
            source.sendFailure(Component.literal("Camera setting could not be saved/restored; see latest.log."));
            return 0;
        }
        source.sendSuccess(() -> Component.literal(enabled ? "Cinematic follow enabled." : "Returned to your player."), false);
        return 1;
    }

    public static String rootMessage(Throwable error) {
        while (error.getCause() != null) error = error.getCause();
        return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
    }

    static ServerPlayer spawn(MinecraftServer server) {
        return spawn(server, new GameProfile(PLAYER_ID, NAME));
    }

    static ServerPlayer spawn(MinecraftServer server, GameProfile profile) {
        if (server.getPlayerList().getPlayerByName(profile.name()) != null) {
            throw new IllegalStateException(profile.name() + " already exists; refusing to take over another player");
        }
        var level = server.overworld();
        var defaults = ClientInformation.createDefault();
        int viewDist = Math.max(10, server.getPlayerList().getViewDistance());
        var info = new ClientInformation(defaults.language(), viewDist, defaults.chatVisibility(),
                defaults.chatColors(), defaults.modelCustomisation(), defaults.mainHand(),
                defaults.textFilteringEnabled(), defaults.allowsListing(), defaults.particleStatus());
        var bot = EntityPlayerMPFake.respawnFake(server, level, profile, info);
        var saved = server.getPlayerList().loadPlayerData(bot.nameAndId());
        if (saved.isPresent()) {
            var dimension = net.minecraft.resources.ResourceKey.create(net.minecraft.core.registries.Registries.DIMENSION,
                    net.minecraft.resources.Identifier.parse(saved.get().getStringOr("Dimension", "minecraft:overworld")));
            var savedLevel = server.getLevel(dimension);
            if (savedLevel == null) throw new IllegalStateException("Saved Willow dimension is unavailable: " + dimension);
            bot.setServerLevel(savedLevel);
            try (var reporter = new net.minecraft.util.ProblemReporter.ScopedCollector(bot.problemPath(), LOG)) {
                var input = net.minecraft.world.level.storage.TagValueInput.create(reporter, server.registryAccess(), saved.get());
                bot.load(input);
            }
        } else {
            var spawn = bot.adjustSpawnLocation(level, level.getLevelData().getRespawnData().pos());
            bot.setPos(spawn.getX() + 0.5, spawn.getY(), spawn.getZ() + 0.5);
        }
        server.getPlayerList().placeNewPlayer(new FakeClientConnection(PacketFlow.SERVERBOUND), bot,
                new CommonListenerCookie(profile, 0, info, false));
        bot.setGameMode(GameType.SURVIVAL);
        LOG.info("Local offline player {} spawned at {}", profile.name(), bot.blockPosition());
        return bot;
    }

    private void tick(MinecraftServer minecraftServer) {
        if (server != minecraftServer) return;
        try {
            director.tick(server, config != null && config.cinematicCamera ? player : null);
        } catch (IOException e) {
            LOG.error("Cinematic follow failed; disabling it for this session", e);
            if (config != null) config.cinematicCamera = false;
            server.getPlayerList().broadcastSystemMessage(Component.literal(
                    "Willow camera could not save/restore your view; see latest.log."), false);
        }
        WillowView.capture(server.isDedicatedServer() ? null : player);
        if (bridge == null || player == null) return;
        if (paused) return;
        try {
            if (died && !player.isRemoved() && !server.overworld().getLevelData().isHardcore()) return;
            if (died || !player.isAlive() || player.isRemoved()) {
                if (server.overworld().getLevelData().isHardcore()) {
                    paused = true;
                    status = "Willow died in hardcore; paused without bypassing hardcore rules";
                    actions.stop(status);
                    return;
                }
                actions.stop("Player died", true);
                JsonObject deathResult = actions.result();
                discardPending = true;
                if (pending != null) pending.cancel(true);
                ServerPlayer connected = server.getPlayerList().getPlayer(PLAYER_ID);
                if (connected != null) return;
                var destination = died ? player.findRespawnPositionAndUseSpawnBlock(false,
                        net.minecraft.world.level.portal.TeleportTransition.DO_NOTHING) : null;
                player = spawn(server);
                AcousticSensor.setPlayer(player);
                if (destination != null) {
                    player.teleportTo(destination.newLevel(), destination.position().x, destination.position().y,
                            destination.position().z, java.util.Set.of(), destination.yRot(), destination.xRot(), true);
                }
                died = false;
                actions = new Actions(player);
                actions.restoreResult(deathResult);
                nextThink = server.getTickCount() + 60L;
            }
            // 20 Hz Fast Reactive Survival Reflex Loop (runs every 50ms server tick; 0% GPU/LLM usage)
            if (actions.reflex20Hz()) {
                discardPending = pending != null;
                if (discardPending && pending != null) pending.cancel(true);
            }
            actions.tick();
            if (player.isUsingItem() && !actions.busy()) return;
            // Pause new LLM inference dispatches while Reflex Loop is actively fighting or fleeing
            if (actions.isCombatOrFleeActive() && pending == null) return;
            if (pending != null) {
                if (!pending.isDone()) return;
                CompletableFuture<JsonObject> completed = pending;
                pending = null;
                if (discardPending) {
                    discardPending = false;
                    return;
                }
                try {
                    JsonObject decision = completed.join();
                    if (sentChat > 0) {
                        LOG.info("Willow processed {} queued message(s). Decision: action={}, goal='{}', say='{}'",
                                sentChat,
                                decision.get("action").getAsString(),
                                decision.get("goal").getAsString(),
                                decision.get("say").getAsString());
                    }
                    chat.subList(0, Math.min(sentChat, chat.size())).clear();
                    if (decision.has("vision") && decision.get("vision").isJsonPrimitive()) {
                        SightPerception.setSnapshotEnabled(decision.get("vision").getAsBoolean());
                    }
                    actions.start(decision);
                    String speech = decision.get("say").getAsString();
                    if (!speech.isBlank()) {
                        server.getPlayerList().broadcastSystemMessage(
                                Component.literal("<" + NAME + "> " + speech), false);
                        LOG.info("<{}> {}", NAME, speech);
                    } else if (sentChat > 0) {
                        String fallback = "Understood. Working on " + decision.get("goal").getAsString();
                        server.getPlayerList().broadcastSystemMessage(
                                Component.literal("<" + NAME + "> " + fallback), false);
                        LOG.info("<{}> {}", NAME, fallback);
                    }
                    status = "Willow: " + decision.get("goal").getAsString() + " [" + decision.get("action").getAsString() + "]";
                    failures = 0;
                    rejectedPlans = 0;
                    died = false;
                    nextThink = server.getTickCount() + config.thinkIntervalTicks;
                } catch (CompletionException | IllegalArgumentException | IllegalStateException e) {
                    if (e.getCause() instanceof BrainBridge.DecisionRejectedException rejected) {
                        status = "Willow is reconsidering a rejected plan: " + rejected.getMessage();
                        LOG.warn(status);
                        if (rejectedPlans++ == 0) server.getPlayerList().broadcastSystemMessage(
                                Component.literal("Willow is reconsidering a blocked or invalid plan."), false);
                        nextThink = server.getTickCount() + Math.min(200L, 40L * rejectedPlans);
                        return;
                    }
                    failures++;
                    actions.stop("Planner unavailable");
                    status = "Willow brain error; retrying. See world/localai/brain.log";
                    LOG.error(status, e);
                    if (failures == 1) server.getPlayerList().broadcastSystemMessage(Component.literal(status), false);
                    nextThink = server.getTickCount() + Math.min(1200L, 200L * failures);
                }
            }
            if (pending == null && !actions.busy() && server.getTickCount() >= nextThink) {
                discardPending = false;
                sentChat = chat.size();
                if (sentChat > 0) {
                    LOG.info("Willow is reading {} queued message(s): {}", sentChat, chat);
                }
                pending = bridge.decide(actions.observe(List.copyOf(chat)));
            }
        } catch (RuntimeException e) {
            actions.stop("Controller error");
            paused = true;
            status = "Willow paused after controller error; see Minecraft latest.log";
            LOG.error(status, e);
            server.getPlayerList().broadcastSystemMessage(Component.literal(status), false);
        }
    }

    private void stopAi(String reason) {
        lifecycle++;
        starting = false;
        selectingModel = false;
        if (actions != null) actions.stop(reason);
        if (server != null) {
            try {
                director.restore(server);
            } catch (IOException e) {
                LOG.error("Cannot restore cinematic viewer; recovery state retained", e);
            }
        }
        if (pending != null) pending.cancel(true);
        if (bridge != null) bridge.close();
        // Always unload the active model from VRAM/RAM when stopping or closing
        String activeModelToUnload = selectedModel;
        Thread.startVirtualThread(() -> OllamaService.get().unloadAll(activeModelToUnload));
        if (player != null && server != null && server.getPlayerList().getPlayer(PLAYER_ID) == player) {
            player.stopRiding();
            player.ejectPassengers();
            // Carpet's fake listener deliberately ignores disconnect(Component).
            player.connection.onDisconnect(new net.minecraft.network.DisconnectionDetails(Component.literal(reason)));
        }
        pending = null;
        bridge = null;
        player = null;
        actions = null;
        AcousticSensor.clear();
        WillowView.clear();
        chat.clear();
        failures = 0;
        rejectedPlans = 0;
        died = false;
        discardPending = false;
        paused = false;
        status = reason;
    }

    private void shutdown() {
        stopAi("World closing");
        server = null;
        status = "Not in a world";
    }
}
