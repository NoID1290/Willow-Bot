package local.minecraft.ai.client;

import com.mojang.blaze3d.platform.InputConstants;
import local.minecraft.ai.LocalAiMod;
import local.minecraft.ai.OllamaService;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.CameraType;
import net.minecraft.client.KeyMapping;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;

public final class WillowClient implements ClientModInitializer {
    private static final KeyMapping.Category CATEGORY = KeyMapping.Category.register(
            Identifier.fromNamespaceAndPath("localai", "willow"));
    private MinecraftServer world;
    private long pickerRequest = -1;
    private CameraType previousCamera;
    private Boolean previousBobbing;
    private KeyMapping zoomIn;
    private KeyMapping zoomOut;
    private KeyMapping zoomReset;
    private KeyMapping toggleInventory;
    private KeyMapping openSettings;
    private long settingsRequest;

    private static KeyMapping key(String name, int glfwKey) {
        return KeyMappingHelper.registerKeyMapping(new KeyMapping("key.localai." + name,
                glfwKey, CATEGORY));
    }

    private void handleKeys(net.minecraft.client.Minecraft client) {
        boolean following = CinematicCamera.follows(client.getCameraEntity()) && client.gui.screen() == null;
        if (following && zoomIn.isDown()) {
            CinematicCamera.zoom(0.93);
            WillowHud.showZoom();
        }
        if (following && zoomOut.isDown()) {
            CinematicCamera.zoom(1 / 0.93);
            WillowHud.showZoom();
        }
        while (zoomReset.consumeClick()) {
            if (!following) continue;
            CinematicCamera.resetZoom();
            WillowHud.showZoom();
        }
        while (toggleInventory.consumeClick()) {
            if (following) WillowHud.inventoryVisible = !WillowHud.inventoryVisible;
        }
        while (openSettings.consumeClick()) {
            if (client.gui.screen() == null) openSettings(client);
        }
    }

    private static void openSettings(net.minecraft.client.Minecraft client) {
        var controller = LocalAiMod.instance();
        String running = controller.runningModel();
        client.gui.setScreen(new WillowSettingsScreen(client.getSingleplayerServer(), null,
                running != null ? running : controller.selectedModel()));
    }

    @Override
    public void onInitializeClient() {
        zoomIn = key("zoom_in", InputConstants.KEY_EQUALS);
        zoomOut = key("zoom_out", InputConstants.KEY_MINUS);
        zoomReset = key("zoom_reset", InputConstants.KEY_0);
        toggleInventory = key("inventory", InputConstants.KEY_I);
        openSettings = key("settings", InputConstants.KEY_O);
        HudElementRegistry.addLast(Identifier.fromNamespaceAndPath("localai", "willow_inventory"), WillowHud::extract);
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> OllamaService.get().shutdownOnExit());
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            handleKeys(client);
            if (Boolean.getBoolean("localai.clientSmoke") && client.level == null && client.gui.overlay() == null
                    && client.gui.screen() != null && !(client.gui.screen() instanceof ModelPickerScreen)) {
                client.gui.setScreen(new ModelPickerScreen(null));
            }
            MinecraftServer current = client.getSingleplayerServer();
            if (current != world) {
                world = current;
                pickerRequest = -1;
            }
            var controller = LocalAiMod.instance();
            if (current != null && client.level != null && client.player != null && client.gui.screen() == null
                    && controller != null && controller.wantsSelection(current)
                    && pickerRequest != controller.selectionRequest()) {
                pickerRequest = controller.selectionRequest();
                client.gui.setScreen(new ModelPickerScreen(current));
            }
            if (controller != null && settingsRequest != controller.settingsRequest()) {
                settingsRequest = controller.settingsRequest();
                if (current != null && client.level != null) openSettings(client);
            }
            if (CinematicCamera.follows(client.getCameraEntity())) {
                if (previousCamera == null) {
                    previousCamera = client.options.getCameraType();
                    previousBobbing = client.options.bobView().get();
                }
                client.options.setCameraType(CameraType.THIRD_PERSON_BACK);
                client.options.bobView().set(false);
            } else if (previousCamera != null) {
                client.options.setCameraType(previousCamera);
                client.options.bobView().set(previousBobbing);
                previousCamera = null;
                previousBobbing = null;
            }
        });
        if (Boolean.getBoolean("localai.clientCameraSmoke")) {
            var smoke = new ClientCameraSmoke();
            ClientTickEvents.END_CLIENT_TICK.register(smoke::tick);
        }
    }
}
