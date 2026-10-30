package local.minecraft.ai.client;

import java.util.concurrent.CompletableFuture;
import local.minecraft.ai.LocalAiMod;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

final class ClientCameraSmoke {
    private int phase;
    private int followingTicks;
    private final long started = System.nanoTime();
    private CompletableFuture<?> operation;
    private Vec3 originalPosition;
    private GameType originalMode;
    private net.minecraft.client.CameraType originalCamera;

    void tick(Minecraft client) {
        if (System.nanoTime() - started > java.time.Duration.ofMinutes(3).toNanos()) {
            throw new IllegalStateException("Cinematic client smoke test timed out at phase " + phase);
        }
        if (phase == 0 && client.gui.screen() != null && client.gui.overlay() == null) {
            phase = 1;
            client.createWorldOpenFlows().openWorld("camera-smoke", () -> {
                throw new IllegalStateException("Could not open isolated camera test world");
            });
        } else if (phase == 1 && client.gui.screen() instanceof ModelPickerScreen
                && client.player != null && client.getSingleplayerServer() != null) {
            originalPosition = client.player.position();
            originalMode = client.gameMode.getPlayerMode();
            originalCamera = client.options.getCameraType();
            operation = LocalAiMod.instance().startSelected(client.getSingleplayerServer(), "qwen3.5:4b");
            client.gui.setScreen(null);
            phase = 2;
        } else if (phase == 2 && operation.isDone()) {
            operation.join();
            var server = client.getSingleplayerServer();
            operation = CompletableFuture.runAsync(() -> {
                var bot = server.getPlayerList().getPlayerByName("Willow_AI");
                if (bot == null) throw new IllegalStateException("Start did not connect Willow");
                BlockPos center = bot.blockPosition();
                for (BlockPos pos : BlockPos.betweenClosed(center.offset(-6, 0, -6), center.offset(6, 6, 6))) {
                    bot.level().setBlockAndUpdate(pos, Blocks.AIR.defaultBlockState());
                }
                for (BlockPos pos : BlockPos.betweenClosed(center.offset(-6, -1, -6), center.offset(6, -1, 6))) {
                    bot.level().setBlockAndUpdate(pos, Blocks.STONE.defaultBlockState());
                }
            }, server);
            phase = 3;
        } else if (phase == 3 && operation.isDone() && CinematicCamera.follows(client.getCameraEntity())) {
            operation.join();
            if (++followingTicks < 80) return;
            var camera = client.gameRenderer.mainCamera();
            if (!client.gameMode.isSpectator() || !camera.isDetached()
                    || camera.position().distanceTo(client.getCameraEntity().getEyePosition()) < 0.5
                    || camera.position().distanceTo(client.getCameraEntity().getEyePosition()) > 8) {
                throw new IllegalStateException("Cinematic camera not correctly following Willow: " + camera.position());
            }
            var server = client.getSingleplayerServer();
            operation = CompletableFuture.runAsync(() -> {
                var bot = server.getPlayerList().getPlayerByName("Willow_AI");
                bot.getInventory().setItem(20, new net.minecraft.world.item.ItemStack(
                        net.minecraft.world.item.Items.DIAMOND, 7));
                bot.getInventory().setItem(4, new net.minecraft.world.item.ItemStack(
                        net.minecraft.world.item.Items.OAK_LOG, 33));
                bot.setItemSlot(net.minecraft.world.entity.EquipmentSlot.HEAD,
                        new net.minecraft.world.item.ItemStack(net.minecraft.world.item.Items.IRON_HELMET));
            }, server);
            followingTicks = 0;
            phase = 30;
        } else if (phase == 30 && operation.isDone()) {
            operation.join();
            var view = local.minecraft.ai.WillowView.current();
            if (view == null || view.main().get(20).getCount() != 7 || !view.main().get(20).is(net.minecraft.world.item.Items.DIAMOND)
                    || !view.armor().get(0).is(net.minecraft.world.item.Items.IRON_HELMET)) {
                if (++followingTicks > 40) throw new IllegalStateException("Live inventory HUD did not mirror Willow: " + view);
                return;
            }
            if (!WillowHud.inventoryVisible) throw new IllegalStateException("Inventory panel should be visible by default");
            net.minecraft.client.Screenshot.grab(client.gameDirectory, "willow-hud.png", client.gameRenderer.mainRenderTarget(), 1,
                    message -> org.slf4j.LoggerFactory.getLogger("localai").info("HUD screenshot: {}", message.getString()));
            var server = client.getSingleplayerServer();
            operation = CompletableFuture.runAsync(() -> {
                // Enclose Willow so a colliding camera would be forced close; noclip must stay at the zoomed distance.
                var bot = server.getPlayerList().getPlayerByName("Willow_AI");
                BlockPos center = bot.blockPosition();
                for (BlockPos pos : BlockPos.betweenClosed(center.offset(-3, -1, -3), center.offset(3, 4, 3))) {
                    if (Math.abs(pos.getX() - center.getX()) == 3 || Math.abs(pos.getZ() - center.getZ()) == 3
                            || pos.getY() == center.getY() + 4) {
                        bot.level().setBlockAndUpdate(pos, Blocks.STONE.defaultBlockState());
                    }
                }
            }, server);
            for (int i = 0; i < 20; i++) CinematicCamera.zoom(1 / 0.93);
            followingTicks = 0;
            phase = 31;
        } else if (phase == 31 && operation.isDone()) {
            operation.join();
            if (++followingTicks < 60) return;
            var camera = client.gameRenderer.mainCamera();
            double distance = camera.position().distanceTo(client.getCameraEntity().getEyePosition());
            if (CinematicCamera.zoomDistance() < 10 || distance < 10) {
                throw new IllegalStateException("Zoom/noclip camera was pulled in: wanted "
                        + CinematicCamera.zoomDistance() + " actual " + distance);
            }
            org.slf4j.LoggerFactory.getLogger("localai").info(
                    "CLIENT ZOOM/NOCLIP PASS: camera {} blocks away through stone walls", String.format("%.1f", distance));
            CinematicCamera.resetZoom();
            operation = command(client, "brain settings");
            phase = 32;
        } else if (phase == 32 && operation.isDone() && client.gui.screen() instanceof WillowSettingsScreen screen
                && screen.profile() != null) {
            operation.join();
            var profile = screen.profile();
            if (!screen.model().equals("qwen3.5:4b") || profile.get("trained_context").getAsInt() < 32768
                    || profile.get("context").getAsInt() != 32768) {
                throw new IllegalStateException("Settings screen showed the wrong model profile: " + profile);
            }
            net.minecraft.client.Screenshot.grab(client.gameDirectory, "willow-settings.png", client.gameRenderer.mainRenderTarget(), 1,
                    message -> org.slf4j.LoggerFactory.getLogger("localai").info("Settings screenshot: {}", message.getString()));
            var override = new local.minecraft.ai.ModelSettings();
            override.contextLength = 8192;
            operation = LocalAiMod.instance().saveModelSettings(client.getSingleplayerServer(), "qwen3.5:4b", override);
            phase = 33;
        } else if (phase == 33 && operation.isDone()) {
            String saved = (String) operation.join();
            if (!saved.contains("restarted") || !LocalAiMod.instance().modelSettings("qwen3.5:4b").contextLength.equals(8192)
                    || !"qwen3.5:4b".equals(LocalAiMod.instance().runningModel())) {
                throw new IllegalStateException("Saving settings did not hot-restart Willow's brain: " + saved);
            }
            operation = LocalAiMod.instance().saveModelSettings(client.getSingleplayerServer(), "qwen3.5:4b",
                    new local.minecraft.ai.ModelSettings());
            phase = 34;
        } else if (phase == 34 && operation.isDone()) {
            operation.join();
            if (!LocalAiMod.instance().modelSettings("qwen3.5:4b").isEmpty()) {
                throw new IllegalStateException("Reset to auto was not saved");
            }
            org.slf4j.LoggerFactory.getLogger("localai").info(
                    "CLIENT SETTINGS PASS: /brain settings opened, profile detected, saved override hot-restarted brain");
            client.gui.setScreen(null);
            operation = command(client, "STOP");
            phase = 4;
        } else if (phase == 4 && operation.isDone() && client.getCameraEntity() == client.player
                && client.gameMode.getPlayerMode() == originalMode) {
            operation.join();
            if (client.player.position().distanceTo(originalPosition) > 1
                    || client.options.getCameraType() != originalCamera) {
                throw new IllegalStateException("STOP did not restore the original player/view");
            }
            operation = command(client, "START");
            phase = 5;
        } else if (phase == 5 && operation.isDone() && client.gui.screen() instanceof ModelPickerScreen) {
            operation.join();
            org.slf4j.LoggerFactory.getLogger("localai").info(
                    "CLIENT CAMERA PASS: startup picker, selected model, cinematic follow, STOP restoration, START picker");
            phase = 6;
            client.stop();
        }
    }

    private CompletableFuture<Void> command(Minecraft client, String command) {
        var server = client.getSingleplayerServer();
        return CompletableFuture.runAsync(() -> {
            try {
                server.getCommands().getDispatcher().execute(command, server.createCommandSourceStack());
            } catch (com.mojang.brigadier.exceptions.CommandSyntaxException e) {
                throw new IllegalStateException(e);
            }
        }, server);
    }
}
