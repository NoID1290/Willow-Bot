package local.minecraft.ai;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Set;
import java.util.UUID;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.storage.LevelResource;

final class CameraDirector {
    private record ReturnState(UUID player, String dimension, double x, double y, double z,
                               float yaw, float pitch, String gameMode) {}

    private static final Gson GSON = new Gson();
    private ReturnState saved;
    private Path file;
    private boolean recovering;

    void load(MinecraftServer server) throws IOException {
        file = server.getWorldPath(LevelResource.ROOT).resolve("localai/camera-viewer.json");
        saved = null;
        recovering = false;
        if (Files.exists(file)) {
            try {
                ReturnState loaded = GSON.fromJson(Files.readString(file), ReturnState.class);
                if (loaded == null || loaded.player == null || loaded.dimension == null || loaded.gameMode == null
                        || !Double.isFinite(loaded.x) || !Double.isFinite(loaded.y) || !Double.isFinite(loaded.z)
                        || !Float.isFinite(loaded.yaw) || !Float.isFinite(loaded.pitch)) {
                    throw new IllegalArgumentException("Invalid camera return state");
                }
                GameType.valueOf(loaded.gameMode);
                Identifier.parse(loaded.dimension);
                saved = loaded;
                recovering = true;
            } catch (RuntimeException e) {
                throw new IOException("Cannot read camera return state; file preserved", e);
            }
        }
    }

    void tick(MinecraftServer server, ServerPlayer willow) throws IOException {
        if (recovering || willow == null) {
            restore(server);
            if (saved != null) return;
        }
        if (willow == null || !willow.isAlive() || willow.isRemoved() || server.isDedicatedServer()) return;
        ServerPlayer viewer = server.getPlayerList().getPlayers().stream()
                .filter(p -> p != willow && server.isSingleplayerOwner(p.nameAndId()) && p.isAlive())
                .findFirst().orElse(null);
        if (viewer == null) return;
        follow(viewer, willow);
    }

    void follow(ServerPlayer viewer, ServerPlayer willow) throws IOException {
        if (saved == null) {
            ReturnState next = new ReturnState(viewer.getUUID(), viewer.level().dimension().identifier().toString(),
                    viewer.getX(), viewer.getY(), viewer.getZ(), viewer.getYRot(), viewer.getXRot(),
                    viewer.gameMode.getGameModeForPlayer().name());
            Files.createDirectories(file.getParent());
            Path temporary = file.resolveSibling("camera-viewer.json.tmp");
            Files.writeString(temporary, GSON.toJson(next));
            Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
            saved = next;
        }
        if (!viewer.getUUID().equals(saved.player)) return;
        if (!viewer.isSpectator()) viewer.setGameMode(GameType.SPECTATOR);
        if (viewer.getCamera() != willow || viewer.level() != willow.level()) {
            viewer.setCamera(null);
            viewer.setCamera(willow);
        } else if (viewer.level().getServer().getTickCount() % 20 == 0) {
            // The first camera packet can precede target-entity tracking after a teleport.
            viewer.connection.send(new net.minecraft.network.protocol.game.ClientboundSetCameraPacket(willow));
        }
    }

    void restore(MinecraftServer server) throws IOException {
        if (saved == null) return;
        ServerPlayer viewer = server.getPlayerList().getPlayer(saved.player);
        if (viewer == null) return;
        var level = server.getLevel(ResourceKey.create(Registries.DIMENSION, Identifier.parse(saved.dimension)));
        if (level == null) throw new IOException("Original camera-viewer dimension is not available");
        viewer.setCamera(null);
        if (!viewer.teleportTo(level, saved.x, saved.y, saved.z, Set.of(), saved.yaw, saved.pitch, true)) {
            throw new IOException("Cannot restore camera-viewer position");
        }
        viewer.setGameMode(GameType.valueOf(saved.gameMode));
        // Persist the restored player before deleting crash-recovery data.
        server.getPlayerList().saveAll();
        Files.deleteIfExists(file);
        saved = null;
        recovering = false;
    }
}
