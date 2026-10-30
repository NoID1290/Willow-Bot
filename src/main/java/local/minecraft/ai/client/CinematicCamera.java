package local.minecraft.ai.client;

import java.util.UUID;
import local.minecraft.ai.CameraMotion;
import net.minecraft.client.Minecraft;
import net.minecraft.world.entity.Entity;

public final class CinematicCamera {
    private static final UUID WILLOW = UUID.nameUUIDFromBytes(
            "OfflinePlayer:Willow_AI".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    private static final CameraMotion MOTION = new CameraMotion();
    private static Entity previous;
    private static long lastFrame;
    private static double zoom = CameraMotion.DEFAULT_DISTANCE;

    private CinematicCamera() {}

    public static boolean follows(Entity entity) {
        return entity != null && entity.isAlive() && !entity.isRemoved() && WILLOW.equals(entity.getUUID());
    }

    /** Multiplicative zoom keeps steps feeling even at both close and far distances. */
    public static void zoom(double factor) {
        zoom = Math.clamp(zoom * factor, CameraMotion.MIN_DISTANCE, CameraMotion.MAX_DISTANCE);
    }

    public static void resetZoom() { zoom = CameraMotion.DEFAULT_DISTANCE; }

    public static double zoomDistance() { return zoom; }

    public static CameraMotion.Pose sample(Entity entity, float partialTick) {
        if (!follows(entity) || Minecraft.getInstance().level == null) {
            previous = null;
            MOTION.reset();
            return null;
        }
        long now = System.nanoTime();
        if (previous != entity) {
            MOTION.reset();
            lastFrame = now;
            previous = entity;
        }
        // No collision sweep: the spectating camera may pass through blocks (noclip).
        CameraMotion.Pose pose = MOTION.update(entity.getEyePosition(partialTick), entity.getViewYRot(partialTick),
                (now - lastFrame) / 1e9, zoom);
        lastFrame = now;
        return pose;
    }
}
