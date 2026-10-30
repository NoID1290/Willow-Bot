package local.minecraft.ai;

import net.minecraft.util.Mth;
import net.minecraft.world.phys.Vec3;

public final class CameraMotion {
    public record Pose(Vec3 position, float yaw, float pitch) {}
    public static final double DEFAULT_DISTANCE = 4.5;
    public static final double MIN_DISTANCE = 1.5;
    public static final double MAX_DISTANCE = 32;
    private Vec3 focus;
    private float heading;
    private double distance = DEFAULT_DISTANCE;

    public Pose update(Vec3 target, float targetYaw, double seconds) {
        return update(target, targetYaw, seconds, DEFAULT_DISTANCE);
    }

    /** {@code wantedDistance} is eased so keyboard zoom glides instead of jumping. */
    public Pose update(Vec3 target, float targetYaw, double seconds, double wantedDistance) {
        boolean reset = focus == null || focus.distanceToSqr(target) > 32 * 32;
        double delta = Math.clamp(seconds, 0, 0.1);
        double wanted = Math.clamp(wantedDistance, MIN_DISTANCE, MAX_DISTANCE);
        if (reset) {
            focus = target;
            heading = targetYaw;
            distance = wanted;
        } else {
            focus = focus.lerp(target, 1 - Math.exp(-5 * delta));
            heading = Mth.rotLerp((float) (1 - Math.exp(-2 * delta)), heading, targetYaw);
            distance += (wanted - distance) * (1 - Math.exp(-8 * delta));
        }
        double angle = Math.toRadians(heading + 22);
        double height = 1.7 * distance / DEFAULT_DISTANCE;
        Vec3 position = focus.add(Math.sin(angle) * distance, height, -Math.cos(angle) * distance);
        return lookingAt(position, target);
    }

    public double distance() { return distance; }

    public static Pose lookingAt(Vec3 position, Vec3 target) {
        Vec3 direction = target.subtract(position);
        float yaw = (float) Math.toDegrees(Math.atan2(-direction.x, direction.z));
        float pitch = (float) -Math.toDegrees(Math.atan2(direction.y, Math.hypot(direction.x, direction.z)));
        return new Pose(position, yaw, pitch);
    }

    public void reset() { focus = null; }
}
