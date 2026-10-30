package local.minecraft.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.sounds.SoundEvent;
import net.minecraft.sounds.SoundSource;
import net.minecraft.util.Mth;
import net.minecraft.world.phys.Vec3;

/**
 * Real-time acoustic sensing stream for Willow.
 * Listens to in-game sound events (creeper fuses, mob noises, footsteps, block actions)
 * and computes relative spatial bearing, distance, and urgency for multimodal local AI.
 */
public final class AcousticSensor {
    public record SoundRecord(
            String soundId,
            String sourceCategory,
            double x, double y, double z,
            double distance,
            String relativeBearing,
            String elevation,
            String urgency,
            float volume,
            float pitch,
            long gameTime
    ) {}

    private static final int MAX_HISTORY = 64;
    private static final Deque<SoundRecord> HISTORY = new ArrayDeque<>();
    private static volatile ServerPlayer activePlayer;

    private AcousticSensor() {}

    public static void setPlayer(ServerPlayer player) {
        activePlayer = player;
    }

    public static void clear() {
        synchronized (HISTORY) {
            HISTORY.clear();
        }
        activePlayer = null;
    }

    public static SoundRecord latestSound() {
        synchronized (HISTORY) {
            return HISTORY.peekLast();
        }
    }

    public static String latestSoundText() {
        SoundRecord last = latestSound();
        if (last == null) return "Quiet (no sounds heard)";
        String name = last.soundId().replaceAll("^.*:", "");
        return String.format("%s (%.1fm, %s)", name, last.distance(), last.relativeBearing());
    }

    public static String latestAlertText() {
        synchronized (HISTORY) {
            for (var it = HISTORY.descendingIterator(); it.hasNext(); ) {
                SoundRecord r = it.next();
                if ("CRITICAL".equals(r.urgency()) || "WARNING".equals(r.urgency())) {
                    String name = r.soundId().replaceAll("^.*:", "");
                    return String.format("[%s] %s (%.1fm, %s)", r.urgency(), name, r.distance(), r.relativeBearing());
                }
            }
        }
        return "None";
    }

    public static void recordSound(ServerLevel level, Holder<SoundEvent> soundHolder, SoundSource source,
                                   double x, double y, double z, float volume, float pitch) {
        ServerPlayer player = activePlayer;
        if (player == null || player.isRemoved() || player.level() != level) {
            return;
        }

        double dx = x - player.getX();
        double dy = y - player.getEyeY();
        double dz = z - player.getZ();
        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Hearing range ~32 blocks
        if (distance > 32.0) {
            return;
        }

        String soundId = soundHolder != null && soundHolder.unwrapKey().isPresent()
                ? soundHolder.unwrapKey().get().identifier().toString()
                : (soundHolder != null ? BuiltInRegistries.SOUND_EVENT.getKey(soundHolder.value()).toString() : "unknown");

        // Calculate relative bearing relative to Willow's current head yaw
        double soundAngle = Math.toDegrees(Math.atan2(-dx, dz));
        double playerYaw = player.getYRot();
        double relYaw = Mth.wrapDegrees(soundAngle - playerYaw);

        String bearing;
        if (relYaw >= -35 && relYaw <= 35) {
            bearing = "ahead";
        } else if (relYaw > 35 && relYaw <= 125) {
            bearing = "right";
        } else if (relYaw < -35 && relYaw >= -125) {
            bearing = "left";
        } else {
            bearing = "behind";
        }

        String elevation;
        if (dy > 2.0) {
            elevation = "above";
        } else if (dy < -2.0) {
            elevation = "below";
        } else {
            elevation = "level";
        }

        String urgency = classifyUrgency(soundId);
        long gameTime = level.getGameTime();

        SoundRecord record = new SoundRecord(
                soundId,
                source != null ? source.getName() : "ambient",
                Math.round(x * 10) / 10.0,
                Math.round(y * 10) / 10.0,
                Math.round(z * 10) / 10.0,
                Math.round(distance * 10) / 10.0,
                bearing,
                elevation,
                urgency,
                Math.round(volume * 100) / 100.0f,
                Math.round(pitch * 100) / 100.0f,
                gameTime
        );

        synchronized (HISTORY) {
            HISTORY.addLast(record);
            while (HISTORY.size() > MAX_HISTORY) {
                HISTORY.removeFirst();
            }
        }
        if ("CRITICAL".equals(urgency) || "WARNING".equals(urgency) || (distance < 12.0 && volume > 0.3f)) {
            LocalAiMod.LOG.info("Willow heard {} (dist={:.1f}m, bearing={}, urgency={})", soundId, distance, bearing, urgency);
        }
    }

    private static String classifyUrgency(String soundId) {
        String lower = soundId.toLowerCase(java.util.Locale.ROOT);
        if (lower.contains("creeper.primed") || lower.contains("explode") || lower.contains("tnt")
                || lower.contains("warden") || lower.contains("player.hurt") || lower.contains("player.death")) {
            return "CRITICAL";
        }
        if (lower.contains("zombie") || lower.contains("skeleton") || lower.contains("spider")
                || lower.contains("enderman") || lower.contains("witch") || lower.contains("phantom")
                || lower.contains("slime") || lower.contains("hostile") || lower.contains("attack")) {
            return "WARNING";
        }
        if (lower.contains("step") || lower.contains("break") || lower.contains("place")
                || lower.contains("door") || lower.contains("chest") || lower.contains("hit")) {
            return "NOTICE";
        }
        return "AMBIENT";
    }

    public static JsonObject summary(long currentGameTime) {
        JsonObject result = new JsonObject();
        JsonArray recent = new JsonArray();
        JsonArray alerts = new JsonArray();
        List<String> textSnippets = new ArrayList<>();

        List<SoundRecord> snapshot;
        synchronized (HISTORY) {
            snapshot = new ArrayList<>(HISTORY);
        }

        // Filter last 100 ticks (5 seconds)
        long window = 100L;
        for (int i = snapshot.size() - 1; i >= 0 && recent.size() < 12; i--) {
            SoundRecord r = snapshot.get(i);
            long ageTicks = currentGameTime - r.gameTime();
            if (ageTicks > window) {
                continue;
            }

            JsonObject soundObj = new JsonObject();
            soundObj.addProperty("sound", r.soundId());
            soundObj.addProperty("distance", r.distance());
            soundObj.addProperty("bearing", r.relativeBearing());
            soundObj.addProperty("elevation", r.elevation());
            soundObj.addProperty("urgency", r.urgency());
            soundObj.addProperty("age_ticks", ageTicks);
            recent.add(soundObj);

            if (r.urgency().equals("CRITICAL") || r.urgency().equals("WARNING")) {
                alerts.add(soundObj);
                if (textSnippets.size() < 3) {
                    textSnippets.add(r.urgency() + ": " + r.soundId().replaceAll("^.*:", "")
                            + " " + r.distance() + "m " + r.relativeBearing() + " (" + r.elevation() + ")");
                }
            }
        }

        result.add("recent_sounds", recent);
        result.add("active_alerts", alerts);

        String summaryText;
        if (!textSnippets.isEmpty()) {
            summaryText = String.join("; ", textSnippets);
        } else if (recent.isEmpty()) {
            summaryText = "Quiet surroundings; no notable acoustic events.";
        } else {
            summaryText = recent.size() + " recent ambient/footstep sounds heard.";
        }
        result.addProperty("acoustic_summary", summaryText);
        return result;
    }
}
