package local.minecraft.ai;

import java.util.ArrayDeque;
import java.util.Deque;
import net.minecraft.core.BlockPos;

final class ProgressTracker {
    private final Deque<BlockPos> recent = new ArrayDeque<>();
    private int failures;
    private int recoveryAttempts;

    void record(String action, String status, BlockPos position) {
        // Interruptions (eating reflex, player chat, pause) are not evidence of being stuck.
        if ("interrupted".equals(status)) return;
        boolean success = "success".equals(status);
        if (action.equals("escape")) recoveryAttempts++;
        if (!success) failures++;
        else failures = 0;
        if (success && (action.equals("mine") || action.equals("craft")
                || action.equals("place") || action.equals("pickup") || action.equals("escape"))) {
            recent.clear();
        } else if (!action.equals("eat") && !action.equals("attack")) {
            recent.addLast(position.immutable());
            while (recent.size() > 8) recent.removeFirst();
        }
    }

    String problem() {
        if (failures >= 3) return "Three or more consecutive actions failed";
        if (recent.size() < 8) return "";
        BlockPos origin = recent.getFirst();
        if (recent.stream().allMatch(pos -> pos.distSqr(origin) <= 25)) {
            return "Eight actions without resource progress, circling within five blocks";
        }
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        int minZ = Integer.MAX_VALUE, maxZ = Integer.MIN_VALUE;
        for (BlockPos pos : recent) {
            minX = Math.min(minX, pos.getX());
            maxX = Math.max(maxX, pos.getX());
            minY = Math.min(minY, pos.getY());
            maxY = Math.max(maxY, pos.getY());
            minZ = Math.min(minZ, pos.getZ());
            maxZ = Math.max(maxZ, pos.getZ());
        }
        if (maxX - minX <= 12 && maxZ - minZ <= 12 && maxY - minY <= 6) {
            return "Eight actions without resource progress, oscillating in the same small area";
        }
        return "";
    }

    int recoveryAttempts() {
        return recoveryAttempts;
    }
}
