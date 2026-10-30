package local.minecraft.ai;

import java.util.ArrayDeque;
import java.util.Deque;
import net.minecraft.core.BlockPos;

final class ProgressTracker {
    private final Deque<BlockPos> recent = new ArrayDeque<>();
    private int failures;
    private int recoveryAttempts;

    void record(String action, boolean success, BlockPos position) {
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
        return "";
    }

    int recoveryAttempts() {
        return recoveryAttempts;
    }
}
