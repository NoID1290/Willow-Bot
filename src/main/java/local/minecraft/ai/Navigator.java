package local.minecraft.ai;

import carpet.fakes.ServerPlayerInterface;
import carpet.helpers.EntityPlayerActionPack;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.util.Mth;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

final class Navigator {
    static final long MAX_TIME_SLICE_NANOS = 2_500_000L; // Strict 2.5ms per game tick budget
    static final int MAX_SEARCH_RADIUS = 68;             // > 4 chunks (64 blocks) radius
    static final int MAX_MANHATTAN_RANGE = 136;
    static final int MAX_NODE_BUDGET = 12_000;

    private static final int[][] CARDINAL_DIRS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    private static final int[][] DIAGONAL_DIRS = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    private record Node(BlockPos pos, int cost, int score) {}

    private static int octileHeuristic(BlockPos a, BlockPos b) {
        int dx = Math.abs(a.getX() - b.getX());
        int dz = Math.abs(a.getZ() - b.getZ());
        int dy = Math.abs(a.getY() - b.getY());
        int diag = Math.min(dx, dz);
        int straight = Math.max(dx, dz) - diag;
        // Slightly weighted (x1.15) so A* expands briskly toward distant 4-chunk targets
        return ((14 * diag + 10 * straight + 8 * dy) * 23) / 20;
    }

    private static final class TimeSlicedSearch {
        final ServerLevel level;
        final BlockPos start;
        final BlockPos goal;
        final int maxDrop;
        final boolean allowWaterWading;
        final PriorityQueue<Node> open = new PriorityQueue<>(Comparator.comparingInt(Node::score));
        final Map<BlockPos, Integer> costs = new HashMap<>();
        final Map<BlockPos, BlockPos> parents = new HashMap<>();
        final Map<BlockPos, Double> heights = new HashMap<>();
        int remainingBudget;
        boolean finished = false;
        List<BlockPos> result = List.of();

        TimeSlicedSearch(ServerLevel level, BlockPos start, BlockPos goal, int maxDrop) {
            this(level, start, goal, maxDrop, MAX_NODE_BUDGET);
        }

        TimeSlicedSearch(ServerLevel level, BlockPos start, BlockPos goal, int maxDrop, int nodeBudget) {
            this.level = level;
            this.start = start;
            this.goal = goal;
            this.maxDrop = maxDrop;
            this.remainingBudget = nodeBudget;
            boolean startWet = !Double.isFinite(standingY(level, start))
                    && !level.getBlockState(start).is(Blocks.LAVA)
                    && (level.getBlockState(start).is(Blocks.WATER) || level.getBlockState(start.below()).is(Blocks.WATER));
            this.allowWaterWading = startWet || goal != null;

            if (goal != null) {
                if (Math.abs(goal.getX() - start.getX()) > MAX_SEARCH_RADIUS
                        || Math.abs(goal.getZ() - start.getZ()) > MAX_SEARCH_RADIUS
                        || start.distManhattan(goal) > MAX_MANHATTAN_RANGE) {
                    this.finished = true;
                    return;
                }
                double goalY = startWet ? waterSurfaceOrDryY(level, goal) : standingY(level, goal);
                if (!Double.isFinite(goalY)) {
                    this.finished = true;
                    return;
                }
            }

            double initialY = standingY(level, start);
            if (!Double.isFinite(initialY) && (startWet || goal != null)) {
                double wetY = waterSurfaceOrDryY(level, start);
                initialY = Double.isFinite(wetY) ? wetY : start.getY();
            }
            heights.put(start, initialY);
            costs.put(start, 0);
            open.add(new Node(start, 0, goal == null ? 0 : octileHeuristic(start, goal)));
        }

        private double resolveY(BlockPos pos) {
            return heights.computeIfAbsent(pos,
                    p -> allowWaterWading ? waterSurfaceOrDryY(level, p) : standingY(level, p));
        }

        boolean advanceSlice(long maxNanos) {
            if (finished) return true;
            long deadline = System.nanoTime() + maxNanos;
            int radius = goal == null ? 8 : MAX_SEARCH_RADIUS;

            while (!open.isEmpty() && remainingBudget-- > 0) {
                if (System.nanoTime() >= deadline) {
                    return false; // Yield to next server tick to protect 50ms tick stability
                }
                Node current = open.remove();
                if (current.cost() != costs.getOrDefault(current.pos(), Integer.MAX_VALUE)) continue;
                if (goal == null ? current.pos().getY() <= start.getY() - 2 : current.pos().equals(goal)) {
                    List<BlockPos> built = new ArrayList<>();
                    BlockPos cursor = current.pos();
                    while (!cursor.equals(start)) {
                        built.add(cursor);
                        cursor = parents.get(cursor);
                    }
                    Collections.reverse(built);
                    this.result = built;
                    this.finished = true;
                    return true;
                }

                double fromY = resolveY(current.pos());
                if (!Double.isFinite(fromY)) continue;

                // 1. Cardinal steps (climb +1 up to drop -maxDrop)
                for (int[] dir : CARDINAL_DIRS) {
                    for (int dy = 1; dy >= -maxDrop; dy--) {
                        BlockPos next = current.pos().offset(dir[0], dy, dir[1]);
                        if (Math.abs(next.getX() - start.getX()) > radius
                                || Math.abs(next.getZ() - start.getZ()) > radius) continue;
                        tryExpand(current, fromY, next, 10 + 6 * Math.abs(dy));
                    }
                }

                // 2. Smooth diagonal steps (flat or gentle -1 step down, only when both flanking corners are clear)
                for (int[] dir : DIAGONAL_DIRS) {
                    for (int dy = 0; dy >= Math.max(-1, -maxDrop); dy--) {
                        BlockPos next = current.pos().offset(dir[0], dy, dir[1]);
                        if (Math.abs(next.getX() - start.getX()) > radius
                                || Math.abs(next.getZ() - start.getZ()) > radius) continue;
                        double nextY = resolveY(next);
                        if (!Double.isFinite(nextY)) continue;
                        double clearanceY = Math.max(fromY, nextY);
                        BlockPos flankA = current.pos().offset(dir[0], 0, 0);
                        BlockPos flankB = current.pos().offset(0, 0, dir[1]);
                        if (!clearAt(level, flankA, clearanceY) || !clearAt(level, flankB, clearanceY)) {
                            continue;
                        }
                        tryExpand(current, fromY, next, 14 + 6 * Math.abs(dy));
                    }
                }
            }
            this.finished = true;
            return true;
        }

        private void tryExpand(Node current, double fromY, BlockPos next, int baseStepCost) {
            double nextY = resolveY(next);
            if (!Double.isFinite(nextY)
                    || nextY - fromY > 1.001 || fromY - nextY > maxDrop + 0.001) return;
            double topRef = Math.max(fromY, nextY);
            if (!clearAt(level, current.pos(), topRef) || !clearAt(level, next, topRef)) return;
            if (nextY < fromY && !level.noBlockCollision(null,
                    new AABB(next.getX() + 0.2, nextY + 0.001, next.getZ() + 0.2,
                            next.getX() + 0.8, fromY + 1.8, next.getZ() + 0.8))) return;
            int waterPenalty = (allowWaterWading && !safe(level, next)) ? 55 : 0;
            int cost = current.cost() + baseStepCost + waterPenalty;
            if (cost < costs.getOrDefault(next, Integer.MAX_VALUE)) {
                costs.put(next, cost);
                parents.put(next, current.pos());
                open.add(new Node(next, cost, cost + (goal == null ? 0 : octileHeuristic(next, goal))));
            }
        }
    }

    private final ServerPlayer player;
    private List<BlockPos> path = List.of();
    private TimeSlicedSearch pendingSearch;
    private BlockPos activeGoal;
    private int activeMaxDrop = 3;
    private boolean repathedOnce = false;
    private int index;
    private int stuck;
    private Vec3 previous;
    private boolean sprinting;

    Navigator(ServerPlayer player) {
        this.player = player;
        previous = player.position();
        sprinting = false;
    }

    static boolean safe(ServerLevel level, BlockPos pos) {
        return Double.isFinite(standingY(level, pos));
    }

    private static boolean isHazardBlock(ServerLevel level, BlockPos cell) {
        var state = level.getBlockState(cell);
        return state.is(Blocks.LAVA)
                || state.is(Blocks.FIRE) || state.is(Blocks.SOUL_FIRE)
                || state.is(Blocks.CACTUS) || state.is(Blocks.MAGMA_BLOCK)
                || state.is(Blocks.CAMPFIRE) || state.is(Blocks.SOUL_CAMPFIRE)
                || state.is(Blocks.POWDER_SNOW) || state.is(Blocks.SWEET_BERRY_BUSH)
                || state.is(Blocks.WITHER_ROSE);
    }

    static double standingY(ServerLevel level, BlockPos pos) {
        if (!level.isLoaded(pos) || !level.isLoaded(pos.above()) || !level.isLoaded(pos.below())
                || !level.isLoaded(pos.above(2))
                || !level.getWorldBorder().isWithinBounds(pos)) return Double.NaN;
        for (BlockPos cell : List.of(pos, pos.above(), pos.below())) {
            var state = level.getBlockState(cell);
            if (!state.getFluidState().isEmpty() || isHazardBlock(level, cell)) return Double.NaN;
        }
        var feetShape = level.getBlockState(pos).getCollisionShape(level, pos);
        double surface = pos.getY();
        if (!feetShape.isEmpty()) {
            // Feet can stand inside the cell on layered snow, carpets, trapdoors, or bottom slabs.
            double height = feetShape.max(Direction.Axis.Y);
            if (height >= 1.0 || feetShape.min(Direction.Axis.X) > 0.125
                    || feetShape.max(Direction.Axis.X) < 0.875 || feetShape.min(Direction.Axis.Z) > 0.125
                    || feetShape.max(Direction.Axis.Z) < 0.875) return Double.NaN;
            surface += height;
        } else {
            var belowPos = pos.below();
            var belowState = level.getBlockState(belowPos);
            if (!belowState.isCollisionShapeFullBlock(level, belowPos)) {
                var belowShape = belowState.getCollisionShape(level, belowPos);
                if (belowShape.isEmpty()) return Double.NaN;
                double topY = belowShape.max(Direction.Axis.Y);
                // Support dirt_path (0.9375), farmland, mud, soul_sand (0.875), top-slabs (1.0), while rejecting fences/walls (1.5 & narrow)
                if (topY < 0.85 || topY > 1.001
                        || belowShape.min(Direction.Axis.X) > 0.125 || belowShape.max(Direction.Axis.X) < 0.875
                        || belowShape.min(Direction.Axis.Z) > 0.125 || belowShape.max(Direction.Axis.Z) < 0.875) {
                    return Double.NaN;
                }
                surface = pos.getY() - 1.0 + topY;
            }
        }
        return clearAt(level, pos, surface) ? surface : Double.NaN;
    }

    private static boolean clearAt(ServerLevel level, BlockPos pos, double surface) {
        return level.noBlockCollision(null, new AABB(pos.getX() + 0.2, surface + 0.001,
                pos.getZ() + 0.2, pos.getX() + 0.8, surface + 1.8, pos.getZ() + 0.8));
    }

    private static double waterSurfaceOrDryY(ServerLevel level, BlockPos pos) {
        double dry = standingY(level, pos);
        if (Double.isFinite(dry)) return dry;
        if (!level.isLoaded(pos) || !level.isLoaded(pos.above()) || !level.isLoaded(pos.below())
                || !level.getWorldBorder().isWithinBounds(pos)) return Double.NaN;
        for (BlockPos cell : List.of(pos, pos.above(), pos.below())) {
            if (isHazardBlock(level, cell)) return Double.NaN;
        }
        var state = level.getBlockState(pos);
        var above = level.getBlockState(pos.above());
        if ((state.is(Blocks.WATER) || state.getCollisionShape(level, pos).isEmpty())
                && above.getFluidState().isEmpty()
                && above.getCollisionShape(level, pos.above()).isEmpty()
                && (state.is(Blocks.WATER) || level.getBlockState(pos.below()).is(Blocks.WATER))) {
            return pos.getY();
        }
        return Double.NaN;
    }

    static List<BlockPos> find(ServerLevel level, BlockPos start, BlockPos goal) {
        return find(level, start, goal, 3);
    }

    static List<BlockPos> find(ServerLevel level, BlockPos start, BlockPos goal, int maxDrop) {
        TimeSlicedSearch search = new TimeSlicedSearch(level, start, goal, maxDrop, MAX_NODE_BUDGET);
        while (!search.advanceSlice(Long.MAX_VALUE / 4)) {
            // Synchronous helper used by tests & fast verification
        }
        return search.result;
    }

    boolean go(BlockPos destination) {
        return go(destination, 3);
    }

    boolean sprintTo(BlockPos destination) {
        boolean ok = go(destination);
        if (ok) sprinting = true;
        return ok;
    }

    /** Commits to a destination ONLY if a complete verified path is found within a bounded node/time budget. */
    boolean goIfReachable(BlockPos destination, int maxDrop, boolean preferSprint) {
        index = 0;
        stuck = 0;
        sprinting = false;
        repathedOnce = false;
        activeGoal = destination;
        activeMaxDrop = maxDrop;
        previous = player.position();
        pendingSearch = null;
        if (player.blockPosition().equals(destination) && safe(player.level(), destination)) {
            path = List.of();
            return false;
        }
        TimeSlicedSearch search = new TimeSlicedSearch(player.level(), player.blockPosition(), destination, maxDrop, 3500);
        if (search.advanceSlice(1_200_000L) && !search.result.isEmpty()) {
            path = search.result;
            sprinting = preferSprint && player.getFoodData().getFoodLevel() > 6;
            return true;
        }
        path = List.of();
        return false;
    }

    boolean descend() {
        index = 0;
        stuck = 0;
        sprinting = false;
        repathedOnce = false;
        activeGoal = null;
        activeMaxDrop = 12;
        previous = player.position();
        pendingSearch = null;
        TimeSlicedSearch search = new TimeSlicedSearch(player.level(), player.blockPosition(), null, 12);
        if (search.advanceSlice(MAX_TIME_SLICE_NANOS)) {
            path = search.result;
            return !path.isEmpty();
        }
        pendingSearch = search;
        path = List.of();
        return true;
    }

    boolean go(BlockPos destination, int maxDrop) {
        index = 0;
        stuck = 0;
        sprinting = false;
        repathedOnce = false;
        activeGoal = destination;
        activeMaxDrop = maxDrop;
        previous = player.position();
        pendingSearch = null;
        if (player.blockPosition().equals(destination) && safe(player.level(), destination)) {
            path = List.of();
            return true;
        }
        TimeSlicedSearch search = new TimeSlicedSearch(player.level(), player.blockPosition(), destination, maxDrop);
        if (search.advanceSlice(MAX_TIME_SLICE_NANOS)) {
            path = search.result;
            return !path.isEmpty();
        }
        pendingSearch = search;
        path = List.of();
        return true;
    }

    private double waypointSurfaceY(BlockPos wp) {
        double y = standingY(player.level(), wp);
        if (!Double.isFinite(y)) {
            y = waterSurfaceOrDryY(player.level(), wp);
        }
        return y;
    }

    boolean tick() {
        EntityPlayerActionPack pack = ((ServerPlayerInterface) player).getActionPack();
        if (pendingSearch != null) {
            if (!pendingSearch.advanceSlice(MAX_TIME_SLICE_NANOS)) {
                return false;
            }
            path = pendingSearch.result;
            pendingSearch = null;
            if (path.isEmpty()) {
                pack.setSprinting(false);
                throw new IllegalArgumentException("No safe path found within search budget");
            }
        }
        if (index >= path.size()) {
            finishArrival(pack);
            return true;
        }

        // Smoothly advance past intermediate waypoints without stopping or breaking stride!
        while (index < path.size() - 1) {
            BlockPos wp = path.get(index);
            double wpY = waypointSurfaceY(wp);
            if (!Double.isFinite(wpY)) break;
            double horiz = Math.hypot((wp.getX() + 0.5) - player.getX(), (wp.getZ() + 0.5) - player.getZ());
            if (horiz < 0.55 && Math.abs(wpY - player.getY()) < 0.65) {
                index++;
            } else {
                break;
            }
        }

        BlockPos waypoint = path.get(index);
        double waypointY = waypointSurfaceY(waypoint);
        if (!Double.isFinite(waypointY)) {
            if (!repathedOnce && activeGoal != null && safe(player.level(), activeGoal)) {
                repathedOnce = true;
                if (go(activeGoal, activeMaxDrop)) return false;
            }
            pack.setSprinting(false);
            throw new IllegalArgumentException("Path became unsafe");
        }

        Vec3 target = new Vec3(waypoint.getX() + 0.5, waypointY, waypoint.getZ() + 0.5);
        double horizontal = Math.hypot(target.x - player.getX(), target.z - player.getZ());
        boolean isFinalWaypoint = (index == path.size() - 1);

        if (isFinalWaypoint && horizontal < 0.35 && Math.abs(target.y - player.getY()) < 0.5
                && (player.onGround() || player.isInWater())) {
            index++;
            finishArrival(pack);
            return true;
        }

        // Track stuck state and apply progressive micro-recovery before aborting
        if (player.position().distanceToSqr(previous) < 0.0004) stuck++;
        else stuck = 0;
        previous = player.position();

        if (stuck == 35 && !repathedOnce && activeGoal != null && safe(player.level(), activeGoal)) {
            repathedOnce = true;
            if (go(activeGoal, activeMaxDrop)) {
                return false;
            }
        }
        if (stuck > 65) {
            pack.setSprinting(false);
            pack.setStrafing(0f);
            throw new IllegalArgumentException("Movement stuck; choose a different route");
        }

        // Steer horizontally toward current (or smoothly blended next flat) waypoint
        Vec3 steerPoint = target;
        if (!isFinalWaypoint && horizontal < 0.85) {
            BlockPos nextWp = path.get(index + 1);
            double nextY = waypointSurfaceY(nextWp);
            if (Double.isFinite(nextY) && Math.abs(nextY - waypointY) < 0.15) {
                steerPoint = target.scale(0.6).add(new Vec3(nextWp.getX() + 0.5, nextY, nextWp.getZ() + 0.5).scale(0.4));
            }
        }

        float desiredYaw = (float) Math.toDegrees(Math.atan2(-(steerPoint.x - player.getX()), steerPoint.z - player.getZ()));
        float yawDelta = Math.abs(Mth.wrapDegrees(desiredYaw - player.getYRot()));
        float smoothYaw = yawDelta > 55.0f ? desiredYaw : Mth.rotLerp(0.55f, player.getYRot(), desiredYaw);

        // Natural human horizon gaze: look 4 waypoints ahead Eye-to-Eye (NEVER stare down at own feet!)
        int lookIdx = Math.min(path.size() - 1, index + 4);
        BlockPos lookWp = path.get(lookIdx);
        double lookBaseY = waypointSurfaceY(lookWp);
        if (!Double.isFinite(lookBaseY)) lookBaseY = waypointY;
        double lookEyeY = lookBaseY + player.getEyeHeight() * 0.92;
        double lookDist = Math.max(1.5, Math.hypot((lookWp.getX() + 0.5) - player.getX(), (lookWp.getZ() + 0.5) - player.getZ()));
        float desiredPitch = (float) Mth.clamp(-Math.toDegrees(Math.atan2(lookEyeY - player.getEyeY(), lookDist)), -28.0, 14.0);
        float smoothPitch = Mth.rotLerp(0.35f, player.getXRot(), desiredPitch);

        boolean canSprint = player.getFoodData().getFoodLevel() > 6 && !player.isInWater();
        int remainingSteps = path.size() - index;
        boolean sprintNow = canSprint && yawDelta < 45.0f
                && (sprinting || remainingSteps > 3 || (!isFinalWaypoint && horizontal > 1.1));
        pack.setSprinting(sprintNow);

        float forwardSpeed;
        if (!player.onGround() && !player.isInWater() && player.getY() > target.y + 0.25 && horizontal < 0.25) {
            // Already airborne directly above a drop landing cell: ease forward so we land cleanly on target
            forwardSpeed = 0.25f;
        } else if (isFinalWaypoint && horizontal < 0.65) {
            forwardSpeed = 0.55f;
        } else {
            forwardSpeed = 1.0f;
        }

        // Micro-unstuck strafe if snagged on a corner/trunk
        float strafe = 0.0f;
        if (stuck >= 10 && stuck < 22) strafe = 0.35f;
        else if (stuck >= 22 && stuck < 34) strafe = -0.35f;

        pack.look(smoothYaw, smoothPitch).setForward(forwardSpeed).setStrafing(strafe);

        boolean needStepJump = target.y > player.getY() + 0.42 && horizontal < 1.45 && player.onGround();
        boolean collisionUnstuckJump = stuck > 0 && (stuck % 12 == 0) && player.horizontalCollision && player.onGround();
        if (needStepJump || collisionUnstuckJump || player.isInWater()) {
            pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }
        return false;
    }

    private void finishArrival(EntityPlayerActionPack pack) {
        pack.setSprinting(false);
        pack.setStrafing(0f);
        pack.stopMovement();
        // Keep head near level horizon on arrival so SightPerception scans chunks ahead instead of dirt
        if (Math.abs(player.getXRot()) > 16.0f) {
            pack.look(player.getYRot(), Mth.clamp(player.getXRot(), -12.0f, 8.0f));
        }
    }
}
