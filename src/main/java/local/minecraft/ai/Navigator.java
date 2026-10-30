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
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.AABB;

final class Navigator {
    private record Node(BlockPos pos, int cost, int score) {}
    private final ServerPlayer player;
    private List<BlockPos> path = List.of();
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

    static double standingY(ServerLevel level, BlockPos pos) {
        if (!level.isLoaded(pos) || !level.isLoaded(pos.above()) || !level.isLoaded(pos.below())
                || !level.isLoaded(pos.above(2))
                || !level.getWorldBorder().isWithinBounds(pos)) return Double.NaN;
        for (BlockPos cell : List.of(pos, pos.above(), pos.below())) {
            var state = level.getBlockState(cell);
            if (!state.getFluidState().isEmpty() || state.is(Blocks.LAVA)
                    || state.is(Blocks.FIRE) || state.is(Blocks.SOUL_FIRE)
                    || state.is(Blocks.CACTUS) || state.is(Blocks.MAGMA_BLOCK)
                    || state.is(Blocks.CAMPFIRE) || state.is(Blocks.SOUL_CAMPFIRE)
                    || state.is(Blocks.POWDER_SNOW) || state.is(Blocks.SWEET_BERRY_BUSH)) return Double.NaN;
        }
        var feetShape = level.getBlockState(pos).getCollisionShape(level, pos);
        double surface = pos.getY();
        if (!feetShape.isEmpty()) {
            // Feet can stand inside the block cell on layered snow or a bottom slab.
            double height = feetShape.max(Direction.Axis.Y);
            if (height >= 1 || feetShape.min(Direction.Axis.X) > 0
                    || feetShape.max(Direction.Axis.X) < 1 || feetShape.min(Direction.Axis.Z) > 0
                    || feetShape.max(Direction.Axis.Z) < 1) return Double.NaN;
            surface += height;
        } else if (!level.getBlockState(pos.below()).isCollisionShapeFullBlock(level, pos.below())) {
            return Double.NaN;
        }
        return clearAt(level, pos, surface) ? surface : Double.NaN;
    }

    private static boolean clearAt(ServerLevel level, BlockPos pos, double surface) {
        return level.noBlockCollision(null, new AABB(pos.getX() + 0.2, surface + 0.001,
                pos.getZ() + 0.2, pos.getX() + 0.8, surface + 1.8, pos.getZ() + 0.8));
    }

    private static double waterEscapeY(ServerLevel level, BlockPos pos) {
        double dry = standingY(level, pos);
        if (Double.isFinite(dry)) return dry;
        if (!level.isLoaded(pos) || !level.isLoaded(pos.above()) || !level.isLoaded(pos.below())
                || !level.getWorldBorder().isWithinBounds(pos)) return Double.NaN;
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
        if (goal != null && (start.distManhattan(goal) > 48 || !safe(level, goal))) return List.of();
        boolean startInWater = !Double.isFinite(standingY(level, start))
                && (level.getBlockState(start).is(Blocks.WATER) || level.getBlockState(start.below()).is(Blocks.WATER));
        var open = new PriorityQueue<Node>(Comparator.comparingInt(Node::score));
        Map<BlockPos, Integer> costs = new HashMap<>();
        Map<BlockPos, BlockPos> parents = new HashMap<>();
        Map<BlockPos, Double> heights = new HashMap<>();
        double initialY = standingY(level, start);
        if (!Double.isFinite(initialY) && (startInWater || goal != null)) {
            initialY = start.getY();
        }
        heights.put(start, initialY);
        costs.put(start, 0);
        open.add(new Node(start, 0, goal == null ? 0 : start.distManhattan(goal)));
        int budget = 3000;
        while (!open.isEmpty() && budget-- > 0) {
            Node current = open.remove();
            if (current.cost() != costs.getOrDefault(current.pos(), Integer.MAX_VALUE)) continue;
            if (goal == null ? current.pos().getY() <= start.getY() - 2 : current.pos().equals(goal)) {
                List<BlockPos> result = new ArrayList<>();
                BlockPos cursor = current.pos();
                while (!cursor.equals(start)) {
                    result.add(cursor);
                    cursor = parents.get(cursor);
                }
                Collections.reverse(result);
                return result;
            }
            double fromY = heights.computeIfAbsent(current.pos(),
                    pos -> startInWater ? waterEscapeY(level, pos) : standingY(level, pos));
            if (!Double.isFinite(fromY)) continue;
            for (int[] direction : new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                for (int dy = 1; dy >= -maxDrop; dy--) {
                    BlockPos next = current.pos().offset(direction[0], dy, direction[1]);
                    int radius = goal == null ? 6 : 24;
                    if (Math.abs(next.getX() - start.getX()) > radius
                            || Math.abs(next.getZ() - start.getZ()) > radius) continue;
                    double nextY = heights.computeIfAbsent(next,
                            pos -> startInWater ? waterEscapeY(level, pos) : standingY(level, pos));
                    if (!Double.isFinite(nextY)
                            || nextY - fromY > 1.001 || fromY - nextY > maxDrop + 0.001) continue;
                    if (!clearAt(level, current.pos(), Math.max(fromY, nextY))
                            || !clearAt(level, next, Math.max(fromY, nextY))) continue;
                    if (nextY < fromY && !level.noBlockCollision(null,
                            new AABB(next.getX() + 0.2, nextY + 0.001, next.getZ() + 0.2,
                                    next.getX() + 0.8, fromY + 1.8, next.getZ() + 0.8))) continue;
                    int waterPenalty = (startInWater && !safe(level, next)) ? 4 : 0;
                    int cost = current.cost() + 1 + Math.abs(dy) + waterPenalty;
                    if (cost < costs.getOrDefault(next, Integer.MAX_VALUE)) {
                        costs.put(next, cost);
                        parents.put(next, current.pos());
                        open.add(new Node(next, cost, cost + (goal == null ? 0 : next.distManhattan(goal))));
                    }
                }
            }
        }
        return List.of();
    }

    boolean go(BlockPos destination) {
        return go(destination, 3);
    }

    boolean sprintTo(BlockPos destination) {
        boolean ok = go(destination);
        if (ok) sprinting = true;
        return ok;
    }

    boolean descend() {
        index = 0;
        stuck = 0;
        sprinting = false;
        previous = player.position();
        path = find(player.level(), player.blockPosition(), null, 12);
        return !path.isEmpty();
    }

    boolean go(BlockPos destination, int maxDrop) {
        index = 0;
        stuck = 0;
        sprinting = false;
        previous = player.position();
        path = find(player.level(), player.blockPosition(), destination, maxDrop);
        return !path.isEmpty() || (player.blockPosition().equals(destination) && safe(player.level(), destination));
    }

    boolean tick() {
        EntityPlayerActionPack pack = ((ServerPlayerInterface) player).getActionPack();
        if (index >= path.size()) {
            pack.setSprinting(false);
            pack.stopMovement();
            return true;
        }
        BlockPos waypoint = path.get(index);
        double waypointY = standingY(player.level(), waypoint);
        if (!Double.isFinite(waypointY) && player.isInWater()) {
            waypointY = waterEscapeY(player.level(), waypoint);
        }
        if (!Double.isFinite(waypointY)) {
            pack.setSprinting(false);
            throw new IllegalArgumentException("Path became unsafe");
        }
        Vec3 target = new Vec3(waypoint.getX() + 0.5, waypointY, waypoint.getZ() + 0.5);
        double horizontal = Math.hypot(target.x - player.getX(), target.z - player.getZ());
        if (horizontal < 0.35 && Math.abs(target.y - player.getY()) < 0.45 && (player.onGround() || player.isInWater())) {
            index++;
            pack.setSprinting(false);
            pack.stopMovement();
            return index >= path.size();
        }
        if (horizontal < 0.2 && player.getY() > target.y + 0.2 && !player.isInWater()) {
            pack.setSprinting(false);
            pack.stopMovement();
            return false;
        }
        if (player.position().distanceToSqr(previous) < 0.0004) stuck++;
        else stuck = 0;
        previous = player.position();
        if (stuck > 50) {
            pack.setSprinting(false);
            throw new IllegalArgumentException("Movement stuck; choose a different route");
        }
        float yaw = (float) Math.toDegrees(Math.atan2(-(target.x - player.getX()), target.z - player.getZ()));
        double dy = target.y - player.getEyeY();
        double dist = Math.sqrt((target.x - player.getX()) * (target.x - player.getX()) + (target.z - player.getZ()) * (target.z - player.getZ()));
        float pitch = (float) Math.clamp(-Math.toDegrees(Math.atan2(dy, Math.max(0.1, dist))), -50, 50);
        boolean canSprint = player.getFoodData().getFoodLevel() > 6;
        boolean sprintNow = canSprint && (sprinting || (path.size() - index > 2 && horizontal > 1.2));
        pack.setSprinting(sprintNow);
        pack.look(yaw, pitch).setForward(horizontal < 0.7 ? 0.45f : 1f);
        if ((target.y > player.getY() + 0.4 && player.onGround()) || player.isInWater()) {
            pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }
        return false;
    }
}
