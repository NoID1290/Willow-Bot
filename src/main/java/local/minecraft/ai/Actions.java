package local.minecraft.ai;

import carpet.fakes.ServerPlayerInterface;
import carpet.helpers.EntityPlayerActionPack;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.BlockTags;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.context.UseOnContext;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

final class Actions {
    private final ServerPlayer player;
    private final Navigator navigator;
    private final ProgressTracker progress = new ProgressTracker();
    private String action = "";
    private String id = "";
    private BlockPos target;
    private int entityId;
    private int ticks;
    private boolean walking;
    private boolean sleeping;
    private boolean attackingBlock;
    private int foodBefore;
    private JsonObject result = new JsonObject();
    private float healthBefore;
    private String recoveryMethod = "";
    private boolean escapeBlockBroken;
    private int recoveryTicks;
    private boolean pillaring;
    private boolean pillarPlaced;
    private net.minecraft.world.item.Item pickupItem;
    private int pickupBefore;
    private String targetName = "";
    private net.minecraft.world.level.block.Block targetBlock;
    private int miningRepositions;
    private List<BlockPos> watched = List.of();
    private final SightPerception sightPerception = new SightPerception();
    private final LiveReflexState liveReflex = new LiveReflexState();
    private int improperToolMineAttempts = 0;
    private int blocksMinedCount = 0;
    private int pathfindingTimeouts = 0;
    private final List<String> toolsBroken = new ArrayList<>();
    private String trackedMainHandTool = "";
    private String currentGoal = "";
    private String lastSpeech = "";

    Actions(ServerPlayer player) {
        this.player = player;
        navigator = new Navigator(player);
    }

    EntityPlayerActionPack pack() {
        return ((ServerPlayerInterface) player).getActionPack();
    }

    boolean busy() {
        return !action.isEmpty();
    }

    void stop(String reason) {
        stop(reason, false);
    }

    /** Ends the current action; {@code failed} marks a real failure (e.g. death) rather than an interruption. */
    void stop(String reason, boolean failed) {
        if (busy()) finish(failed ? "failed" : "interrupted", reason);
        pack().stopAll();
        pack().setSprinting(false);
        pack().setSneaking(false);
        pillaring = false;
        pillarPlaced = false;
        player.stopUsingItem();
    }

    JsonObject result() {
        return result.deepCopy();
    }

    void restoreResult(JsonObject previous) {
        result = previous.deepCopy();
    }

    void setGoalAndSpeech(String goal, String speech) {
        if (goal != null) currentGoal = goal;
        if (speech != null) lastSpeech = speech;
    }

    JsonObject doing() {
        JsonObject d = new JsonObject();
        d.addProperty("action", action.isEmpty() ? "idle" : action);
        d.addProperty("target", targetName.isEmpty() ? (entityId != 0 ? String.valueOf(entityId) : "") : targetName);
        if (target != null) {
            JsonArray tp = new JsonArray();
            tp.add(target.getX());
            tp.add(target.getY());
            tp.add(target.getZ());
            d.add("target_pos", tp);
        }
        d.addProperty("ticks_active", ticks);
        d.addProperty("walking", walking);
        d.addProperty("attacking_block", attackingBlock);
        d.addProperty("pillaring", pillaring);

        String progressStr;
        if (walking) {
            progressStr = "Navigating to " + (target != null ? target.toShortString() : "destination");
        } else if (attackingBlock) {
            progressStr = "Mining " + targetName + " at " + (target != null ? target.toShortString() : "");
        } else if (pillaring) {
            progressStr = "Pillaring under feet";
        } else if (!action.isEmpty()) {
            progressStr = "Executing " + action + " (" + ticks + " ticks)";
        } else {
            progressStr = "Idle / observing";
        }
        d.addProperty("progress", progressStr);

        String stance = player.isInWater() ? "swimming"
                : (player.isCrouching() ? "sneaking"
                : (player.isSprinting() ? "sprinting"
                : (player.onGround() ? "ground" : "falling")));
        d.addProperty("stance", stance);

        ItemStack main = player.getMainHandItem();
        JsonObject mh = new JsonObject();
        mh.addProperty("item", main.isEmpty() ? "minecraft:air" : BuiltInRegistries.ITEM.getKey(main.getItem()).toString());
        mh.addProperty("count", main.getCount());
        d.add("main_hand", mh);

        ItemStack off = player.getOffhandItem();
        JsonObject oh = new JsonObject();
        oh.addProperty("item", off.isEmpty() ? "minecraft:air" : BuiltInRegistries.ITEM.getKey(off.getItem()).toString());
        oh.addProperty("count", off.getCount());
        d.add("off_hand", oh);

        d.addProperty("active_goal", currentGoal);
        d.addProperty("last_speech", lastSpeech);
        return d;
    }

    void start(JsonObject decision) {
        pack().stopAll();
        pack().setSprinting(false);
        pack().setSneaking(false);
        ticks = 0;
        target = null;
        targetBlock = null;
        walking = false;
        pillaring = false;
        pillarPlaced = false;
        attackingBlock = false;
        healthBefore = player.getHealth();
        recoveryMethod = "";
        escapeBlockBroken = false;
        recoveryTicks = 0;
        miningRepositions = 0;
        sleeping = false;
        id = decision.get("id").getAsString();
        action = decision.get("action").getAsString();
        String name = decision.get("target").getAsString();
        targetName = name;
        if (decision.has("goal")) currentGoal = decision.get("goal").getAsString();
        if (decision.has("say")) lastSpeech = decision.get("say").getAsString();
        if (decision.has("watch") && decision.get("watch").isJsonArray()) {
            var positions = new ArrayList<BlockPos>();
            for (var element : decision.getAsJsonArray("watch")) {
                if (positions.size() >= 64 || !element.isJsonArray() || element.getAsJsonArray().size() != 3) continue;
                var at = element.getAsJsonArray();
                positions.add(new BlockPos(at.get(0).getAsInt(), at.get(1).getAsInt(), at.get(2).getAsInt()));
            }
            watched = List.copyOf(positions);
        }
        try {
            switch (action) {
                case "wait" -> { }
                case "jump" -> {
                    BlockPos heading = coordinatesOrZero(decision);
                    if (heading != null) {
                        float yaw = (float) Math.toDegrees(Math.atan2(-(heading.getX() + 0.5 - player.getX()), heading.getZ() + 0.5 - player.getZ()));
                        pack().look(yaw, 0).setForward(1.0f);
                    }
                    pack().start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
                }
                case "run", "sprint" -> {
                    if (player.getFoodData().getFoodLevel() <= 6) {
                        throw new IllegalArgumentException("Cannot sprint: food level is too low (" + player.getFoodData().getFoodLevel() + ")");
                    }
                    BlockPos dest = coordinatesOrZero(decision);
                    if (dest != null) {
                        if (!navigator.sprintTo(dest)) throw new IllegalArgumentException("No safe running path to " + dest.toShortString());
                        walking = true;
                    } else {
                        pack().setSprinting(true).setForward(1.0f);
                    }
                }
                case "sneak", "crouch" -> pack().setSneaking(true);
                case "pillar" -> {
                    BlockPos pos = coordinatesOrZero(decision);
                    if (pos == null) pos = player.blockPosition();
                    pillarUp(name, pos);
                }
                case "craft" -> finish(true, Crafting.craft(player, name));
                case "eat" -> {
                    foodBefore = player.getFoodData().getFoodLevel();
                    if (!player.getFoodData().needsFood()) throw new IllegalArgumentException("Not hungry");
                    if (!selectFood()) throw new IllegalArgumentException("No edible food in inventory");
                    player.getMainHandItem().use(player.level(), player, InteractionHand.MAIN_HAND);
                    if (!player.isUsingItem()) throw new IllegalArgumentException("Food cannot be used");
                }
                case "goto" -> {
                    BlockPos pos = new BlockPos(decision.get("x").getAsInt(),
                            decision.get("y").getAsInt(), decision.get("z").getAsInt());
                    if (player.blockPosition().distManhattan(pos) > 48
                            || Math.abs(pos.getX() - player.getX()) > 24
                            || Math.abs(pos.getZ() - player.getZ()) > 24
                            || !player.level().isLoaded(pos)) {
                        explore(pos);
                        recoveryMethod = "Global pathfinding step toward " + pos.toShortString();
                    } else {
                        walkTo(pos);
                    }
                }
                case "explore" -> {
                    BlockPos heading = headingOrZero(decision);
                    explore(heading);
                }
                case "escape" -> escape();
                case "mine" -> {
                    BlockPos pos = coordinatesOrZero(decision);
                    mine(name, pos);
                }
                case "place" -> place(name, coordinates(decision));
                case "smelt" -> smelt(name);
                case "chest" -> useChest(name);
                case "sleep" -> sleepInBed();
                case "trade" -> tradeWithVillager(name);
                case "boat" -> useBoat();
                case "tunnel" -> tunnelForward(headingOrZero(decision));
                case "bridge" -> bridgeForward(headingOrZero(decision));
                case "shelter" -> buildShelter();
                case "portal" -> usePortal();
                case "pickup" -> {
                    boolean found = false;
                    for (Entity entity : nearbyEntities().stream().filter(e -> e instanceof ItemEntity item
                            && canHold(item.getItem())
                            && (name.isEmpty() ? !isJunkDrop(item.getItem()) : BuiltInRegistries.ITEM.getKey(item.getItem().getItem())
                                    .toString().equals(name))).limit(8).toList()) {
                        if (!followDrop(entity)) continue;
                        entityId = entity.getId();
                        pickupItem = ((ItemEntity) entity).getItem().getItem();
                        pickupBefore = player.getInventory().countItem(pickupItem);
                        found = true;
                        break;
                    }
                    if (!found) throw new IllegalArgumentException("No reachable pickup position or inventory space near item drops");
                }
                case "attack" -> {
                    try {
                        entityId = Integer.parseInt(name);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("Attack target must be an observed numeric entity ID");
                    }
                    Entity entity = player.level().getEntity(entityId);
                    if (!(entity instanceof LivingEntity) || entity instanceof Player
                            || player.distanceToSqr(entity) > 144) {
                        throw new IllegalArgumentException("Target missing, too far away, or a player");
                    }
                    if (player.distanceToSqr(entity) > 8) approach(entity.blockPosition(), 2);
                }
                default -> throw new IllegalArgumentException("Unsupported action: " + action);
            }
        } catch (IllegalArgumentException e) {
            finish(false, e.getMessage());
        }
    }

    private boolean hasCoordinates(JsonObject decision) {
        return decision != null && decision.has("x") && decision.has("y") && decision.has("z")
                && !decision.get("x").isJsonNull() && !decision.get("y").isJsonNull() && !decision.get("z").isJsonNull();
    }

    private BlockPos headingOrZero(JsonObject decision) {
        if (!hasCoordinates(decision)) return null;
        return new BlockPos(decision.get("x").getAsInt(), decision.get("y").getAsInt(), decision.get("z").getAsInt());
    }

    private BlockPos coordinatesOrZero(JsonObject decision) {
        if (!hasCoordinates(decision)) return null;
        BlockPos pos = new BlockPos(decision.get("x").getAsInt(), decision.get("y").getAsInt(), decision.get("z").getAsInt());
        if (player.blockPosition().distManhattan(pos) > 48
                || Math.abs(pos.getX() - player.getX()) > 24
                || Math.abs(pos.getZ() - player.getZ()) > 24
                || !player.level().isLoaded(pos)) return null;
        return pos;
    }

    private BlockPos coordinates(JsonObject decision) {
        BlockPos pos = new BlockPos(decision.get("x").getAsInt(),
                decision.get("y").getAsInt(), decision.get("z").getAsInt());
        if (player.blockPosition().distManhattan(pos) > 48
                || Math.abs(pos.getX() - player.getX()) > 24
                || Math.abs(pos.getZ() - player.getZ()) > 24
                || !player.level().isLoaded(pos)) throw new IllegalArgumentException("Coordinate outside local loaded area");
        return pos;
    }

    private void walkTo(BlockPos pos) {
        if (player.blockPosition().equals(pos)) {
            throw new IllegalArgumentException("Already at destination " + pos.toShortString());
        }
        if (navigator.go(pos)) {
            walking = true;
            return;
        }
        try {
            approach(pos, 2, false);
            return;
        } catch (IllegalArgumentException ignored) {
        }
        try {
            approach(pos, 3, false);
            return;
        } catch (IllegalArgumentException ignored) {
        }
        throw new IllegalArgumentException("No safe walking path to " + pos.toShortString());
    }

    private boolean canHold(ItemStack stack) {
        if (stack.isEmpty()) return false;
        for (int i = 0; i < 36; i++) {
            ItemStack slot = player.getInventory().getItem(i);
            if (slot.isEmpty()) return true;
            if (ItemStack.isSameItemSameComponents(slot, stack) && slot.getCount() < slot.getMaxStackSize()) {
                return true;
            }
        }
        return false;
    }

    private boolean isJunkDrop(ItemStack stack) {
        String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
        return id.equals("minecraft:rotten_flesh") || id.equals("minecraft:spider_eye")
                || id.equals("minecraft:poisonous_potato") || id.equals("minecraft:leaf_litter");
    }

    private boolean followDrop(Entity item) {
        if (player.getBoundingBox().inflate(0.75, 0.5, 0.75).intersects(item.getBoundingBox())) return true;
        List<BlockPos> candidates = new ArrayList<>();
        BlockPos.betweenClosedStream(item.blockPosition().offset(-1, -2, -1),
                item.blockPosition().offset(1, 1, 1)).forEach(pos -> {
            double y = Navigator.standingY(player.level(), pos);
            if (Double.isFinite(y) && item.getY() >= y - 0.5 && item.getY() <= y + 1.8
                    && Math.abs(pos.getX() + 0.5 - item.getX()) <= 1.1
                    && Math.abs(pos.getZ() + 0.5 - item.getZ()) <= 1.1) candidates.add(pos.immutable());
        });
        candidates.sort(Comparator.comparingDouble(pos -> Vec3.atCenterOf(pos).distanceToSqr(item.position())));
        for (BlockPos pos : candidates.stream().limit(6).toList()) {
            if (navigator.go(pos)) {
                walking = true;
                return true;
            }
        }
        return ticks < 40 && !item.isNoGravity() && !item.onGround() && item.getY() > player.getY();
    }

    private void explore(BlockPos heading) {
        int offset = player.getRandom().nextInt(8);
        List<Double> angles = new ArrayList<>();
        for (int n = 0; n < 8; n++) angles.add((n + offset) * Math.PI / 4);
        if (heading != null) {
            // Travel toward a far remembered destination: try the most aligned directions first.
            double wanted = Math.atan2(heading.getZ() - player.getZ(), heading.getX() - player.getX());
            angles.clear();
            for (int n = 0; n < 8; n++) angles.add(wanted + n * Math.PI / 4);
            angles.sort(Comparator.comparingDouble(a -> Math.abs(Math.IEEEremainder(a - wanted, 2 * Math.PI))));
            recoveryMethod = "Heading toward " + heading.toShortString();
        }
        for (double angle : angles) {
            for (int distance : new int[]{20, 16, 12, 8, 5}) {
                for (int dy : new int[]{0, 1, -1, 2, -2, 3, -3}) {
                    BlockPos destination = player.blockPosition().offset(
                            (int) Math.round(Math.cos(angle) * distance), dy,
                            (int) Math.round(Math.sin(angle) * distance));
                    if (navigator.go(destination)) {
                        walking = true;
                        return;
                    }
                }
            }
        }
        throw new IllegalArgumentException("No safe local exploration path");
    }

    private void escape() {
        BlockPos start = player.blockPosition();
        if (navigator.descend()) {
            walking = true;
            recoveryMethod = "Risky descent from " + start.toShortString();
            LocalAiMod.LOG.info("Willow recovery: {}", recoveryMethod);
            return;
        }
        List<BlockPos> obstructions = new ArrayList<>();
        BlockPos.betweenClosedStream(start.offset(-2, -1, -2), start.offset(2, 2, 2))
                .filter(pos -> {
                    var state = player.level().getBlockState(pos);
                    return state.is(BlockTags.LEAVES) || state.is(BlockTags.LOGS)
                            || state.is(BlockTags.DIRT) || state.is(Blocks.SNOW);
                }).filter(pos -> visible(player, pos)).forEach(pos -> obstructions.add(pos.immutable()));
        obstructions.sort(Comparator.comparingDouble(pos -> pos.distSqr(start)));
        if (progress.recoveryAttempts() % 2 == 1 && obstructions.remove(start.below())) {
            obstructions.addFirst(start.below());
        }
        if (!obstructions.isEmpty()) {
            target = obstructions.getFirst();
            targetName = BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(target).getBlock()).toString();
            recoveryMethod = "Breaking escape obstruction/support " + target.toShortString();
            LocalAiMod.LOG.info("Willow recovery: {}", recoveryMethod);
            return;
        }
        // If trapped in a cave, hole, or pit without soft natural obstructions, pillar up to gain height and escape!
        if (canPlaceUnder(start)) {
            int slot = findBuildingBlockSlot("");
            if (slot >= 0) {
                String blockName = BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(slot).getItem()).toString();
                pillarUp(blockName, start);
                recoveryMethod = "Pillaring up out of cave/pit from " + start.toShortString() + " with " + blockName;
                LocalAiMod.LOG.info("Willow recovery: {}", recoveryMethod);
                return;
            }
        }
        // If headroom is obstructed by a ceiling block directly above in a cave, break it to make headroom
        BlockPos ceiling = start.above(2);
        if (!player.level().getBlockState(ceiling).canBeReplaced()
                && player.level().getBlockState(ceiling).getDestroySpeed(player.level(), ceiling) >= 0) {
            target = ceiling;
            targetName = BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(ceiling).getBlock()).toString();
            recoveryMethod = "Breaking ceiling block " + target.toShortString() + " to create headroom to escape cave";
            LocalAiMod.LOG.info("Willow recovery: {}", recoveryMethod);
            return;
        }
        // Try breaking mineable stone/deepslate/wall obstructions
        List<BlockPos> hardObstructions = new ArrayList<>();
        BlockPos.betweenClosedStream(start.offset(-2, -1, -2), start.offset(2, 2, 2))
                .filter(pos -> {
                    var state = player.level().getBlockState(pos);
                    if (state.getDestroySpeed(player.level(), pos) < 0) return false;
                    return state.is(BlockTags.MINEABLE_WITH_PICKAXE) || state.is(BlockTags.MINEABLE_WITH_SHOVEL);
                }).filter(pos -> visible(player, pos)).forEach(pos -> hardObstructions.add(pos.immutable()));
        hardObstructions.sort(Comparator.comparingDouble(pos -> pos.distSqr(start)));
        if (!hardObstructions.isEmpty()) {
            target = hardObstructions.getFirst();
            targetName = BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(target).getBlock()).toString();
            recoveryMethod = "Mining cave wall/support " + target.toShortString() + " to escape";
            LocalAiMod.LOG.info("Willow recovery: {}", recoveryMethod);
            return;
        }
        throw new IllegalArgumentException("Recovery found no reachable landing within a 12-block drop, breakable obstruction, or pillaring route");
    }

    private void approach(BlockPos pos, int radius) {
        approach(pos, radius, false);
    }

    private void approach(BlockPos pos, int radius, boolean mining) {
        BlockPos start = player.blockPosition();
        double currentDistSqr = start.distSqr(pos);
        List<BlockPos> candidates = new ArrayList<>();
        BlockPos.betweenClosedStream(pos.offset(-radius, -4, -radius), pos.offset(radius, 1, radius))
                .filter(p -> !p.equals(start) && Navigator.safe(player.level(), p))
                .filter(p -> mining ? miningView(p, pos) : p.distSqr(pos) < currentDistSqr)
                .forEach(p -> candidates.add(p.immutable()));
        candidates.sort(mining
                ? Comparator.comparingDouble(p -> p.distSqr(start))
                : Comparator.comparingDouble((BlockPos p) -> p.distSqr(pos)).thenComparingDouble(p -> p.distSqr(start)));
        for (BlockPos candidate : candidates) {
            if (navigator.go(candidate)) {
                walking = true;
                return;
            }
        }
        throw new IllegalArgumentException("Cannot approach target safely");
    }

    private boolean miningView(BlockPos feet, BlockPos block) {
        double y = Navigator.standingY(player.level(), feet) + player.getEyeHeight();
        Vec3 centerEye = new Vec3(feet.getX() + 0.5, y, feet.getZ() + 0.5);
        if (visibleFrom(player, centerEye, block)) return true;
        for (double dx : new double[]{0.2, 0.8}) {
            for (double dz : new double[]{0.2, 0.8}) {
                if (visibleFrom(player, new Vec3(feet.getX() + dx, y, feet.getZ() + dz), block)) return true;
            }
        }
        if (player.level().getBlockState(block).is(BlockTags.LOGS)
                && centerEye.distanceToSqr(Vec3.atCenterOf(block)) <= 18.0) {
            BlockHitResult ray = player.level().clip(new ClipContext(centerEye, Vec3.atCenterOf(block),
                    ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
            if (ray.getType() == HitResult.Type.BLOCK
                    && player.level().getBlockState(ray.getBlockPos()).is(BlockTags.LEAVES)) {
                return true;
            }
        }
        return false;
    }

    private void mine(String name, BlockPos requested) {
        List<BlockPos> candidates = nearbyBlocks().stream()
                .filter(p -> BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(p).getBlock())
                        .toString().equals(name))
                .filter(p -> requested == null || p.equals(requested)).toList();
        if (candidates.isEmpty() && requested != null && player.level().isLoaded(requested)) {
            var state = player.level().getBlockState(requested);
            if (BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString().equals(name)) {
                candidates = List.of(requested);
            }
        }
        if (candidates.isEmpty() && requested != null) {
            candidates = nearbyBlocks().stream()
                    .filter(p -> BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(p).getBlock())
                            .toString().equals(name)).toList();
        }
        boolean missingRequiredTool = false;
        for (BlockPos candidate : candidates) {
            var state = player.level().getBlockState(candidate);
            if (state.getDestroySpeed(player.level(), candidate) < 0
                    || candidate.equals(player.blockPosition().below())) continue;
            if (state.requiresCorrectToolForDrops() && !hasTool(candidate)) {
                missingRequiredTool = true;
                continue;
            }
            target = candidate;
            if (visible(player, target)) return;
            try {
                approach(target, 3, true);
                return;
            } catch (IllegalArgumentException e) {
                if (state.is(BlockTags.LOGS)) {
                    try {
                        approach(target, 2, false);
                        return;
                    } catch (IllegalArgumentException ignored) {
                    }
                }
            }
        }
        if (missingRequiredTool) {
            improperToolMineAttempts++;
        }
        throw new IllegalArgumentException("No reachable matching block with a suitable tool");
    }

    private boolean hasTool(BlockPos pos) {
        for (int i = 0; i < 36; i++) {
            if (player.getInventory().getItem(i).isCorrectToolForDrops(player.level().getBlockState(pos))) return true;
        }
        return false;
    }

    private void selectTool(BlockPos pos) {
        int bestSlot = player.getInventory().getSelectedSlot();
        float best = -1;
        var state = player.level().getBlockState(pos);
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (state.requiresCorrectToolForDrops() && !stack.isCorrectToolForDrops(state)) continue;
            float speed = stack.getDestroySpeed(state);
            if (speed > best) {
                best = speed;
                bestSlot = i;
            }
        }
        select(bestSlot);
    }

    private void select(int slot) {
        var inventory = player.getInventory();
        if (slot < 9) {
            inventory.setSelectedSlot(slot);
        } else {
            int selected = inventory.getSelectedSlot();
            ItemStack old = inventory.getItem(selected);
            inventory.setItem(selected, inventory.getItem(slot));
            inventory.setItem(slot, old);
        }
        player.inventoryMenu.broadcastChanges();
    }

    void selectHotbarOrSwap(int slot) {
        select(slot);
    }

    void equipBestWeapon() {
        int bestSlot = -1;
        int bestRank = -1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.isEmpty()) continue;
            String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).getPath();
            int rank = -1;
            if (id.equals("netherite_sword")) rank = 60;
            else if (id.equals("diamond_sword")) rank = 50;
            else if (id.equals("iron_sword")) rank = 40;
            else if (id.equals("stone_sword")) rank = 30;
            else if (id.equals("wooden_sword")) rank = 20;
            else if (id.endsWith("_axe")) rank = 15;
            if (rank > bestRank) {
                bestRank = rank;
                bestSlot = i;
            }
        }
        if (bestSlot >= 0) select(bestSlot);
    }

    boolean selectHighestSaturationFood() {
        int bestSlot = -1;
        float bestScore = -1.0f;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.isEmpty()) continue;
            String name = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
            var food = stack.get(DataComponents.FOOD);
            if (food != null && !name.contains("rotten")
                    && !name.contains("spider_eye") && !name.contains("pufferfish")
                    && !name.contains("poisonous") && !name.contains("suspicious")) {
                float score = food.saturation() * 2.0f + food.nutrition();
                if (score > bestScore) {
                    bestScore = score;
                    bestSlot = i;
                }
            }
        }
        if (bestSlot >= 0) {
            select(bestSlot);
            return true;
        }
        return false;
    }

    private boolean selectFood() {
        return selectHighestSaturationFood();
    }

    private boolean canPlaceUnder(BlockPos pos) {
        if (!pos.equals(player.blockPosition())) return false;
        if (!player.level().isLoaded(pos) || !player.level().getBlockState(pos).canBeReplaced()) return false;
        BlockPos support = pos.below();
        if (player.level().getBlockState(support).getCollisionShape(player.level(), support).isEmpty()) return false;
        if (!player.level().getBlockState(pos.above()).canBeReplaced()
                || !player.level().getBlockState(pos.above(2)).canBeReplaced()) return false;
        return player.onGround() || (player.getDeltaMovement().y <= 0.05 && Math.abs(player.getY() - pos.getY()) < 0.2);
    }

    private int findBuildingBlockSlot(String requestedName) {
        var inventory = player.getInventory();
        if (requestedName != null && !requestedName.isEmpty()) {
            for (int i = 0; i < 36; i++) {
                ItemStack stack = inventory.getItem(i);
                if (stack.getItem() instanceof BlockItem && BuiltInRegistries.ITEM.getKey(stack.getItem())
                        .toString().equals(requestedName)) {
                    return i;
                }
            }
        }
        int fallbackSlot = -1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = inventory.getItem(i);
            if (!(stack.getItem() instanceof BlockItem blockItem)) continue;
            var block = blockItem.getBlock();
            var shape = block.defaultBlockState().getCollisionShape(player.level(), player.blockPosition());
            if (shape.isEmpty()) continue;
            String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
            if (id.contains("sand") || id.contains("gravel") || id.contains("anvil") || id.contains("tnt")) continue;
            if (id.equals("minecraft:cobblestone") || id.equals("minecraft:dirt")
                    || id.equals("minecraft:cobbled_deepslate") || id.equals("minecraft:stone")
                    || id.contains("planks") || id.contains("diorite") || id.contains("andesite") || id.contains("granite")) {
                return i;
            }
            if (fallbackSlot == -1 && !id.contains("crafting_table") && !id.contains("furnace") && !id.contains("chest")) {
                fallbackSlot = i;
            }
        }
        return fallbackSlot;
    }

    private void pillarUp(String name, BlockPos pos) {
        if (!canPlaceUnder(pos)) {
            throw new IllegalArgumentException("Cannot place block under feet: need solid ground, headroom, and to be standing on ground");
        }
        int slot = findBuildingBlockSlot(name);
        if (slot < 0) throw new IllegalArgumentException("Requested or suitable block not in inventory" + (name.isEmpty() ? "" : ": " + name));
        name = BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(slot).getItem()).toString();
        select(slot);
        target = pos;
        targetName = name;
        pillaring = true;
        pillarPlaced = false;
        pack().stopMovement();
        pack().look(player.getYRot(), 90.0f);
        pack().start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
    }

    private void place(String name, BlockPos pos) {
        if (pos.equals(player.blockPosition())) {
            pillarUp(name, pos);
            return;
        }
        if (!player.level().getBlockState(pos).canBeReplaced()
                || player.getBoundingBox().intersects(new net.minecraft.world.phys.AABB(pos))) {
            throw new IllegalArgumentException("Placement space is occupied");
        }
        int slot = -1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.getItem() instanceof BlockItem && BuiltInRegistries.ITEM.getKey(stack.getItem())
                    .toString().equals(name)) {
                slot = i;
                break;
            }
        }
        if (slot < 0) throw new IllegalArgumentException("Requested block not in inventory");
        BlockHitResult ray = placementHit(pos);
        if (ray == null) throw new IllegalArgumentException("No reachable visible support face for placement");
        select(slot);
        pack().lookAt(ray.getLocation());
        player.getMainHandItem().useOn(new UseOnContext(player, InteractionHand.MAIN_HAND, ray));
        if (player.level().getBlockState(pos).canBeReplaced()) {
            throw new IllegalArgumentException("Block placement rejected by Minecraft");
        }
        target = pos;
        finish(true, "Placed " + name + " at " + pos.toShortString());
    }

    private BlockHitResult placementHit(BlockPos pos) {
        if (!player.level().isLoaded(pos) || !player.level().getBlockState(pos).canBeReplaced()
                || player.getBoundingBox().intersects(new net.minecraft.world.phys.AABB(pos))
                || !player.level().noEntityCollision(player, new net.minecraft.world.phys.AABB(pos))) return null;
        for (Direction direction : Direction.values()) {
            BlockPos support = pos.relative(direction);
            if (player.level().getBlockState(support).getCollisionShape(player.level(), support).isEmpty()) continue;
            Vec3 hitPoint = Vec3.atCenterOf(support).add(Vec3.atLowerCornerOf(direction.getOpposite().getUnitVec3i()).scale(0.5));
            if (player.getEyePosition().distanceToSqr(hitPoint) > 20.25) continue;
            BlockHitResult ray = player.level().clip(new ClipContext(player.getEyePosition(), Vec3.atCenterOf(support),
                    ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
            if (ray.getType() != HitResult.Type.BLOCK || !ray.getBlockPos().equals(support)
                    || ray.getDirection() != direction.getOpposite()) continue;
            return ray;
        }
        return null;
    }

    private List<BlockPos> placementSites() {
        List<BlockPos> sites = new ArrayList<>();
        BlockPos feet = player.blockPosition();
        BlockPos.betweenClosedStream(feet.offset(-3, -2, -3),
                feet.offset(3, 2, 3))
                .filter(pos -> !pos.equals(feet) && placementHit(pos) != null)
                .forEach(pos -> sites.add(pos.immutable()));
        sites.sort(Comparator.comparingDouble(pos -> pos.distSqr(feet)));
        List<BlockPos> result = new ArrayList<>(sites.stream().limit(4).toList());
        if (canPlaceUnder(feet)) {
            result.add(feet);
        }
        return result;
    }

    private void smelt(String requestedItem) {
        if (!Crafting.furnaceReachable(player)) {
            BlockPos nearbyFurnace = nearbyBlocks().stream()
                    .filter(p -> {
                        var s = player.level().getBlockState(p);
                        return s.is(Blocks.FURNACE) || s.is(Blocks.BLAST_FURNACE) || s.is(Blocks.SMOKER);
                    })
                    .findFirst().orElse(null);
            if (nearbyFurnace != null) {
                approach(nearbyFurnace, 2);
                recoveryMethod = "Approaching furnace at " + nearbyFurnace.toShortString();
                return;
            }
            List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(player.blockPosition())).toList();
            if (!sites.isEmpty() && player.getInventory().countItem(net.minecraft.world.item.Items.FURNACE) > 0) {
                place("minecraft:furnace", sites.getFirst());
                return;
            }
            throw new IllegalArgumentException("No reachable furnace to smelt items");
        }
        int inputSlot = -1;
        int fuelSlot = -1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.isEmpty()) continue;
            String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
            if (inputSlot < 0 && !Crafting.smeltResult(stack).isEmpty()
                    && (requestedItem.isEmpty() || id.equals(requestedItem))) {
                inputSlot = i;
            }
        }
        if (inputSlot < 0) throw new IllegalArgumentException("Smeltable item not found in inventory: " + requestedItem);
        for (int i = 0; i < 36; i++) {
            if (i == inputSlot && player.getInventory().getItem(i).getCount() <= 1) continue;
            if (Crafting.isFuel(player.getInventory().getItem(i))) {
                fuelSlot = i;
                break;
            }
        }
        if (fuelSlot < 0) throw new IllegalArgumentException("No furnace fuel (coal, charcoal, planks, logs, sticks) in inventory");
        ItemStack inputStack = player.getInventory().getItem(inputSlot);
        ItemStack fuelStack = player.getInventory().getItem(fuelSlot);
        ItemStack resultProto = Crafting.smeltResult(inputStack);
        boolean isLava = fuelStack.is(Items.LAVA_BUCKET);
        int maxCapacity = Crafting.fuelCapacity(fuelStack);
        int batch = Math.min(inputStack.getCount(), maxCapacity);
        player.getInventory().removeItem(fuelSlot, 1);
        if (isLava) {
            player.getInventory().placeItemBackInInventory(new ItemStack(Items.BUCKET), net.minecraft.util.Prediction.SERVER_ONLY);
        }
        player.getInventory().removeItem(inputSlot, batch);
        ItemStack smelted = resultProto.copyWithCount(batch);
        player.getInventory().placeItemBackInInventory(smelted, net.minecraft.util.Prediction.SERVER_ONLY);
        player.inventoryMenu.broadcastChanges();
        finish(true, "Smelted " + batch + " " + BuiltInRegistries.ITEM.getKey(resultProto.getItem()));
    }

    private void useChest(String requestedItem) {
        BlockPos chestPos = BlockPos.betweenClosedStream(
                player.blockPosition().offset(-4, -2, -4),
                player.blockPosition().offset(4, 2, 4))
                .filter(p -> {
                    var s = player.level().getBlockState(p);
                    return (s.is(Blocks.CHEST) || s.is(Blocks.TRAPPED_CHEST) || s.is(Blocks.BARREL)) && visible(player, p);
                })
                .map(BlockPos::immutable)
                .findFirst().orElse(null);
        if (chestPos == null) {
            BlockPos farChest = nearbyBlocks().stream()
                    .filter(p -> {
                        var s = player.level().getBlockState(p);
                        return s.is(Blocks.CHEST) || s.is(Blocks.TRAPPED_CHEST) || s.is(Blocks.BARREL);
                    })
                    .findFirst().orElse(null);
            if (farChest != null) {
                approach(farChest, 2);
                recoveryMethod = "Approaching chest at " + farChest.toShortString();
                return;
            }
            List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(player.blockPosition())).toList();
            if (!sites.isEmpty() && player.getInventory().countItem(net.minecraft.world.item.Items.CHEST) > 0) {
                place("minecraft:chest", sites.getFirst());
                return;
            }
            throw new IllegalArgumentException("No reachable chest or barrel nearby");
        }
        if (!(player.level().getBlockEntity(chestPos) instanceof net.minecraft.world.Container container)) {
            throw new IllegalArgumentException("Target block at " + chestPos.toShortString() + " is not a valid container");
        }
        // First try storing requested item (or excess bulk items when requestedItem is empty)
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.isEmpty()) continue;
            String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
            boolean shouldStore = !requestedItem.isEmpty()
                    ? id.equals(requestedItem)
                    : (stack.getCount() >= 32 && (id.equals("minecraft:cobblestone") || id.equals("minecraft:dirt")
                            || id.equals("minecraft:cobbled_deepslate") || id.equals("minecraft:gravel") || id.equals("minecraft:sand")));
            if (!shouldStore) continue;
            for (int c = 0; c < container.getContainerSize(); c++) {
                ItemStack cStack = container.getItem(c);
                if (cStack.isEmpty()) {
                    ItemStack moved = player.getInventory().removeItem(i, stack.getCount());
                    container.setItem(c, moved);
                    container.setChanged();
                    player.inventoryMenu.broadcastChanges();
                    finish(true, "Stored " + moved.getCount() + " " + id + " in chest at " + chestPos.toShortString());
                    return;
                }
            }
        }
        // Otherwise retrieve requested item (or first available stack if requestedItem is empty) from chest
        for (int c = 0; c < container.getContainerSize(); c++) {
            ItemStack cStack = container.getItem(c);
            if (cStack.isEmpty()) continue;
            String id = BuiltInRegistries.ITEM.getKey(cStack.getItem()).toString();
            if (!requestedItem.isEmpty() && !id.equals(requestedItem)) continue;
            if (!canHold(cStack)) continue;
            ItemStack taken = container.removeItem(c, cStack.getCount());
            container.setChanged();
            player.getInventory().placeItemBackInInventory(taken, net.minecraft.util.Prediction.SERVER_ONLY);
            player.inventoryMenu.broadcastChanges();
            finish(true, "Retrieved " + taken.getCount() + " " + id + " from chest at " + chestPos.toShortString());
            return;
        }
        throw new IllegalArgumentException("No matching items could be stored in or retrieved from chest");
    }

    private void sleepInBed() {
        var bedRule = player.level().environmentAttributes().getValue(net.minecraft.world.attribute.EnvironmentAttributes.BED_RULE, player.blockPosition());
        if (bedRule != null && bedRule.destroyOnUse()) {
            throw new IllegalArgumentException("Beds explode in this dimension");
        }
        BlockPos bedPos = BlockPos.betweenClosedStream(
                player.blockPosition().offset(-4, -2, -4),
                player.blockPosition().offset(4, 2, 4))
                .filter(p -> player.level().getBlockState(p).is(BlockTags.BEDS) && visible(player, p))
                .map(BlockPos::immutable)
                .findFirst().orElse(null);
        if (bedPos == null) {
            BlockPos farBed = nearbyBlocks().stream()
                    .filter(p -> player.level().getBlockState(p).is(BlockTags.BEDS))
                    .findFirst().orElse(null);
            if (farBed != null) {
                approach(farBed, 2);
                recoveryMethod = "Approaching bed at " + farBed.toShortString();
                return;
            }
            for (int i = 0; i < 36; i++) {
                ItemStack stack = player.getInventory().getItem(i);
                if (!stack.isEmpty() && stack.is(net.minecraft.tags.ItemTags.BEDS)) {
                    List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(player.blockPosition())).toList();
                    if (!sites.isEmpty()) {
                        place(BuiltInRegistries.ITEM.getKey(stack.getItem()).toString(), sites.getFirst());
                        return;
                    }
                }
            }
            throw new IllegalArgumentException("No reachable bed nearby and no bed placeable");
        }
        BlockHitResult hit = new BlockHitResult(Vec3.atCenterOf(bedPos), Direction.UP, bedPos, false);
        player.level().getBlockState(bedPos).useWithoutItem(player.level(), player, hit);
    }

    private List<String> availableTrades() {
        List<String> trades = new ArrayList<>();
        for (Entity entity : nearbyEntities()) {
            if (entity instanceof net.minecraft.world.item.trading.Merchant merchant) {
                for (net.minecraft.world.item.trading.MerchantOffer offer : merchant.getOffers()) {
                    if (offer.isOutOfStock()) continue;
                    ItemStack costA = offer.getCostA();
                    ItemStack costB = offer.getCostB();
                    ItemStack playerA = new ItemStack(costA.getItem(), player.getInventory().countItem(costA.getItem()));
                    ItemStack playerB = costB.isEmpty() ? ItemStack.EMPTY : new ItemStack(costB.getItem(), player.getInventory().countItem(costB.getItem()));
                    if (offer.satisfiedBy(playerA, playerB)) {
                        String id = BuiltInRegistries.ITEM.getKey(offer.getResult().getItem()).toString();
                        if (!trades.contains(id)) trades.add(id);
                    }
                }
            }
        }
        return trades;
    }

    private void tradeWithVillager(String requestedItem) {
        Entity traderEntity = nearbyEntities().stream()
                .filter(e -> e instanceof net.minecraft.world.item.trading.Merchant)
                .findFirst().orElse(null);
        if (!(traderEntity instanceof net.minecraft.world.item.trading.Merchant merchant)) {
            throw new IllegalArgumentException("No villager or wandering trader nearby");
        }
        if (player.distanceToSqr(traderEntity) > 16) {
            approach(traderEntity.blockPosition(), 2);
            recoveryMethod = "Approaching trader at " + traderEntity.blockPosition().toShortString();
            return;
        }
        for (net.minecraft.world.item.trading.MerchantOffer offer : merchant.getOffers()) {
            if (offer.isOutOfStock()) continue;
            String resultId = BuiltInRegistries.ITEM.getKey(offer.getResult().getItem()).toString();
            if (!requestedItem.isEmpty() && !resultId.equals(requestedItem)) continue;
            ItemStack costA = offer.getCostA();
            ItemStack costB = offer.getCostB();
            ItemStack playerA = new ItemStack(costA.getItem(), player.getInventory().countItem(costA.getItem()));
            ItemStack playerB = costB.isEmpty() ? ItemStack.EMPTY : new ItemStack(costB.getItem(), player.getInventory().countItem(costB.getItem()));
            if (!offer.satisfiedBy(playerA, playerB)) continue;
            removeItemsFromInventory(costA.getItem(), costA.getCount());
            if (!costB.isEmpty()) removeItemsFromInventory(costB.getItem(), costB.getCount());
            offer.increaseUses();
            merchant.notifyTrade(offer);
            ItemStack output = offer.assemble();
            player.getInventory().placeItemBackInInventory(output, net.minecraft.util.Prediction.SERVER_ONLY);
            player.inventoryMenu.broadcastChanges();
            finish(true, "Traded with villager for " + output.getCount() + " " + resultId);
            return;
        }
        throw new IllegalArgumentException("No affordable matching villager trade for " + (requestedItem.isEmpty() ? "current inventory" : requestedItem));
    }

    private void removeItemsFromInventory(net.minecraft.world.item.Item item, int countNeeded) {
        int remaining = countNeeded;
        for (int i = 0; i < 36 && remaining > 0; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (stack.is(item)) {
                int take = Math.min(remaining, stack.getCount());
                player.getInventory().removeItem(i, take);
                remaining -= take;
            }
        }
    }

    private void useBoat() {
        if (player.getVehicle() != null) {
            pack().setForward(1.0f);
            return;
        }
        for (Entity entity : player.level().getEntities(player, player.getBoundingBox().inflate(6))) {
            String type = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString();
            if (type.contains("boat") || type.contains("raft")) {
                if (player.startRiding(entity)) {
                    pack().setForward(1.0f);
                    return;
                }
            }
        }
        int boatSlot = -1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (!stack.isEmpty() && stack.is(net.minecraft.tags.ItemTags.BOATS)) {
                boatSlot = i;
                break;
            }
        }
        if (boatSlot < 0) throw new IllegalArgumentException("No boat entity nearby and no boat in inventory");
        BlockPos waterPos = BlockPos.betweenClosedStream(
                player.blockPosition().offset(-3, -2, -3),
                player.blockPosition().offset(3, 1, 3))
                .filter(p -> player.level().getBlockState(p).is(Blocks.WATER) && player.level().getBlockState(p.above()).isAir())
                .map(BlockPos::immutable)
                .findFirst().orElse(null);
        if (waterPos == null) throw new IllegalArgumentException("No open water surface within reach to place boat");
        select(boatSlot);
        pack().lookAt(Vec3.atCenterOf(waterPos));
        player.getMainHandItem().use(player.level(), player, InteractionHand.MAIN_HAND);
        finish(true, "Used boat at water " + waterPos.toShortString());
    }

    private Direction horizontalDirectionToward(BlockPos heading) {
        if (heading != null) {
            int dx = heading.getX() - player.blockPosition().getX();
            int dz = heading.getZ() - player.blockPosition().getZ();
            if (Math.abs(dx) >= Math.abs(dz) && dx != 0) {
                return dx > 0 ? Direction.EAST : Direction.WEST;
            } else if (dz != 0) {
                return dz > 0 ? Direction.SOUTH : Direction.NORTH;
            }
        }
        return player.getDirection();
    }

    private void tunnelForward(BlockPos heading) {
        Direction dir = horizontalDirectionToward(heading);
        BlockPos feetNow = player.blockPosition();
        // A heading below the player digs a staircase: clear head, feet and the step below, then walk down.
        boolean descend = heading != null && heading.getY() < feetNow.getY();
        if (descend) {
            BlockPos ahead = feetNow.relative(dir);
            for (BlockPos candidate : List.of(ahead.above(), ahead, ahead.below())) {
                if (digCandidate(candidate)) return;
            }
            if (navigator.go(ahead.below())) {
                walking = true;
                action = "goto";
                recoveryMethod = "Stepping down the dug staircase";
                return;
            }
            throw new IllegalArgumentException("Staircase step below is not safe to stand on (cave, water or lava); choose another heading");
        }
        for (int step = 1; step <= 2; step++) {
            BlockPos feet = feetNow.relative(dir, step);
            BlockPos head = feet.above();
            for (BlockPos candidate : List.of(head, feet)) {
                if (digCandidate(candidate)) return;
            }
        }
        BlockPos stepAhead = feetNow.relative(dir);
        if (navigator.go(stepAhead)) {
            walking = true;
            action = "goto";
            recoveryMethod = "Stepping forward into dug tunnel";
            return;
        }
        throw new IllegalArgumentException("No diggable 1x2 tunnel block ahead in direction " + dir.getName());
    }

    private boolean digCandidate(BlockPos candidate) {
        var state = player.level().getBlockState(candidate);
        if (state.isAir() || !state.getFluidState().isEmpty() || state.getDestroySpeed(player.level(), candidate) < 0) return false;
        if (state.requiresCorrectToolForDrops() && !hasTool(candidate)) return false;
        target = candidate;
        targetName = BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString();
        action = "mine";
        return true;
    }

    private void bridgeForward(BlockPos heading) {
        int slot = findBuildingBlockSlot("");
        if (slot < 0) throw new IllegalArgumentException("No solid building block in inventory for bridging");
        String blockName = BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(slot).getItem()).toString();
        Direction dir = horizontalDirectionToward(heading);
        BlockPos aheadBelow = player.blockPosition().relative(dir).below();
        BlockPos aheadFeet = player.blockPosition().relative(dir);
        for (BlockPos candidate : List.of(aheadBelow, aheadFeet)) {
            if (placementHit(candidate) != null) {
                place(blockName, candidate);
                return;
            }
        }
        List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(player.blockPosition())).toList();
        if (!sites.isEmpty()) {
            place(blockName, sites.getFirst());
            return;
        }
        throw new IllegalArgumentException("No reachable support face ahead to bridge across");
    }

    private void buildShelter() {
        int slot = findBuildingBlockSlot("");
        if (slot < 0) throw new IllegalArgumentException("No building blocks in inventory to build shelter");
        String blockName = BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(slot).getItem()).toString();
        BlockPos center = player.blockPosition();
        List<BlockPos> wallSites = new ArrayList<>();
        BlockPos.betweenClosedStream(center.offset(-2, 0, -2), center.offset(2, 2, 2))
                .filter(p -> {
                    int dx = Math.abs(p.getX() - center.getX());
                    int dz = Math.abs(p.getZ() - center.getZ());
                    int dy = p.getY() - center.getY();
                    return ((dx == 2 || dz == 2) && dy <= 1) || (dx <= 1 && dz <= 1 && dy == 2);
                })
                .filter(p -> placementHit(p) != null)
                .forEach(p -> wallSites.add(p.immutable()));
        if (!wallSites.isEmpty()) {
            place(blockName, wallSites.getFirst());
            return;
        }
        List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(center)).toList();
        if (!sites.isEmpty()) {
            place(blockName, sites.getFirst());
            return;
        }
        throw new IllegalArgumentException("No reachable placement site for shelter wall/roof");
    }

    private void usePortal() {
        BlockPos portalPos = nearbyBlocks().stream()
                .filter(p -> player.level().getBlockState(p).is(Blocks.NETHER_PORTAL))
                .findFirst().orElse(null);
        if (portalPos != null) {
            walkTo(portalPos);
            recoveryMethod = "Entering Nether portal at " + portalPos.toShortString();
            action = "goto";
            return;
        }
        BlockPos obsidianPos = BlockPos.betweenClosedStream(
                player.blockPosition().offset(-3, -2, -3),
                player.blockPosition().offset(3, 2, 3))
                .filter(p -> player.level().getBlockState(p).is(Blocks.OBSIDIAN) && player.level().getBlockState(p.above()).isAir())
                .map(BlockPos::immutable)
                .findFirst().orElse(null);
        if (obsidianPos != null) {
            for (int i = 0; i < 36; i++) {
                ItemStack stack = player.getInventory().getItem(i);
                if (stack.is(net.minecraft.world.item.Items.FLINT_AND_STEEL) || stack.is(net.minecraft.world.item.Items.FIRE_CHARGE)) {
                    select(i);
                    BlockHitResult ray = new BlockHitResult(Vec3.atCenterOf(obsidianPos).add(0, 0.5, 0), Direction.UP, obsidianPos, false);
                    pack().lookAt(ray.getLocation());
                    player.getMainHandItem().useOn(new UseOnContext(player, InteractionHand.MAIN_HAND, ray));
                    finish(true, "Lit obsidian portal frame at " + obsidianPos.above().toShortString());
                    return;
                }
            }
        }
        if (player.getInventory().countItem(net.minecraft.world.item.Items.OBSIDIAN) > 0) {
            List<BlockPos> sites = placementSites().stream().filter(p -> !p.equals(player.blockPosition())).toList();
            if (!sites.isEmpty()) {
                place("minecraft:obsidian", sites.getFirst());
                return;
            }
        }
        throw new IllegalArgumentException("No active Nether portal or lightable obsidian frame nearby");
    }

    void tick() {
        if (!busy()) return;
        try {
            if (++ticks > 2400) throw new IllegalArgumentException("Action timed out after 120 seconds");
            if (action.equals("pickup") && player.getInventory().countItem(pickupItem) > pickupBefore) {
                finish(true, "Collected item into inventory");
                return;
            }
            if (walking) {
                if (!navigator.tick()) return;
                walking = false;
                if (action.equals("goto") || action.equals("explore") || action.equals("escape")
                        || action.equals("run") || action.equals("sprint")
                        || action.equals("smelt") || action.equals("chest")
                        || action.equals("sleep") || action.equals("trade")) {
                    if (action.equals("escape")) {
                        finish(true, recoveryMethod + "; landed");
                        return;
                    }
                    if (action.equals("sleep")) {
                        sleepInBed();
                        return;
                    }
                    finish(true, (recoveryMethod.isEmpty() ? "" : recoveryMethod + "; ")
                            + "Reached " + player.blockPosition().toShortString());
                    return;
                }
            }
            switch (action) {
                case "sleep" -> {
                    if (player.isSleeping()) {
                        sleeping = true;
                    } else if (sleeping) {
                        finish(true, "Finished sleeping in bed");
                    } else if (ticks > 40 && !walking) {
                        finish(false, "Could not sleep in bed");
                    }
                }
                case "jump" -> {
                    if (ticks >= 8 && (player.onGround() || ticks >= 20)) {
                        pack().stopMovement();
                        finish(true, "Jumped" + (player.onGround() ? " and landed" : ""));
                    }
                }
                case "run", "sprint" -> {
                    if (!walking) {
                        if (ticks >= 30 || player.horizontalCollision) {
                            pack().setSprinting(false).setForward(0f);
                            finish(true, "Finished sprint burst");
                        }
                    }
                }
                case "boat" -> {
                    if (ticks >= 40) {
                        pack().setForward(0f);
                        finish(true, "Rowed boat forward");
                    }
                }
                case "sneak", "crouch" -> {
                    if (ticks >= 20) {
                        pack().setSneaking(false);
                        finish(true, "Sneaked");
                    }
                }
                case "mine", "escape" -> {
                    if (pillaring) {
                        handlePillaringTick();
                        break;
                    }
                    var targetState = player.level().getBlockState(target);
                    if (targetBlock == null && !targetState.isAir()) targetBlock = targetState.getBlock();
                    // A broken block may be replaced by water, falling gravel or a plant rather than air.
                    boolean blockGone = targetState.isAir() || (targetBlock != null && !targetState.is(targetBlock));
                    if (escapeBlockBroken || blockGone) {
                        if (action.equals("escape")) {
                            if (!escapeBlockBroken) {
                                escapeBlockBroken = true;
                                pack().stopAll();
                            }
                            // Wait for the fall and health update before recording this experiment.
                            if (++recoveryTicks >= 10 && player.onGround()) {
                                finish(true, recoveryMethod + "; settled after block break");
                            } else if (recoveryTicks >= 100) {
                                finish(false, recoveryMethod + "; did not settle within five seconds");
                            }
                        } else {
                            finish(true, "Mined block at " + target.toShortString() + "; drops may need pickup");
                        }
                    } else if (!visible(player, target)) {
                        Vec3 eye = player.getEyePosition();
                        Vec3 center = Vec3.atCenterOf(target);
                        BlockHitResult ray = eye.distanceToSqr(center) <= 20.25
                                ? player.level().clip(new ClipContext(eye, center,
                                        ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player))
                                : null;
                        if (ray != null && ray.getType() == HitResult.Type.BLOCK
                                && player.level().getBlockState(ray.getBlockPos()).is(BlockTags.LEAVES)) {
                            selectTool(ray.getBlockPos());
                            pack().lookAt(center);
                            if (!attackingBlock) {
                                pack().start(EntityPlayerActionPack.ActionType.ATTACK,
                                        EntityPlayerActionPack.Action.continuous());
                                attackingBlock = true;
                            }
                        } else if (action.equals("mine") && miningRepositions++ < 3) {
                            pack().stopAll();
                            attackingBlock = false;
                            approach(target, 3, true);
                        } else {
                            throw new IllegalArgumentException("Block is out of reach or obscured after repositioning");
                        }
                    } else {
                        selectTool(target);
                        pack().lookAt(Vec3.atCenterOf(target));
                        if (!attackingBlock) {
                            pack().start(EntityPlayerActionPack.ActionType.ATTACK,
                                    EntityPlayerActionPack.Action.continuous());
                            attackingBlock = true;
                        }
                    }
                }
                case "eat" -> {
                    if (!player.isUsingItem()) {
                        finish(player.getFoodData().getFoodLevel() > foodBefore, "Food level " + player.getFoodData().getFoodLevel());
                    }
                }
                case "attack" -> {
                    Entity entity = player.level().getEntity(entityId);
                    if (entity == null || !entity.isAlive()) {
                        finish(true, "Target defeated or removed");
                    } else if (entity instanceof Player) {
                        throw new IllegalArgumentException("Attacking players is disabled");
                    } else if (player.distanceToSqr(entity) > 9) {
                        if (ticks % 20 == 0) approach(entity.blockPosition(), 1);
                    } else if (player.hasLineOfSight(entity) && player.getAttackStrengthScale(0) >= 0.95) {
                        pack().lookAt(entity.getEyePosition());
                        player.attack(entity);
                        player.swing(InteractionHand.MAIN_HAND, net.minecraft.world.item.component.SwingAnimation.DEFAULT, true);
                    }
                }
                case "pickup" -> {
                    if (ticks > 160) {
                        finish(false, "Could not collect item drop within 8 seconds");
                        return;
                    }
                    Entity entity = player.level().getEntity(entityId);
                    if (entity == null) finish(false, "Drop disappeared without inventory gain");
                    else if (entity instanceof ItemEntity item && !canHold(item.getItem())) {
                        finish(false, "Inventory full; cannot pick up item");
                    } else {
                        pack().lookAt(entity.position().add(0, 0.2, 0));
                        if (ticks % 20 == 0 && !followDrop(entity)) {
                            finish(false, "Item moved beyond reachable pickup positions");
                        }
                    }
                }
                case "place", "pillar" -> {
                    if (pillaring) {
                        handlePillaringTick();
                    }
                }
                case "wait" -> {
                    var lastSound = AcousticSensor.latestSound();
                    if (lastSound != null && player.level().getGameTime() - lastSound.gameTime() < 40) {
                        pack().lookAt(new Vec3(lastSound.x(), lastSound.y(), lastSound.z()));
                    }
                    if (ticks >= 60) finish(true, "Rested briefly");
                }
                default -> { }
            }
        } catch (IllegalArgumentException e) {
            finish(false, e.getMessage());
        }
    }

    private void handlePillaringTick() {
        if (!pillarPlaced) {
            if (player.getY() >= target.getY() + 0.95
                    && !player.getBoundingBox().intersects(new net.minecraft.world.phys.AABB(target))) {
                BlockPos support = target.below();
                BlockHitResult ray = new BlockHitResult(
                        Vec3.atCenterOf(support).add(0, 0.5, 0),
                        Direction.UP,
                        support,
                        false);
                pack().look(player.getYRot(), 90.0f);
                player.getMainHandItem().useOn(new UseOnContext(player, InteractionHand.MAIN_HAND, ray));
                if (!player.level().getBlockState(target).canBeReplaced()) {
                    pillarPlaced = true;
                }
            } else if (ticks % 5 == 0 && player.onGround() && ticks < 20) {
                pack().stopMovement();
                pack().look(player.getYRot(), 90.0f);
                pack().start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
            } else if (ticks > 35) {
                pillaring = false;
                throw new IllegalArgumentException("Failed to jump and place block under feet within time limit");
            }
        } else {
            if (player.onGround() || player.getY() >= target.getY() + 1.0 || ticks > 45) {
                pillaring = false;
                if (action.equals("escape")) {
                    finish(true, recoveryMethod + "; elevated onto " + targetName);
                } else {
                    finish(true, "Placed " + targetName + " under feet to get higher at " + target.toShortString());
                }
            }
        }
    }

    boolean reflex20Hz() {
        return liveReflex.tick20Hz(player, this);
    }

    boolean isCombatOrFleeActive() {
        return liveReflex.isCombatOrFleeActive();
    }

    boolean reflex() {
        return reflex20Hz();
    }

    private void finish(boolean success, String detail) {
        if (!success && detail != null && (detail.contains("timed out") || detail.contains("stuck") || detail.contains("No safe"))) {
            pathfindingTimeouts++;
        }
        if (success && "mine".equals(action)) {
            blocksMinedCount++;
        }
        finish(success ? "success" : "failed", detail);
    }

    private void finish(String status, String detail) {
        pack().stopAll();
        pillaring = false;
        pillarPlaced = false;
        result = new JsonObject();
        result.addProperty("id", id);
        result.addProperty("status", status);
        result.addProperty("detail", detail);
        result.addProperty("action", action);
        result.addProperty("target", targetName);
        result.addProperty("duration_ticks", ticks);
        if (target != null) result.addProperty("block_position", target.toShortString());
        result.addProperty("health_before", healthBefore);
        result.addProperty("health_after", player.getHealth());
        result.addProperty("damage", Math.max(0, healthBefore - player.getHealth()));
        result.addProperty("position", player.blockPosition().toShortString());
        progress.record(action, status, player.blockPosition());
        LocalAiMod.LOG.info("Willow action {} {}: {}", action, status, detail);
        action = "";
    }

    static boolean visible(ServerPlayer player, BlockPos pos) {
        return visibleFrom(player, player.getEyePosition(), pos);
    }

    private static boolean visibleFrom(ServerPlayer player, Vec3 eye, BlockPos pos) {
        Vec3 center = Vec3.atCenterOf(pos);
        if (eye.distanceToSqr(center) > 20.25) return false;
        BlockHitResult ray = player.level().clip(new ClipContext(eye, center,
                ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
        return ray.getType() == HitResult.Type.BLOCK && ray.getBlockPos().equals(pos);
    }

    private static boolean isNonResourceGroundFoliage(String id) {
        return id.equals("minecraft:leaf_litter") || id.equals("minecraft:wildflowers")
                || id.equals("minecraft:pink_petals") || id.equals("minecraft:bush")
                || id.equals("minecraft:firefly_bush") || id.equals("minecraft:short_dry_grass")
                || id.equals("minecraft:tall_dry_grass") || id.equals("minecraft:short_grass")
                || id.equals("minecraft:tall_grass") || id.equals("minecraft:fern")
                || id.equals("minecraft:large_fern") || id.equals("minecraft:dead_bush")
                || id.equals("minecraft:seagrass") || id.equals("minecraft:tall_seagrass");
    }

    /** Whether some safe standing cell within arm's reach of the block exists (no path search, no raycast). */
    private boolean reachable(BlockPos block) {
        BlockPos feet = player.blockPosition();
        if (player.getEyePosition().distanceToSqr(Vec3.atCenterOf(block)) <= 20.25 && !block.equals(feet.below())) return true;
        for (int dy = -2; dy <= 1; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                for (int dz = -2; dz <= 2; dz++) {
                    BlockPos stand = block.offset(dx, dy, dz);
                    if (stand.equals(block) || stand.above().equals(block) || stand.below().equals(block)) continue;
                    if (Navigator.safe(player.level(), stand)) return true;
                }
            }
        }
        return false;
    }

    private List<BlockPos> cachedNearbyBlocks = null;
    private JsonObject cachedNearbyWorkstations = null;

    private void scanNearbyBlocksAndWorkstations() {
        List<BlockPos> blocks = new ArrayList<>();
        JsonObject stations = new JsonObject();
        BlockPos center = player.blockPosition();
        int ox = center.getX();
        int oy = center.getY();
        int oz = center.getZ();
        double bestTable = Double.MAX_VALUE;
        double bestFurnace = Double.MAX_VALUE;
        double bestChest = Double.MAX_VALUE;
        double bestBed = Double.MAX_VALUE;
        double bestPortal = Double.MAX_VALUE;

        int minX = -16, maxX = 16;
        int minZ = -16, maxZ = 16;
        int minY = -5, maxY = 6;

        for (BlockPos pos : BlockPos.betweenClosed(center.offset(minX, minY, minZ), center.offset(maxX, maxY, maxZ))) {
            if (!player.level().isLoaded(pos)) continue;
            var state = player.level().getBlockState(pos);
            if (state.isAir()) continue;

            int dx = pos.getX() - ox;
            int dy = pos.getY() - oy;
            int dz = pos.getZ() - oz;

            // Check workstations (-16..16, -4..4, -16..16)
            if (dy >= -4 && dy <= 4) {
                double dSq = dx * dx + dy * dy + dz * dz;
                if (state.is(Blocks.CRAFTING_TABLE) && dSq < bestTable) {
                    bestTable = dSq;
                    stations.add("crafting_table", posJson(pos));
                } else if ((state.is(Blocks.FURNACE) || state.is(Blocks.BLAST_FURNACE) || state.is(Blocks.SMOKER)) && dSq < bestFurnace) {
                    bestFurnace = dSq;
                    stations.add("furnace", posJson(pos));
                } else if ((state.is(Blocks.CHEST) || state.is(Blocks.TRAPPED_CHEST) || state.is(Blocks.BARREL)) && dSq < bestChest) {
                    bestChest = dSq;
                    stations.add("chest", posJson(pos));
                } else if (state.is(BlockTags.BEDS) && dSq < bestBed) {
                    bestBed = dSq;
                    stations.add("bed", posJson(pos));
                } else if (state.is(Blocks.NETHER_PORTAL) && dSq < bestPortal) {
                    bestPortal = dSq;
                    stations.add("nether_portal", posJson(pos));
                }
            }

            // Check nearby blocks (-12..12, -5..6, -12..12)
            if (dx >= -12 && dx <= 12 && dz >= -12 && dz <= 12 && dy >= -5 && dy <= 6) {
                if (state.getFluidState().isEmpty()) {
                    if (!(state.is(BlockTags.LEAVES) && pos.distManhattan(center) > 2)) {
                        String id = BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString();
                        if (!isNonResourceGroundFoliage(id) || pos.equals(center) || pos.equals(center.above())) {
                            boolean exposed = false;
                            for (Direction direction : Direction.values()) {
                                var adj = player.level().getBlockState(pos.relative(direction));
                                if (adj.isAir() || adj.is(BlockTags.LEAVES) || adj.canBeReplaced()) {
                                    exposed = true;
                                    break;
                                }
                            }
                            if (exposed) blocks.add(pos.immutable());
                        }
                    }
                }
            }
        }
        blocks.sort(Comparator.comparingDouble(p -> p.distSqr(center)));
        cachedNearbyBlocks = blocks;
        cachedNearbyWorkstations = stations;
    }

    List<BlockPos> nearbyBlocks() {
        if (cachedNearbyBlocks == null) {
            scanNearbyBlocksAndWorkstations();
        }
        return cachedNearbyBlocks;
    }

    List<Entity> nearbyEntities() {
        var entities = new ArrayList<>(player.level().getEntities(player, player.getBoundingBox().inflate(12),
                e -> e instanceof LivingEntity || e instanceof ItemEntity));
        entities.sort(Comparator.comparingDouble(player::distanceToSqr));
        return entities;
    }

    JsonObject observe(List<String> chat) {
        cachedNearbyBlocks = null;
        cachedNearbyWorkstations = null;
        scanNearbyBlocksAndWorkstations();
        JsonObject observation = new JsonObject();
        JsonArray pos = new JsonArray();
        pos.add(player.blockPosition().getX());
        pos.add(player.blockPosition().getY());
        pos.add(player.blockPosition().getZ());
        observation.add("position", pos);
        observation.addProperty("dimension", player.level().dimension().identifier().toString());
        observation.addProperty("world_tick", player.level().getGameTime());
        observation.addProperty("health", player.getHealth());
        observation.addProperty("max_health", player.getMaxHealth());
        observation.addProperty("absorption", player.getAbsorptionAmount());
        observation.addProperty("food", player.getFoodData().getFoodLevel());
        observation.addProperty("saturation", player.getFoodData().getSaturationLevel());
        observation.addProperty("armor_points", player.getArmorValue());
        observation.addProperty("is_alive", player.isAlive());
        observation.addProperty("fire_ticks", player.getRemainingFireTicks());
        observation.addProperty("fall_distance", player.fallDistance);
        observation.addProperty("in_lava", player.isInLava());
        observation.addProperty("suffocating", player.isInWall());
        observation.addProperty("on_ground", player.onGround());
        observation.addProperty("in_water", player.isInWater());
        observation.addProperty("air", player.getAirSupply());
        observation.addProperty("time", player.level().getGameTime() % 24000);
        observation.addProperty("weather", player.level().isThundering() ? "thunder"
                : player.level().isRaining() ? "rain" : "clear");
        player.level().getBiome(player.blockPosition()).unwrapKey().ifPresent(
                key -> observation.addProperty("biome", key.identifier().toString()));
        JsonObject diagnosis = new JsonObject();
        diagnosis.addProperty("stuck", !progress.problem().isEmpty());
        diagnosis.addProperty("reason", progress.problem());
        diagnosis.addProperty("recovery_attempts", progress.recoveryAttempts());
        diagnosis.addProperty("risk_policy", "May take fall damage or die; try descent or breaking natural support");
        observation.add("diagnosis", diagnosis);
        JsonObject terrain = new JsonObject();
        terrain.addProperty("feet", BuiltInRegistries.BLOCK.getKey(
                player.level().getBlockState(player.blockPosition()).getBlock()).toString());
        terrain.addProperty("below", BuiltInRegistries.BLOCK.getKey(
                player.level().getBlockState(player.blockPosition().below()).getBlock()).toString());
        JsonArray exits = new JsonArray();
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dz == 0) continue;
                BlockPos cell = player.blockPosition().offset(dx, 0, dz);
                JsonObject exit = new JsonObject();
                exit.addProperty("offset", dx + ",0," + dz);
                exit.addProperty("feet", BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(cell).getBlock()).toString());
                exit.addProperty("head", BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(cell.above()).getBlock()).toString());
                exit.addProperty("standable", Navigator.safe(player.level(), cell));
                JsonArray at = new JsonArray();
                at.add(cell.getX());
                at.add(cell.getY());
                at.add(cell.getZ());
                exit.add("at", at);
                exits.add(exit);
            }
        }
        terrain.add("exits", exits);
        observation.add("terrain", terrain);
        Map<String, Integer> inventory = new TreeMap<>();
        var blockItems = new java.util.TreeSet<String>();
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (!stack.isEmpty()) inventory.merge(BuiltInRegistries.ITEM.getKey(stack.getItem()).toString(), stack.getCount(), Integer::sum);
            if (!stack.isEmpty() && stack.getItem() instanceof BlockItem) {
                blockItems.add(BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
            }
        }
        ItemStack offhand = player.getOffhandItem();
        if (!offhand.isEmpty()) {
            inventory.merge(BuiltInRegistries.ITEM.getKey(offhand.getItem()).toString(), offhand.getCount(), Integer::sum);
            if (offhand.getItem() instanceof BlockItem) {
                blockItems.add(BuiltInRegistries.ITEM.getKey(offhand.getItem()).toString());
            }
        }
        for (net.minecraft.world.entity.EquipmentSlot slot : new net.minecraft.world.entity.EquipmentSlot[]{
                net.minecraft.world.entity.EquipmentSlot.HEAD, net.minecraft.world.entity.EquipmentSlot.CHEST,
                net.minecraft.world.entity.EquipmentSlot.LEGS, net.minecraft.world.entity.EquipmentSlot.FEET}) {
            ItemStack armor = player.getItemBySlot(slot);
            if (!armor.isEmpty()) {
                inventory.merge(BuiltInRegistries.ITEM.getKey(armor.getItem()).toString(), armor.getCount(), Integer::sum);
            }
        }
        JsonArray inv = new JsonArray();
        inventory.forEach((name, count) -> inv.add(name + "=" + count));
        observation.add("inventory", inv);
        int freeSlots = 0;
        for (int i = 0; i < 36; i++) {
            if (player.getInventory().getItem(i).isEmpty()) freeSlots++;
        }
        observation.addProperty("free_slots", freeSlots);
        JsonArray placeable = new JsonArray();
        blockItems.forEach(placeable::add);
        observation.add("placeable", placeable);
        JsonArray blocks = new JsonArray();
        Map<String, Integer> seen = new TreeMap<>();
        List<BlockPos> nearby = nearbyBlocks();
        nearby.sort(Comparator.comparingInt(p -> {
            var state = player.level().getBlockState(p);
            if (state.is(BlockTags.LOGS)) return 0;
            if (state.is(Blocks.CRAFTING_TABLE)) return 1;
            if (state.is(Blocks.COAL_ORE) || state.is(Blocks.IRON_ORE) || state.is(Blocks.STONE)) return 2;
            return 3;
        }));
        boolean reachableStone = false;
        for (BlockPos block : nearby) {
            var blockState = player.level().getBlockState(block);
            String name = BuiltInRegistries.BLOCK.getKey(blockState.getBlock()).toString();
            if (seen.merge(name, 1, Integer::sum) > 3) continue;
            boolean canReach = reachable(block);
            if (canReach && (blockState.is(Blocks.STONE) || blockState.is(Blocks.COBBLESTONE) || blockState.is(Blocks.DEEPSLATE))
                    && (!blockState.requiresCorrectToolForDrops() || hasTool(block))) {
                reachableStone = true;
            }
            JsonObject resource = new JsonObject();
            resource.addProperty("id", name);
            resource.addProperty("visible", visible(player, block));
            resource.addProperty("reachable", canReach);
            resource.addProperty("requires_tool", blockState.requiresCorrectToolForDrops());
            resource.addProperty("has_tool", !blockState.requiresCorrectToolForDrops() || hasTool(block));
            JsonArray location = new JsonArray();
            location.add(block.getX());
            location.add(block.getY());
            location.add(block.getZ());
            resource.add("at", location);
            blocks.add(resource);
            if (blocks.size() >= 36) break;
        }
        observation.add("nearby_blocks", blocks);
        JsonArray entities = new JsonArray();
        for (Entity entity : nearbyEntities().stream().limit(16).toList()) {
            String type = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString();
            JsonObject observed = new JsonObject();
            observed.addProperty("id", entity.getId());
            observed.addProperty("type", type);
            observed.addProperty("distance", Math.round(player.distanceTo(entity) * 10) / 10.0);
            observed.addProperty("hostile", entity instanceof Monster);
            if (entity instanceof ItemEntity item) {
                observed.addProperty("item", BuiltInRegistries.ITEM.getKey(item.getItem().getItem()).toString());
                observed.addProperty("count", item.getItem().getCount());
            }
            JsonArray location = new JsonArray();
            location.add(entity.blockPosition().getX());
            location.add(entity.blockPosition().getY());
            location.add(entity.blockPosition().getZ());
            observed.add("at", location);
            entities.add(observed);
        }
        observation.add("entities", entities);
        JsonArray recipes = new JsonArray();
        List<Crafting.Plan> plans = Crafting.available(player);
        plans.forEach(p -> recipes.add(p.holder().id().identifier().toString()));
        observation.add("craftable", recipes);
        JsonArray smeltable = new JsonArray();
        Crafting.smeltable(player).forEach(smeltable::add);
        observation.add("smeltable", smeltable);
        JsonArray trades = new JsonArray();
        availableTrades().forEach(trades::add);
        observation.add("trades", trades);
        JsonObject crafting = Crafting.awareness(player, plans, placementSites());
        if (!crafting.has("next_step") && !reachableStone && player.onGround() && !player.isInWater()
                && crafting.get("pickaxes").getAsInt() > 0 && crafting.get("stone_pickaxes").getAsInt() == 0
                && player.getInventory().countItem(net.minecraft.world.item.Items.IRON_PICKAXE) == 0
                && crafting.get("cobblestone").getAsInt() < 3) {
            // Plains/lakes rarely expose stone; a staircase a few blocks down always reaches it.
            BlockPos feet = player.blockPosition();
            JsonObject step = new JsonObject();
            step.addProperty("action", "tunnel");
            step.addProperty("target", "");
            step.addProperty("x", feet.getX());
            step.addProperty("y", feet.getY() - 4);
            step.addProperty("z", feet.getZ());
            crafting.add("next_step", step);
            crafting.addProperty("missing", "No reachable stone in sight; dig a staircase down with tunnel (heading below you) until stone is exposed, then mine 3 cobblestone.");
        }
        observation.add("crafting", crafting);
        observation.add("last_result", result());
        JsonArray remembered = new JsonArray();
        for (BlockPos watchedPos : watched) {
            if (!player.level().isLoaded(watchedPos)) continue;
            JsonObject checked = new JsonObject();
            JsonArray at = new JsonArray();
            at.add(watchedPos.getX());
            at.add(watchedPos.getY());
            at.add(watchedPos.getZ());
            checked.add("at", at);
            checked.addProperty("id", BuiltInRegistries.BLOCK.getKey(player.level().getBlockState(watchedPos).getBlock()).toString());
            remembered.add(checked);
        }
        observation.add("remembered_blocks", remembered);
        JsonArray messages = new JsonArray();
        chat.forEach(messages::add);
        observation.add("chat", messages);
        observation.add("doing", doing());
        observation.add("sight", sightPerception.observe(player));
        observation.add("hearing", AcousticSensor.summary(player.level().getGameTime()));
        observation.add("equipment", captureEquipment());
        observation.add("unlocked_advancements", captureUnlockedAdvancements());
        observation.add("nearby_workstations", captureNearbyWorkstations());
        observation.add("execution_telemetry", captureAndResetExecutionTelemetry());
        return observation;
    }

    private JsonObject captureEquipment() {
        JsonObject eq = new JsonObject();
        record SlotEntry(String label, net.minecraft.world.entity.EquipmentSlot slot) {}
        for (SlotEntry se : new SlotEntry[]{
                new SlotEntry("MainHand", net.minecraft.world.entity.EquipmentSlot.MAINHAND),
                new SlotEntry("OffHand", net.minecraft.world.entity.EquipmentSlot.OFFHAND),
                new SlotEntry("Head", net.minecraft.world.entity.EquipmentSlot.HEAD),
                new SlotEntry("Chest", net.minecraft.world.entity.EquipmentSlot.CHEST),
                new SlotEntry("Legs", net.minecraft.world.entity.EquipmentSlot.LEGS),
                new SlotEntry("Feet", net.minecraft.world.entity.EquipmentSlot.FEET)}) {
            ItemStack stack = player.getItemBySlot(se.slot());
            JsonObject itemObj = new JsonObject();
            if (stack.isEmpty()) {
                itemObj.addProperty("item_id", "minecraft:air");
                itemObj.addProperty("current_durability", 0);
                itemObj.addProperty("max_durability", 0);
            } else {
                String id = BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
                int maxDur = stack.getMaxDamage();
                int curDur = maxDur > 0 ? Math.max(0, maxDur - stack.getDamageValue()) : 0;
                itemObj.addProperty("item_id", id);
                itemObj.addProperty("current_durability", curDur);
                itemObj.addProperty("max_durability", maxDur);
            }
            eq.add(se.label(), itemObj);
        }
        ItemStack mh = player.getMainHandItem();
        if (mh.isEmpty() && !trackedMainHandTool.isEmpty()) {
            boolean stillOwned = false;
            for (int i = 0; i < 36; i++) {
                ItemStack s = player.getInventory().getItem(i);
                if (!s.isEmpty() && BuiltInRegistries.ITEM.getKey(s.getItem()).toString().equals(trackedMainHandTool)) {
                    stillOwned = true;
                    break;
                }
            }
            if (!stillOwned && toolsBroken.size() < 8) {
                toolsBroken.add(trackedMainHandTool);
            }
            trackedMainHandTool = "";
        } else if (!mh.isEmpty() && mh.getMaxDamage() > 0) {
            trackedMainHandTool = BuiltInRegistries.ITEM.getKey(mh.getItem()).toString();
        }
        return eq;
    }

    private JsonArray captureUnlockedAdvancements() {
        JsonArray unlocked = new JsonArray();
        var server = player.level().getServer();
        if (server == null) return unlocked;
        var playerAdv = player.getAdvancements();
        for (var holder : server.getAdvancements().getAllAdvancements()) {
            String id = holder.id().toString();
            if (id.contains(":recipes/")) continue;
            if (playerAdv.getOrStartProgress(holder).isDone()) {
                unlocked.add(id);
            }
        }
        return unlocked;
    }

    private JsonObject captureNearbyWorkstations() {
        if (cachedNearbyWorkstations == null) {
            scanNearbyBlocksAndWorkstations();
        }
        return cachedNearbyWorkstations;
    }

    private static JsonArray posJson(BlockPos p) {
        JsonArray arr = new JsonArray();
        arr.add(p.getX());
        arr.add(p.getY());
        arr.add(p.getZ());
        return arr;
    }

    private JsonObject captureAndResetExecutionTelemetry() {
        JsonObject exec = new JsonObject();
        int durTicks = result.has("duration_ticks") ? result.get("duration_ticks").getAsInt() : ticks;
        exec.addProperty("duration_ms", durTicks * 50L);
        exec.addProperty("compilation_failed", false);
        exec.add("runtime_exceptions", new JsonArray());
        exec.addProperty("pathfinding_timeouts", pathfindingTimeouts);
        exec.addProperty("blocks_mined_count", blocksMinedCount);
        exec.addProperty("improper_tool_mine_attempts", improperToolMineAttempts);
        JsonArray broken = new JsonArray();
        toolsBroken.forEach(broken::add);
        exec.add("tools_broken", broken);

        pathfindingTimeouts = 0;
        blocksMinedCount = 0;
        improperToolMineAttempts = 0;
        toolsBroken.clear();
        return exec;
    }
}
