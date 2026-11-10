package local.minecraft.ai;

import carpet.fakes.ServerPlayerInterface;
import carpet.helpers.EntityPlayerActionPack;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.Vec3;

/**
 * Zero-allocation mutable state updated in-place on every 20Hz (50ms) server tick.
 * Preempts the Slow Cognitive Loop and pauses new LLM inference requests during active combat/fleeing.
 */
final class LiveReflexState {
    private final BlockPos.MutableBlockPos scanCursor = new BlockPos.MutableBlockPos();

    float health = 20.0f;
    int foodLevel = 20;
    int oxygen = 300;
    int fireTicks = 0;
    double fallDistance = 0.0;
    boolean inLava = false;
    boolean lavaWithin2Blocks = false;
    boolean suffocating = false;

    int nearestHostileId = -1;
    double nearestHostileDistSqr = 9999.0;
    boolean creeperEmergency = false;
    boolean skeletonLineOfSight = false;
    int zombieSwarmCount = 0;

    boolean combatOrFleeActive = false;
    String activeHazard = "NONE";
    private int shieldHoldTicks = 0;

    boolean tick20Hz(ServerPlayer player, Actions actions) {
        if (player == null || !player.isAlive() || player.isRemoved()) {
            combatOrFleeActive = false;
            activeHazard = "NONE";
            return false;
        }

        // Update primitive fields in-place (zero heap allocations)
        this.health = player.getHealth();
        this.foodLevel = player.getFoodData().getFoodLevel();
        this.oxygen = player.getAirSupply();
        this.fireTicks = player.getRemainingFireTicks();
        this.fallDistance = player.fallDistance;
        this.inLava = player.isInLava();
        this.suffocating = player.isInWall();

        // Scan 2-block neighborhood for lava using pre-allocated MutableBlockPos
        boolean nearLava = this.inLava;
        if (!nearLava) {
            int px = player.blockPosition().getX();
            int py = player.blockPosition().getY();
            int pz = player.blockPosition().getZ();
            outer:
            for (int dx = -2; dx <= 2; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -2; dz <= 2; dz++) {
                        scanCursor.set(px + dx, py + dy, pz + dz);
                        if (player.level().isLoaded(scanCursor)
                                && player.level().getBlockState(scanCursor).is(Blocks.LAVA)) {
                            nearLava = true;
                            break outer;
                        }
                    }
                }
            }
        }
        this.lavaWithin2Blocks = nearLava;

        // Evaluate hostiles within 6.0 blocks (distSqr < 36.0) + Creeper acoustic fuse
        this.nearestHostileId = -1;
        this.nearestHostileDistSqr = 9999.0;
        this.creeperEmergency = false;
        this.skeletonLineOfSight = false;
        this.zombieSwarmCount = 0;

        var latestSound = AcousticSensor.latestSound();
        if (latestSound != null
                && player.level().getGameTime() - latestSound.gameTime() <= 30
                && latestSound.soundId().contains("creeper.primed")
                && latestSound.distance() < 6.5) {
            this.creeperEmergency = true;
        }

        Entity closeTarget = null;
        for (Entity entity : player.level().getEntities(player, player.getBoundingBox().inflate(6.0),
                e -> e instanceof Monster && e.isAlive())) {
            double dSqr = player.distanceToSqr(entity);
            String path = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).getPath();
            if (dSqr < this.nearestHostileDistSqr) {
                this.nearestHostileDistSqr = dSqr;
                this.nearestHostileId = entity.getId();
                closeTarget = entity;
            }
            if (path.contains("creeper") && dSqr < 36.0) {
                this.creeperEmergency = true;
            } else if ((path.contains("skeleton") || path.contains("stray") || path.contains("bogged"))
                    && player.hasLineOfSight(entity)) {
                this.skeletonLineOfSight = true;
            } else if (path.contains("zombie") || path.contains("husk") || path.contains("drowned")) {
                this.zombieSwarmCount++;
            }
        }

        EntityPlayerActionPack pack = ((ServerPlayerInterface) player).getActionPack();
        if (shieldHoldTicks > 0) {
            shieldHoldTicks--;
            if (shieldHoldTicks == 0 && player.isUsingItem() && player.getUseItem().is(Items.SHIELD)) {
                player.stopUsingItem();
            }
        }

        // 1. Water-bucket MLG (>3m fall) or fire/lava extinguish
        if ((this.fallDistance > 3.0 && !player.onGround() && !player.isInWater())
                || this.inLava || this.fireTicks > 0) {
            int waterSlot = findItemSlot(player, Items.WATER_BUCKET);
            if (waterSlot >= 0) {
                actions.stop("Reflex: water bucket MLG / fire extinguish");
                actions.selectHotbarOrSwap(waterSlot);
                pack.look(player.getYRot(), 90.0f);
                player.getMainHandItem().use(player.level(), player, InteractionHand.MAIN_HAND);
                return markInterrupt("WATER_MLG_OR_FIRE");
            }
            if (this.inLava || this.lavaWithin2Blocks) {
                pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
                pack.setSprinting(true).setForward(1.0f);
                return markInterrupt("LAVA_ESCAPE");
            }
        }

        // 2. Suffocation or critical oxygen (< 60)
        if (this.suffocating) {
            pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
            return markInterrupt("SUFFOCATION_ESCAPE");
        }
        if (this.oxygen < 60 && player.isInWater()) {
            actions.stop("Reflex: surfacing for oxygen");
            pack.look(player.getYRot(), -75.0f);
            pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
            pack.setForward(1.0f);
            return markInterrupt("LOW_OXYGEN_SURFACE");
        }

        // 3. Creeper fuse / close Creeper: auto-equip shield & block, or sprint away
        if (this.creeperEmergency) {
            actions.stop("Reflex: evading/blocking Creeper");
            if (ensureShieldReady(player, actions)) {
                if (closeTarget != null) {
                    pack.lookAt(closeTarget.getEyePosition());
                }
                InteractionHand hand = player.getOffhandItem().is(Items.SHIELD)
                        ? InteractionHand.OFF_HAND : InteractionHand.MAIN_HAND;
                player.getItemInHand(hand).use(player.level(), player, hand);
                shieldHoldTicks = 15;
            } else if (closeTarget != null) {
                Vec3 away = player.position().subtract(closeTarget.position());
                float fleeYaw = (float) Math.toDegrees(Math.atan2(-away.x, away.z));
                pack.look(fleeYaw, 0.0f).setSprinting(this.foodLevel > 6).setForward(1.0f);
            }
            return markInterrupt("CREEPER_EVADE");
        }

        // 4. Close hostile (< 6.0m): block Skeleton arrow or counter-attack / retreat on critical HP
        if (closeTarget != null && this.nearestHostileDistSqr < 36.0) {
            if (this.skeletonLineOfSight && ensureShieldReady(player, actions) && !player.isUsingItem()) {
                pack.lookAt(closeTarget.getEyePosition());
                InteractionHand hand = player.getOffhandItem().is(Items.SHIELD)
                        ? InteractionHand.OFF_HAND : InteractionHand.MAIN_HAND;
                player.getItemInHand(hand).use(player.level(), player, hand);
                shieldHoldTicks = 10;
                return markInterrupt("SKELETON_SHIELD_BLOCK");
            }
            if (this.health <= 8.0f && this.zombieSwarmCount >= 2) {
                actions.stop("Reflex: retreating from Zombie swarm at critical HP");
                Vec3 away = player.position().subtract(closeTarget.position());
                float fleeYaw = (float) Math.toDegrees(Math.atan2(-away.x, away.z));
                pack.look(fleeYaw, 0.0f).setSprinting(this.foodLevel > 6).setForward(1.0f);
                return markInterrupt("SWARM_FLEE");
            }
            if (this.nearestHostileDistSqr < 9.0 && player.hasLineOfSight(closeTarget)
                    && player.getAttackStrengthScale(0) >= 0.95f) {
                actions.equipBestWeapon();
                pack.lookAt(closeTarget.getEyePosition());
                player.attack(closeTarget);
                player.swing(InteractionHand.MAIN_HAND, net.minecraft.world.item.component.SwingAnimation.DEFAULT, true);
                this.combatOrFleeActive = true;
                this.activeHazard = "MELEE_DEFENSE";
                return false;
            }
        }

        // 5. Critical vitals (Health <= 8.0 or Food <= 6, plus safe proactive hunger < 14)
        if (!player.isUsingItem()) {
            boolean criticalVitals = (this.health <= 8.0f && player.getFoodData().needsFood()) || this.foodLevel <= 6;
            boolean proactiveEat = this.foodLevel < 14 && closeTarget == null;
            if ((criticalVitals || proactiveEat) && actions.selectHighestSaturationFood()) {
                actions.stop("Interrupted to consume highest-saturation food");
                player.getMainHandItem().use(player.level(), player, InteractionHand.MAIN_HAND);
                if (player.isUsingItem()) {
                    return markInterrupt(criticalVitals ? "CRITICAL_VITALS_EAT" : "PROACTIVE_EAT");
                }
            }
        }

        if (player.isInWater()) {
            pack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }

        if (this.combatOrFleeActive) {
            this.combatOrFleeActive = false;
            this.activeHazard = "NONE";
            LocalAiMod.LOG.info("[REFLEX-INTERRUPT] Hazard cleared; resuming cognitive planning.");
        }
        return false;
    }

    private boolean markInterrupt(String hazard) {
        if (!this.combatOrFleeActive || !hazard.equals(this.activeHazard)) {
            LocalAiMod.LOG.warn("[REFLEX-INTERRUPT] Immediate 20Hz survival reflex triggered: {} (HP={}, Food={})",
                    hazard, this.health, this.foodLevel);
        }
        this.combatOrFleeActive = true;
        this.activeHazard = hazard;
        return true;
    }

    private static int findItemSlot(ServerPlayer player, net.minecraft.world.item.Item target) {
        for (int i = 0; i < 36; i++) {
            if (player.getInventory().getItem(i).is(target)) return i;
        }
        return -1;
    }

    private static boolean ensureShieldReady(ServerPlayer player, Actions actions) {
        if (player.getOffhandItem().is(Items.SHIELD) || player.getMainHandItem().is(Items.SHIELD)) {
            return true;
        }
        int slot = findItemSlot(player, Items.SHIELD);
        if (slot < 0) return false;
        ItemStack shield = player.getInventory().removeItem(slot, 1);
        ItemStack oldOff = player.getOffhandItem();
        player.setItemSlot(EquipmentSlot.OFFHAND, shield);
        if (!oldOff.isEmpty()) {
            player.getInventory().setItem(slot, oldOff);
        }
        player.inventoryMenu.broadcastChanges();
        return true;
    }

    boolean isCombatOrFleeActive() {
        return combatOrFleeActive;
    }
}
