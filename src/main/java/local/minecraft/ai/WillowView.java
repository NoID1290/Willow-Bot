package local.minecraft.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.ItemStack;

/** Copied server-side state of Willow for the singleplayer host's live inventory HUD. */
public final class WillowView {
    private static final EquipmentSlot[] ARMOR = {EquipmentSlot.HEAD, EquipmentSlot.CHEST,
            EquipmentSlot.LEGS, EquipmentSlot.FEET};

    public record Snapshot(List<ItemStack> main, List<ItemStack> armor, ItemStack offhand, int selected,
                           float health, float maxHealth, float absorption, int food, int armorValue,
                           int air, int maxAir, int xpLevel, float xpProgress,
                           String lookingAt, String visibleSummary, String hearingSummary, String activeAlert) {}

    private static volatile Snapshot current;
    private static int lastInvHash = 0;

    private WillowView() {}

    public static Snapshot current() { return current; }

    static void clear() {
        current = null;
        lastInvHash = 0;
    }

    private static int computeFullInventoryHash(ServerPlayer player) {
        int hash = 1;
        var inv = player.getInventory();
        for (int i = 0; i < 36; i++) {
            ItemStack s = inv.getItem(i);
            if (!s.isEmpty()) {
                hash = 31 * hash + s.getItem().hashCode();
                hash = 31 * hash + s.getCount();
                hash = 31 * hash + s.getDamageValue();
            } else {
                hash = 31 * hash;
            }
        }
        for (EquipmentSlot slot : ARMOR) {
            ItemStack s = player.getItemBySlot(slot);
            if (!s.isEmpty()) {
                hash = 31 * hash + s.getItem().hashCode();
                hash = 31 * hash + s.getCount();
                hash = 31 * hash + s.getDamageValue();
            } else {
                hash = 31 * hash;
            }
        }
        ItemStack off = player.getItemBySlot(EquipmentSlot.OFFHAND);
        if (!off.isEmpty()) {
            hash = 31 * hash + off.getItem().hashCode();
            hash = 31 * hash + off.getCount();
            hash = 31 * hash + off.getDamageValue();
        }
        return hash;
    }

    /** Must run on the server thread; stacks are copied so the client never reads live server objects. */
    static void capture(ServerPlayer player) {
        if (player == null || player.isRemoved()) {
            current = null;
            lastInvHash = 0;
            return;
        }
        var inventory = player.getInventory();
        int invHash = computeFullInventoryHash(player);
        int selected = inventory.getSelectedSlot();
        float health = player.getHealth();
        float maxHealth = player.getMaxHealth();
        float absorption = player.getAbsorptionAmount();
        int food = player.getFoodData().getFoodLevel();
        int armorVal = player.getArmorValue();
        int air = player.getAirSupply();
        int maxAir = player.getMaxAirSupply();
        int xpLevel = player.experienceLevel;
        float xpProgress = player.experienceProgress;
        String lookingAt = SightPerception.currentLookingAt();
        String visibleSummary = SightPerception.currentVisibleSummary();
        String hearingSummary = AcousticSensor.latestSoundText();
        String activeAlert = AcousticSensor.latestAlertText();

        Snapshot prev = current;
        if (prev != null
                && invHash == lastInvHash
                && selected == prev.selected()
                && health == prev.health()
                && maxHealth == prev.maxHealth()
                && absorption == prev.absorption()
                && food == prev.food()
                && armorVal == prev.armorValue()
                && air == prev.air()
                && maxAir == prev.maxAir()
                && xpLevel == prev.xpLevel()
                && xpProgress == prev.xpProgress()
                && Objects.equals(lookingAt, prev.lookingAt())
                && Objects.equals(visibleSummary, prev.visibleSummary())
                && Objects.equals(hearingSummary, prev.hearingSummary())
                && Objects.equals(activeAlert, prev.activeAlert())) {
            return;
        }

        List<ItemStack> main;
        List<ItemStack> armor;
        ItemStack offhand;
        if (prev != null && invHash == lastInvHash && selected == prev.selected()) {
            main = prev.main();
            armor = prev.armor();
            offhand = prev.offhand();
        } else {
            main = new ArrayList<>(36);
            for (int i = 0; i < 36; i++) main.add(inventory.getItem(i).copy());
            armor = new ArrayList<>(4);
            for (EquipmentSlot slot : ARMOR) armor.add(player.getItemBySlot(slot).copy());
            offhand = player.getItemBySlot(EquipmentSlot.OFFHAND).copy();
            main = List.copyOf(main);
            armor = List.copyOf(armor);
            lastInvHash = invHash;
        }

        current = new Snapshot(main, armor, offhand,
                selected, health, maxHealth, absorption,
                food, armorVal, air, maxAir, xpLevel, xpProgress,
                lookingAt, visibleSummary, hearingSummary, activeAlert);
    }
}
