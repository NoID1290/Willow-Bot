package local.minecraft.ai;

import java.util.ArrayList;
import java.util.List;
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

    private WillowView() {}

    public static Snapshot current() { return current; }

    static void clear() { current = null; }

    /** Must run on the server thread; stacks are copied so the client never reads live server objects. */
    static void capture(ServerPlayer player) {
        if (player == null || player.isRemoved()) {
            current = null;
            return;
        }
        var inventory = player.getInventory();
        List<ItemStack> main = new ArrayList<>(36);
        for (int i = 0; i < 36; i++) main.add(inventory.getItem(i).copy());
        List<ItemStack> armor = new ArrayList<>(4);
        for (EquipmentSlot slot : ARMOR) armor.add(player.getItemBySlot(slot).copy());
        current = new Snapshot(List.copyOf(main), List.copyOf(armor), player.getItemBySlot(EquipmentSlot.OFFHAND).copy(),
                inventory.getSelectedSlot(), player.getHealth(), player.getMaxHealth(), player.getAbsorptionAmount(),
                player.getFoodData().getFoodLevel(), player.getArmorValue(), player.getAirSupply(),
                player.getMaxAirSupply(), player.experienceLevel, player.experienceProgress,
                SightPerception.currentLookingAt(), SightPerception.currentVisibleSummary(),
                AcousticSensor.latestSoundText(), AcousticSensor.latestAlertText());
    }
}
