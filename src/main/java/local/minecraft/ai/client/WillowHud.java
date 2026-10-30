package local.minecraft.ai.client;

import java.util.List;
import local.minecraft.ai.WillowView;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.item.ItemStack;

/** Vanilla-styled live copy of Willow's HUD and inventory window while the camera follows Willow. */
final class WillowHud {
    private static final Identifier INVENTORY = Identifier.withDefaultNamespace("textures/gui/container/inventory.png");
    private static final Identifier HOTBAR = sprite("hud/hotbar");
    private static final Identifier SELECTION = sprite("hud/hotbar_selection");
    private static final Identifier OFFHAND = sprite("hud/hotbar_offhand_left");
    private static final Identifier XP_BACKGROUND = sprite("hud/experience_bar_background");
    private static final Identifier XP_PROGRESS = sprite("hud/experience_bar_progress");
    private static final Identifier HEART_CONTAINER = sprite("hud/heart/container");
    private static final Identifier HEART_FULL = sprite("hud/heart/full");
    private static final Identifier HEART_HALF = sprite("hud/heart/half");
    private static final Identifier ABSORB_FULL = sprite("hud/heart/absorbing_full");
    private static final Identifier ABSORB_HALF = sprite("hud/heart/absorbing_half");
    private static final Identifier FOOD_EMPTY = sprite("hud/food_empty");
    private static final Identifier FOOD_FULL = sprite("hud/food_full");
    private static final Identifier FOOD_HALF = sprite("hud/food_half");
    private static final Identifier ARMOR_EMPTY = sprite("hud/armor_empty");
    private static final Identifier ARMOR_FULL = sprite("hud/armor_full");
    private static final Identifier ARMOR_HALF = sprite("hud/armor_half");
    private static final Identifier AIR = sprite("hud/air");
    private static final Identifier AIR_EMPTY = sprite("hud/air_empty");
    private static final Identifier[] EMPTY_ARMOR = {sprite("container/slot/helmet"), sprite("container/slot/chestplate"),
            sprite("container/slot/leggings"), sprite("container/slot/boots")};
    private static final Identifier EMPTY_OFFHAND = sprite("container/slot/shield");

    static boolean inventoryVisible = true;
    private static long zoomShownUntil;

    private WillowHud() {}

    private static Identifier sprite(String path) { return Identifier.withDefaultNamespace(path); }

    static void showZoom() { zoomShownUntil = System.currentTimeMillis() + 1500; }

    static void extract(GuiGraphicsExtractor g, DeltaTracker delta) {
        Minecraft client = Minecraft.getInstance();
        WillowView.Snapshot view = WillowView.current();
        if (view == null || !CinematicCamera.follows(client.getCameraEntity())) return;
        Font font = client.font;
        int width = g.guiWidth();
        int height = g.guiHeight();
        int center = width / 2;
        hotbar(g, font, view, center, height);
        int bottom = height - 39;
        int rows = statusBars(g, view, center, bottom);
        experience(g, font, view, center, height);
        sensoryOverlay(g, font, view);
        if (System.currentTimeMillis() < zoomShownUntil) {
            g.centeredText(font, Component.literal(String.format("Camera distance %.1f blocks",
                    CinematicCamera.zoomDistance())), center, bottom - rows - 14, 0xFFFFFFFF);
        }
        if (inventoryVisible && client.getCameraEntity() instanceof LivingEntity willow) {
            inventory(g, font, view, willow, width, height);
        }
    }

    private static void hotbar(GuiGraphicsExtractor g, Font font, WillowView.Snapshot view, int center, int height) {
        int left = center - 91;
        g.blitSprite(RenderPipelines.GUI_TEXTURED, HOTBAR, left, height - 22, 182, 22);
        g.blitSprite(RenderPipelines.GUI_TEXTURED, SELECTION, left - 1 + view.selected() * 20, height - 23, 24, 23);
        if (!view.offhand().isEmpty()) {
            g.blitSprite(RenderPipelines.GUI_TEXTURED, OFFHAND, left - 29, height - 23, 29, 24);
            slot(g, font, view.offhand(), left - 26, height - 19);
        }
        for (int i = 0; i < 9; i++) slot(g, font, view.main().get(i), center - 90 + i * 20 + 2, height - 19);
    }

    /** Returns the pixel height used above the hotbar so other text can avoid overlapping it. */
    private static int statusBars(GuiGraphicsExtractor g, WillowView.Snapshot view, int center, int bottom) {
        int left = center - 91;
        int right = center + 91;
        float max = Math.max(view.maxHealth(), 1);
        int healthHalves = (int) Math.ceil(view.health());
        int absorbHalves = (int) Math.ceil(view.absorption());
        int containers = (int) Math.ceil(max / 2);
        int hearts = containers + (absorbHalves + 1) / 2;
        int heartRows = (int) Math.ceil(hearts / 10.0);
        int rowHeight = Math.max(10 - (heartRows - 2), 3);
        for (int i = hearts - 1; i >= 0; i--) {
            int x = left + (i % 10) * 8;
            int y = bottom - (i / 10) * rowHeight;
            g.blitSprite(RenderPipelines.GUI_TEXTURED, HEART_CONTAINER, x, y, 9, 9);
            if (i < containers) {
                int half = i * 2;
                if (half + 1 < healthHalves) g.blitSprite(RenderPipelines.GUI_TEXTURED, HEART_FULL, x, y, 9, 9);
                else if (half + 1 == healthHalves) g.blitSprite(RenderPipelines.GUI_TEXTURED, HEART_HALF, x, y, 9, 9);
            } else {
                int half = (i - containers) * 2;
                g.blitSprite(RenderPipelines.GUI_TEXTURED, half + 1 < absorbHalves ? ABSORB_FULL : ABSORB_HALF, x, y, 9, 9);
            }
        }
        int used = (heartRows - 1) * rowHeight + 10;
        if (view.armorValue() > 0) {
            int y = bottom - used;
            for (int i = 0; i < 10; i++) {
                Identifier icon = i * 2 + 1 < view.armorValue() ? ARMOR_FULL
                        : i * 2 + 1 == view.armorValue() ? ARMOR_HALF : ARMOR_EMPTY;
                g.blitSprite(RenderPipelines.GUI_TEXTURED, icon, left + i * 8, y, 9, 9);
            }
            used += 10;
        }
        for (int i = 0; i < 10; i++) {
            int x = right - i * 8 - 9;
            g.blitSprite(RenderPipelines.GUI_TEXTURED, FOOD_EMPTY, x, bottom, 9, 9);
            if (i * 2 + 1 < view.food()) g.blitSprite(RenderPipelines.GUI_TEXTURED, FOOD_FULL, x, bottom, 9, 9);
            else if (i * 2 + 1 == view.food()) g.blitSprite(RenderPipelines.GUI_TEXTURED, FOOD_HALF, x, bottom, 9, 9);
        }
        if (view.air() < view.maxAir()) {
            int full = (int) Math.ceil(Math.max(0, view.air()) * 10.0 / Math.max(1, view.maxAir()));
            for (int i = 0; i < 10; i++) {
                g.blitSprite(RenderPipelines.GUI_TEXTURED, i < full ? AIR : AIR_EMPTY, right - i * 8 - 9, bottom - 10, 9, 9);
            }
            used = Math.max(used, 20);
        }
        return used;
    }

    private static void experience(GuiGraphicsExtractor g, Font font, WillowView.Snapshot view, int center, int height) {
        int left = center - 91;
        g.blitSprite(RenderPipelines.GUI_TEXTURED, XP_BACKGROUND, left, height - 29, 182, 5);
        int progress = (int) (view.xpProgress() * 183);
        if (progress > 0) {
            g.blitSprite(RenderPipelines.GUI_TEXTURED, XP_PROGRESS, 182, 5, 0, 0, left, height - 29, progress, 5);
        }
        if (view.xpLevel() > 0) {
            String level = Integer.toString(view.xpLevel());
            int x = center - font.width(level) / 2;
            int y = height - 35;
            g.text(font, level, x + 1, y, 0xFF000000, false);
            g.text(font, level, x - 1, y, 0xFF000000, false);
            g.text(font, level, x, y + 1, 0xFF000000, false);
            g.text(font, level, x, y - 1, 0xFF000000, false);
            g.text(font, level, x, y, 0xFF80FF20, false);
        }
    }

    private static void inventory(GuiGraphicsExtractor g, Font font, WillowView.Snapshot view, LivingEntity willow,
                                  int width, int height) {
        int x = Math.max(0, width - 176 - 4);
        int y = Math.max(0, Math.min(4, height - 166));
        g.blit(RenderPipelines.GUI_TEXTURED, INVENTORY, x, y, 0, 0, 176, 166, 256, 256);
        g.text(font, Component.translatable("container.crafting"), x + 97, y + 8, 0xFF404040, false);
        InventoryScreen.extractEntityInInventoryFollowsMouse(g, x + 26, y + 8, x + 75, y + 78, 30, 0.0625f,
                x + 51, y + 30, willow);
        List<ItemStack> armor = view.armor();
        for (int i = 0; i < 4; i++) {
            if (armor.get(i).isEmpty()) {
                g.blitSprite(RenderPipelines.GUI_TEXTURED, EMPTY_ARMOR[i], x + 8, y + 8 + i * 18, 16, 16);
            } else {
                slot(g, font, armor.get(i), x + 8, y + 8 + i * 18);
            }
        }
        if (view.offhand().isEmpty()) {
            g.blitSprite(RenderPipelines.GUI_TEXTURED, EMPTY_OFFHAND, x + 77, y + 62, 16, 16);
        } else {
            slot(g, font, view.offhand(), x + 77, y + 62);
        }
        for (int i = 9; i < 36; i++) {
            slot(g, font, view.main().get(i), x + 8 + (i - 9) % 9 * 18, y + 84 + (i - 9) / 9 * 18);
        }
        for (int i = 0; i < 9; i++) {
            int sx = x + 8 + i * 18;
            if (i == view.selected()) g.fill(sx, y + 142, sx + 16, y + 158, 0x60FFFFFF);
            slot(g, font, view.main().get(i), sx, y + 142);
        }
    }

    private static void sensoryOverlay(GuiGraphicsExtractor g, Font font, WillowView.Snapshot view) {
        int x = 6;
        int y = 6;
        int w = 270;
        int h = 48;
        g.fill(x, y, x + w, y + h, 0x90000000);
        g.fill(x, y, x + 2, y + h, 0xFF55FFFF);
        g.text(font, Component.literal("§b👁 Vision: §f" + (view.lookingAt() != null ? view.lookingAt() : "Air")), x + 6, y + 4, 0xFFFFFFFF, false);
        g.text(font, Component.literal("§7   FOV: " + (view.visibleSummary() != null ? view.visibleSummary() : "Clear")), x + 6, y + 15, 0xFFAAAAAA, false);
        g.text(font, Component.literal("§6👂 Hearing: §f" + (view.hearingSummary() != null ? view.hearingSummary() : "Quiet")), x + 6, y + 26, 0xFFFFFFFF, false);
        if (view.activeAlert() != null && !"None".equals(view.activeAlert())) {
            g.text(font, Component.literal("§c⚠ Alert: " + view.activeAlert()), x + 6, y + 37, 0xFFFF5555, false);
        } else {
            g.text(font, Component.literal("§a✔ Senses: Online (Multimodal Sensors)"), x + 6, y + 37, 0xFF55FF55, false);
        }
    }

    private static void slot(GuiGraphicsExtractor g, Font font, ItemStack stack, int x, int y) {
        if (stack.isEmpty()) return;
        g.item(stack, x, y);
        g.itemDecorations(font, stack, x, y);
    }
}
