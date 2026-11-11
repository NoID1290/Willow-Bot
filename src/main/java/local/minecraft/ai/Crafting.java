package local.minecraft.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import net.minecraft.core.BlockPos;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.crafting.CraftingInput;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.ShapedRecipe;
import net.minecraft.world.item.crafting.ShapelessRecipe;
import net.minecraft.world.level.block.Blocks;

final class Crafting {
    record Plan(RecipeHolder<?> holder, CraftingRecipe recipe, CraftingInput input, int[] slots) {}

    static boolean tableReachable(ServerPlayer player) {
        return BlockPos.betweenClosedStream(player.blockPosition().offset(-3, -2, -3),
                player.blockPosition().offset(3, 2, 3))
                .anyMatch(pos -> player.level().getBlockState(pos).is(Blocks.CRAFTING_TABLE)
                        && Actions.visible(player, pos));
    }

    static boolean furnaceReachable(ServerPlayer player) {
        return BlockPos.betweenClosedStream(player.blockPosition().offset(-4, -2, -4),
                player.blockPosition().offset(4, 2, 4))
                .anyMatch(pos -> {
                    var state = player.level().getBlockState(pos);
                    return (state.is(Blocks.FURNACE) || state.is(Blocks.BLAST_FURNACE) || state.is(Blocks.SMOKER))
                            && Actions.visible(player, pos);
                });
    }

    static boolean chestReachable(ServerPlayer player) {
        return BlockPos.betweenClosedStream(player.blockPosition().offset(-4, -2, -4),
                player.blockPosition().offset(4, 2, 4))
                .anyMatch(pos -> {
                    var state = player.level().getBlockState(pos);
                    return (state.is(Blocks.CHEST) || state.is(Blocks.TRAPPED_CHEST) || state.is(Blocks.BARREL))
                            && Actions.visible(player, pos);
                });
    }

    static boolean bedReachable(ServerPlayer player) {
        return BlockPos.betweenClosedStream(player.blockPosition().offset(-4, -2, -4),
                player.blockPosition().offset(4, 2, 4))
                .anyMatch(pos -> player.level().getBlockState(pos).is(net.minecraft.tags.BlockTags.BEDS)
                        && Actions.visible(player, pos));
    }

    static boolean isFuel(ItemStack stack) {
        if (stack.isEmpty()) return false;
        return stack.is(Items.COAL) || stack.is(Items.CHARCOAL) || stack.is(Items.COAL_BLOCK)
                || stack.is(Items.LAVA_BUCKET) || stack.is(Items.BLAZE_ROD) || stack.is(Items.STICK)
                || stack.is(ItemTags.LOGS_THAT_BURN) || stack.is(ItemTags.PLANKS);
    }

    static int fuelCapacity(ItemStack fuel) {
        if (fuel.isEmpty()) return 0;
        if (fuel.is(Items.LAVA_BUCKET)) return 100;
        if (fuel.is(Items.COAL_BLOCK)) return 80;
        if (fuel.is(Items.BLAZE_ROD)) return 12;
        if (fuel.is(Items.COAL) || fuel.is(Items.CHARCOAL)) return 8;
        if (fuel.is(ItemTags.LOGS_THAT_BURN) || fuel.is(ItemTags.PLANKS)) return 1;
        if (fuel.is(Items.STICK)) return 1;
        return 1;
    }

    static ItemStack smeltResult(ItemStack input) {
        if (input.isEmpty()) return ItemStack.EMPTY;
        if (input.is(Items.RAW_IRON) || input.is(Items.IRON_ORE) || input.is(Items.DEEPSLATE_IRON_ORE))
            return new ItemStack(Items.IRON_INGOT);
        if (input.is(Items.RAW_COPPER) || input.is(Items.COPPER_ORE) || input.is(Items.DEEPSLATE_COPPER_ORE))
            return new ItemStack(Items.COPPER_INGOT);
        if (input.is(Items.RAW_GOLD) || input.is(Items.GOLD_ORE) || input.is(Items.DEEPSLATE_GOLD_ORE))
            return new ItemStack(Items.GOLD_INGOT);
        if (input.is(Items.BEEF)) return new ItemStack(Items.COOKED_BEEF);
        if (input.is(Items.PORKCHOP)) return new ItemStack(Items.COOKED_PORKCHOP);
        if (input.is(Items.MUTTON)) return new ItemStack(Items.COOKED_MUTTON);
        if (input.is(Items.CHICKEN)) return new ItemStack(Items.COOKED_CHICKEN);
        if (input.is(Items.RABBIT)) return new ItemStack(Items.COOKED_RABBIT);
        if (input.is(Items.COD)) return new ItemStack(Items.COOKED_COD);
        if (input.is(Items.SALMON)) return new ItemStack(Items.COOKED_SALMON);
        if (input.is(Items.POTATO)) return new ItemStack(Items.BAKED_POTATO);
        if (input.is(Items.KELP)) return new ItemStack(Items.DRIED_KELP);
        if (input.is(Items.SAND) || input.is(Items.RED_SAND)) return new ItemStack(Items.GLASS);
        if (input.is(Items.COBBLESTONE)) return new ItemStack(Items.STONE);
        if (input.is(Items.CLAY_BALL)) return new ItemStack(Items.BRICK);
        if (input.is(ItemTags.LOGS_THAT_BURN)) return new ItemStack(Items.CHARCOAL);
        return ItemStack.EMPTY;
    }

    static List<String> smeltable(ServerPlayer player) {
        boolean furnace = furnaceReachable(player)
                || player.getInventory().countItem(Items.FURNACE) > 0
                || player.getInventory().countItem(Items.BLAST_FURNACE) > 0
                || player.getInventory().countItem(Items.SMOKER) > 0;
        if (!furnace) return List.of();
        boolean fuel = false;
        List<String> items = new ArrayList<>();
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = player.getInventory().getItem(slot);
            if (isFuel(stack)) fuel = true;
            if (!smeltResult(stack).isEmpty() && !stack.is(Items.COBBLESTONE)) {
                String id = net.minecraft.core.registries.BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
                if (!items.contains(id)) items.add(id);
            }
        }
        if (!fuel) return List.of();
        return items;
    }

    private static long cachedGameTime = -1;
    private static boolean cachedTableReachable = false;
    private static int cachedInventoryHash = 0;
    private static UUID cachedPlayerUuid = null;
    private static List<Plan> cachedPlans = null;

    public static synchronized void invalidateCache() {
        cachedGameTime = -1;
        cachedPlans = null;
    }

    private static int inventoryHash(ServerPlayer player) {
        int hash = 1;
        for (int i = 0; i < 36; i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (!stack.isEmpty()) {
                hash = 31 * hash + stack.getItem().hashCode();
                hash = 31 * hash + stack.getCount();
            }
        }
        ItemStack offhand = player.getOffhandItem();
        if (!offhand.isEmpty()) {
            hash = 31 * hash + offhand.getItem().hashCode();
            hash = 31 * hash + offhand.getCount();
        }
        return hash;
    }

    static synchronized List<Plan> available(ServerPlayer player) {
        if (player == null || player.level() == null || player.level().getServer() == null) {
            return List.of();
        }
        long gameTime = player.level().getGameTime();
        boolean table = tableReachable(player);
        int invHash = inventoryHash(player);
        UUID uuid = player.getUUID();

        if (cachedPlans != null
                && cachedGameTime == gameTime
                && cachedTableReachable == table
                && cachedInventoryHash == invHash
                && uuid.equals(cachedPlayerUuid)) {
            return cachedPlans;
        }

        List<Plan> plans = new ArrayList<>();
        for (RecipeHolder<?> holder : player.level().getServer().getRecipeManager().getRecipes()) {
            Plan plan = plan(player, holder, table);
            if (plan != null) plans.add(plan);
        }
        plans.sort(java.util.Comparator.comparing(p -> p.holder().id().identifier().toString()));
        cachedGameTime = gameTime;
        cachedTableReachable = table;
        cachedInventoryHash = invHash;
        cachedPlayerUuid = uuid;
        cachedPlans = plans;
        return plans;
    }

    static JsonObject awareness(ServerPlayer player, List<Plan> plans, List<BlockPos> placementSites) {
        var result = new JsonObject();
        boolean table = tableReachable(player);
        int logs = 0;
        int planks = 0;
        int pickaxes = 0;
        int cobblestone = 0;
        int stonePickaxes = 0;
        int woodenPickaxes = 0;
        int swords = 0;
        int axes = 0;
        int shovels = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = player.getInventory().getItem(slot);
            if (stack.is(ItemTags.LOGS)) logs += stack.getCount();
            if (stack.is(ItemTags.PLANKS)) planks += stack.getCount();
            if (stack.is(ItemTags.PICKAXES)) {
                pickaxes += stack.getCount();
                if (stack.is(Items.STONE_PICKAXE)) stonePickaxes += stack.getCount();
                else if (stack.is(Items.WOODEN_PICKAXE)) woodenPickaxes += stack.getCount();
            }
            if (stack.is(ItemTags.SWORDS)) swords += stack.getCount();
            if (stack.is(ItemTags.AXES)) axes += stack.getCount();
            if (stack.is(ItemTags.SHOVELS)) shovels += stack.getCount();
            if (stack.is(Items.COBBLESTONE)) cobblestone += stack.getCount();
        }
        ItemStack offhand = player.getOffhandItem();
        if (!offhand.isEmpty()) {
            if (offhand.is(ItemTags.LOGS)) logs += offhand.getCount();
            if (offhand.is(ItemTags.PLANKS)) planks += offhand.getCount();
            if (offhand.is(ItemTags.PICKAXES)) {
                pickaxes += offhand.getCount();
                if (offhand.is(Items.STONE_PICKAXE)) stonePickaxes += offhand.getCount();
                else if (offhand.is(Items.WOODEN_PICKAXE)) woodenPickaxes += offhand.getCount();
            }
            if (offhand.is(ItemTags.SWORDS)) swords += offhand.getCount();
            if (offhand.is(ItemTags.AXES)) axes += offhand.getCount();
            if (offhand.is(ItemTags.SHOVELS)) shovels += offhand.getCount();
            if (offhand.is(Items.COBBLESTONE)) cobblestone += offhand.getCount();
        }
        int tables = player.getInventory().countItem(Items.CRAFTING_TABLE) + (offhand.is(Items.CRAFTING_TABLE) ? offhand.getCount() : 0);
        int furnaces = player.getInventory().countItem(Items.FURNACE) + (offhand.is(Items.CRAFTING_TABLE) ? 0 : (offhand.is(Items.FURNACE) ? offhand.getCount() : 0));
        int chests = player.getInventory().countItem(Items.CHEST) + (offhand.is(Items.CHEST) ? offhand.getCount() : 0);
        int sticks = player.getInventory().countItem(Items.STICK) + (offhand.is(Items.STICK) ? offhand.getCount() : 0);
        int rawIron = player.getInventory().countItem(Items.RAW_IRON) + (offhand.is(Items.RAW_IRON) ? offhand.getCount() : 0);
        int ironIngots = player.getInventory().countItem(Items.IRON_INGOT) + (offhand.is(Items.IRON_INGOT) ? offhand.getCount() : 0);
        int ironPickaxes = player.getInventory().countItem(Items.IRON_PICKAXE) + (offhand.is(Items.IRON_PICKAXE) ? offhand.getCount() : 0);
        boolean furnaceNear = furnaceReachable(player);
        boolean chestNear = chestReachable(player);
        boolean bedNear = bedReachable(player);
        result.addProperty("grid", table ? "3x3" : "2x2");
        result.addProperty("table_reachable", table);
        result.addProperty("furnace_reachable", furnaceNear);
        result.addProperty("chest_reachable", chestNear);
        result.addProperty("bed_reachable", bedNear);
        result.addProperty("table_in_inventory", tables);
        result.addProperty("furnace_in_inventory", furnaces);
        result.addProperty("chest_in_inventory", chests);
        result.addProperty("logs", logs);
        result.addProperty("planks", planks);
        result.addProperty("sticks", sticks);
        result.addProperty("pickaxes", pickaxes);
        result.addProperty("stone_pickaxes", stonePickaxes);
        result.addProperty("wooden_pickaxes", woodenPickaxes);
        result.addProperty("cobblestone", cobblestone);
        result.addProperty("swords", swords);
        result.addProperty("axes", axes);
        result.addProperty("shovels", shovels);
        result.addProperty("knowledge", "A table is CRAFTED from four planks in your 2x2 inventory; "
                + "logs make four matching planks. No table is needed for planks, sticks, or a table. "
                + "PLACE an inventory table to enable 3x3 recipes. A wooden pickaxe needs three planks and two sticks. "
                + "To get higher (climb out of pits or reach high blocks), place a block under yourself at your current feet position. "
                + "Never craft duplicate tools that you already have in inventory.");
        var sites = new JsonArray();
        for (BlockPos site : placementSites) {
            var pos = new JsonArray();
            pos.add(site.getX());
            pos.add(site.getY());
            pos.add(site.getZ());
            sites.add(pos);
        }
        result.add("placement_sites", sites);
        BlockPos feet = player.blockPosition();
        if (placementSites.contains(feet)) {
            var pos = new JsonArray();
            pos.add(feet.getX());
            pos.add(feet.getY());
            pos.add(feet.getZ());
            result.add("under_feet_placement_site", pos);
        }
        if (pickaxes > 0) {
            if (stonePickaxes == 0 && ironPickaxes == 0 && cobblestone >= 3 && table) {
                Plan stonePickPlan = plans.stream()
                        .filter(p -> p.holder().id().identifier().toString().equals("minecraft:stone_pickaxe"))
                        .findFirst().orElse(null);
                if (stonePickPlan != null) {
                    result.add("next_step", step("craft", "minecraft:stone_pickaxe", BlockPos.ZERO));
                    return result;
                } else if (sticks < 2) {
                    Plan stickPlan = plans.stream()
                            .filter(p -> p.holder().id().identifier().toString().equals("minecraft:stick"))
                            .findFirst().orElse(null);
                    if (stickPlan != null) {
                        result.add("next_step", step("craft", "minecraft:stick", BlockPos.ZERO));
                        return result;
                    }
                }
            }
            if (ironPickaxes == 0 && ironIngots >= 3 && table) {
                Plan ironPickPlan = plans.stream()
                        .filter(p -> p.holder().id().identifier().toString().equals("minecraft:iron_pickaxe"))
                        .findFirst().orElse(null);
                if (ironPickPlan != null) {
                    result.add("next_step", step("craft", "minecraft:iron_pickaxe", BlockPos.ZERO));
                    return result;
                }
            }
            if (rawIron > 0 && !furnaceNear) {
                if (furnaces > 0 && !placementSites.isEmpty()) {
                    BlockPos groundSite = placementSites.stream().filter(p -> !p.equals(player.blockPosition())).findFirst().orElse(placementSites.getFirst());
                    result.add("next_step", step("place", "minecraft:furnace", groundSite));
                    return result;
                } else if (furnaces == 0 && cobblestone >= 8 && table) {
                    Plan furnacePlan = plans.stream()
                            .filter(p -> p.holder().id().identifier().toString().equals("minecraft:furnace"))
                            .findFirst().orElse(null);
                    if (furnacePlan != null) {
                        result.add("next_step", step("craft", "minecraft:furnace", BlockPos.ZERO));
                        return result;
                    }
                }
            }
            if (stonePickaxes == 0 && ironPickaxes == 0 && cobblestone < 3) {
                result.addProperty("missing", "You already have a wooden pickaxe! Stop gathering logs/wood; mine 3 cobblestone (mine exposed minecraft:stone or explore/tunnel to stone) to craft a stone_pickaxe.");
            }
            return result;
        }
        if (!table && tables > 0) {
            BlockPos groundSite = placementSites.stream().filter(p -> !p.equals(player.blockPosition())).findFirst().orElse(null);
            if (groundSite != null) {
                result.add("next_step", step("place", "minecraft:crafting_table", groundSite));
            } else if (!placementSites.isEmpty()) {
                result.add("next_step", step("place", "minecraft:crafting_table", placementSites.getFirst()));
            } else {
                result.addProperty("missing", "No reachable table placement site; move to open ground and try again");
            }
            return result;
        }
        String wanted = !table && tables == 0 ? "minecraft:crafting_table"
                : sticks < 2 ? "minecraft:stick" : "minecraft:wooden_pickaxe";
        Plan next = plans.stream().filter(p -> p.holder().id().identifier().toString().equals(wanted))
                .findFirst().orElse(null);
        if (next == null) {
            next = plans.stream().filter(p -> p.recipe().assemble(p.input()).is(ItemTags.PLANKS))
                    .findFirst().orElse(null);
        }
        if (next != null) result.add("next_step", step("craft", next.holder().id().identifier().toString(), BlockPos.ZERO));
        else result.addProperty("missing", logs == 0 && planks < 4 ? "Collect logs, then craft planks"
                : !table ? "Place your table on a listed placement site; do not search for another"
                : "Need three planks and two sticks");
        return result;
    }

    private static JsonObject step(String action, String target, BlockPos pos) {
        var result = new JsonObject();
        result.addProperty("action", action);
        result.addProperty("target", target);
        result.addProperty("x", pos.getX());
        result.addProperty("y", pos.getY());
        result.addProperty("z", pos.getZ());
        return result;
    }

    static Plan plan(ServerPlayer player, RecipeHolder<?> holder, boolean table) {
        if (!(holder.value() instanceof CraftingRecipe recipe)) return null;
        int width;
        int height;
        List<Optional<Ingredient>> ingredients;
        if (recipe instanceof ShapedRecipe shaped) {
            width = shaped.getWidth();
            height = shaped.getHeight();
            ingredients = shaped.getIngredients();
        } else if (recipe instanceof ShapelessRecipe) {
            List<Ingredient> list = recipe.placementInfo().ingredients();
            width = list.size() <= 4 ? 2 : 3;
            height = (list.size() + width - 1) / width;
            ingredients = new ArrayList<>();
            for (Ingredient ingredient : list) ingredients.add(Optional.of(ingredient));
            while (ingredients.size() < width * height) ingredients.add(Optional.empty());
        } else {
            return null;
        }
        if (!table && (width > 2 || height > 2)) return null;
        Inventory inventory = player.getInventory();
        int[] remaining = new int[36];
        for (int i = 0; i < remaining.length; i++) remaining[i] = inventory.getItem(i).getCount();
        int[] slots = new int[ingredients.size()];
        java.util.Arrays.fill(slots, -1);
        if (!assign(ingredients, 0, inventory, remaining, slots, new int[]{4096})) return null;
        List<ItemStack> grid = new ArrayList<>(Collections.nCopies(width * height, ItemStack.EMPTY));
        for (int i = 0; i < slots.length; i++) {
            if (slots[i] >= 0) grid.set(i, inventory.getItem(slots[i]).copyWithCount(1));
        }
        CraftingInput input = CraftingInput.of(width, height, grid);
        if (!recipe.matches(input, player.level())) return null;
        return new Plan(holder, recipe, input, slots);
    }

    private static boolean assign(List<Optional<Ingredient>> ingredients, int index, Inventory inventory,
                                  int[] remaining, int[] slots, int[] budget) {
        if (--budget[0] < 0) return false;
        if (index == ingredients.size()) return true;
        if (ingredients.get(index).isEmpty()) {
            return assign(ingredients, index + 1, inventory, remaining, slots, budget);
        }
        for (int i = 0; i < remaining.length; i++) {
            if (remaining[i] > 0 && ingredients.get(index).get().test(inventory.getItem(i))) {
                remaining[i]--;
                slots[index] = i;
                if (assign(ingredients, index + 1, inventory, remaining, slots, budget)) return true;
                remaining[i]++;
                slots[index] = -1;
            }
        }
        return false;
    }

    static String craft(ServerPlayer player, String id) {
        Plan plan = available(player).stream()
                .filter(p -> p.holder().id().identifier().toString().equals(id)).findFirst().orElse(null);
        if (plan == null) throw new IllegalArgumentException("Recipe unavailable: ingredients or reachable table missing");
        ItemStack output = plan.recipe().assemble(plan.input());
        if (output.isEmpty()) throw new IllegalArgumentException("Recipe produced no output");
        var remainders = plan.recipe().getRemainingItems(plan.input());
        for (int slot : plan.slots()) {
            if (slot >= 0) player.getInventory().removeItem(slot, 1);
        }
        player.getInventory().placeItemBackInInventory(output.copy(), net.minecraft.util.Prediction.SERVER_ONLY);
        for (ItemStack remainder : remainders) {
            if (!remainder.isEmpty()) {
                player.getInventory().placeItemBackInInventory(remainder, net.minecraft.util.Prediction.SERVER_ONLY);
            }
        }
        player.awardRecipes(List.of(plan.holder()));
        player.inventoryMenu.broadcastChanges();
        invalidateCache();
        return "Crafted " + output.getCount() + " " + output.getItem().toString();
    }
}
