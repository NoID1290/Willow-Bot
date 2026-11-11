package local.minecraft.ai;

import com.google.gson.JsonObject;
import java.util.Set;
import net.fabricmc.fabric.api.gametest.v1.GameTest;
import net.minecraft.core.BlockPos;
import net.minecraft.gametest.framework.GameTestHelper;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;

public final class LocalAiGameTests {
    static JsonObject decision(String action, String target, BlockPos pos) {
        JsonObject json = new JsonObject();
        json.addProperty("id", "test-" + action);
        json.addProperty("action", action);
        json.addProperty("target", target);
        json.addProperty("x", pos.getX());
        json.addProperty("y", pos.getY());
        json.addProperty("z", pos.getZ());
        return json;
    }

    @GameTest(maxTicks = 1000000, skyAccess = true)
    public void survivalPlayer(GameTestHelper helper) {
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) helper.setBlock(x, 1, z, Blocks.STONE);
        }
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer());
        BlockPos start = helper.absolutePos(new BlockPos(2, 2, 2));
        bot.teleportTo(helper.getLevel(), start.getX() + 0.5, start.getY(), start.getZ() + 0.5,
                Set.of(), 0, 0, true);
        bot.getInventory().clearContent();
        helper.assertTrue(bot.gameMode.getGameModeForPlayer() == GameType.SURVIVAL, "Bot must be survival");
        helper.assertTrue(bot.connection != null, "Bot must have a real server-player connection");
        verifyCommandPermissions(helper, bot);
        bot.getInventory().add(new ItemStack(Items.OAK_LOG, 3));
        Crafting.craft(bot, "minecraft:oak_planks");
        helper.assertTrue(bot.getInventory().countItem(Items.OAK_LOG) == 2, "Crafting consumes exactly one log");
        helper.assertTrue(bot.getInventory().countItem(Items.OAK_PLANKS) == 4, "Crafting yields four planks");
        Crafting.craft(bot, "minecraft:crafting_table");
        Actions actions = new Actions(bot);
        var crafting = actions.observe(java.util.List.of()).getAsJsonObject("crafting");
        helper.assertTrue(crafting.get("grid").getAsString().equals("2x2")
                        && crafting.get("table_in_inventory").getAsInt() == 1
                        && crafting.getAsJsonObject("next_step").get("action").getAsString().equals("place"),
                "An owned table must be placed, not searched for: " + crafting);
        helper.assertTrue(actions.observe(java.util.List.of()).getAsJsonArray("placeable").asList().stream()
                        .anyMatch(item -> item.getAsString().equals("minecraft:crafting_table")),
                "Observation exposes actual inventory block items");
        BlockPos table = helper.absolutePos(new BlockPos(3, 2, 2));
        actions.start(decision("place", "minecraft:crafting_table", table));
        helper.assertTrue(helper.getLevel().getBlockState(table).is(Blocks.CRAFTING_TABLE),
                "Place must use inventory block on a real support face: " + actions.result());
        helper.assertTrue(bot.getInventory().countItem(Items.CRAFTING_TABLE) == 0, "Placement consumes table");
        helper.assertTrue(actions.result().get("block_position").getAsString().equals(table.toShortString()),
                "Placement result must report the exact block position for Willow's memory: " + actions.result());
        var watch = decision("wait", "", BlockPos.ZERO);
        var watched = new com.google.gson.JsonArray();
        watched.add(new com.google.gson.JsonArray());
        watched.get(0).getAsJsonArray().add(table.getX());
        watched.get(0).getAsJsonArray().add(table.getY());
        watched.get(0).getAsJsonArray().add(table.getZ());
        watch.add("watch", watched);
        actions.start(watch);
        var check = actions.observe(java.util.List.of()).getAsJsonArray("remembered_blocks");
        helper.assertTrue(check.size() == 1 && check.get(0).getAsJsonObject().get("id").getAsString()
                .equals("minecraft:crafting_table"), "Remembered placements must be checked in the world: " + check);
        actions.stop("memory check done");
        bot.getInventory().add(new ItemStack(Items.COBBLESTONE_SLAB, 18));
        bot.getInventory().add(new ItemStack(Items.STONE_PICKAXE));
        var slabObservation = actions.observe(java.util.List.of());
        helper.assertFalse(slabObservation.getAsJsonArray("placeable").asList().stream()
                        .anyMatch(item -> item.getAsString().equals("minecraft:stone_pickaxe")),
                "Tools must not be offered as placeable blocks");
        var slabSite = slabObservation.getAsJsonObject("crafting").getAsJsonArray("placement_sites").get(0).getAsJsonArray();
        BlockPos slab = new BlockPos(slabSite.get(0).getAsInt(), slabSite.get(1).getAsInt(), slabSite.get(2).getAsInt());
        actions.start(decision("place", "minecraft:cobblestone_slab", slab));
        helper.assertTrue(helper.getLevel().getBlockState(slab).is(Blocks.COBBLESTONE_SLAB)
                        && actions.result().get("status").getAsString().equals("success")
                        && bot.getInventory().countItem(Items.COBBLESTONE_SLAB) == 17,
                "A reported placement site must allow a real survival slab placement: " + actions.result());
        helper.getLevel().setBlockAndUpdate(slab, Blocks.AIR.defaultBlockState());
        Crafting.craft(bot, "minecraft:oak_planks");
        Crafting.craft(bot, "minecraft:oak_planks");
        Crafting.craft(bot, "minecraft:stick");
        Crafting.craft(bot, "minecraft:wooden_pickaxe");
        helper.assertTrue(bot.getInventory().countItem(Items.WOODEN_PICKAXE) == 1, "3x3 tool recipe works");
        helper.setBlock(3, 2, 2, Blocks.AIR);
        bot.getInventory().clearContent();
        bot.getInventory().add(new ItemStack(Items.CRAFTING_TABLE));
        bot.getInventory().add(new ItemStack(Items.SPRUCE_LOG, 12));
        bot.getInventory().add(new ItemStack(Items.STICK, 16));
        var blockedPlacement = Crafting.awareness(bot, Crafting.available(bot), java.util.List.of());
        helper.assertFalse(blockedPlacement.has("next_step"),
                "A missing table placement site must not waste all logs crafting more planks");
        for (int step = 0; step < 4 && bot.getInventory().countItem(Items.WOODEN_PICKAXE) == 0; step++) {
            var next = actions.observe(java.util.List.of()).getAsJsonObject("crafting").getAsJsonObject("next_step");
            helper.assertTrue(next != null, "Saved-world inventory needs an actionable crafting next step");
            next.addProperty("id", "saved-inventory-" + step);
            actions.start(next);
            helper.assertTrue(actions.result().get("status").getAsString().equals("success"),
                    "Saved-world crafting next step must work: " + actions.result());
        }
        helper.assertTrue(bot.getInventory().countItem(Items.WOODEN_PICKAXE) == 1
                        && bot.getInventory().countItem(Items.SPRUCE_LOG) == 11
                        && bot.getInventory().countItem(Items.STICK) == 14
                        && bot.getInventory().countItem(Items.CRAFTING_TABLE) == 0,
                "Place the owned table and consume the exact spruce/stick ingredients for a pickaxe");
        helper.setBlock(4, 2, 3, Blocks.OAK_LOG);
        helper.setBlock(3, 2, 3, Blocks.OAK_LOG);
        actions.start(decision("mine", "minecraft:oak_log", helper.absolutePos(new BlockPos(4, 2, 3))));
        var liveTick = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        helper.onEachTick(() -> {
            actions.tick();
            if (liveTick.get() != null) liveTick.get().run();
        });
        helper.runAtTickTime(180, () -> {
            helper.assertTrue(helper.getBlockState(new BlockPos(4, 2, 3)).isAir(),
                    "Continuous survival mining must break the log: " + actions.result());
            helper.assertTrue(helper.getBlockState(new BlockPos(3, 2, 3)).is(Blocks.OAK_LOG),
                    "Explicit mining coordinates must not silently select a nearer block");
            helper.setBlock(3, 2, 3, Blocks.AIR);
            for (var entity : actions.nearbyEntities()) {
                if (entity instanceof net.minecraft.world.entity.item.ItemEntity) entity.discard();
            }
            var item = new net.minecraft.world.entity.item.ItemEntity(helper.getLevel(),
                    start.getX() + 2.5, start.getY() + 1.2, start.getZ() + 0.5, new ItemStack(Items.DIAMOND));
            item.setNoGravity(true);
            item.setDeltaMovement(net.minecraft.world.phys.Vec3.ZERO);
            helper.getLevel().addFreshEntity(item);
            var nearer = new net.minecraft.world.entity.item.ItemEntity(helper.getLevel(),
                    bot.getX(), bot.getY(), bot.getZ(), new ItemStack(Items.DIRT));
            helper.getLevel().addFreshEntity(nearer);
            actions.start(decision("pickup", "minecraft:diamond", BlockPos.ZERO));
        });
        helper.runAtTickTime(250, () -> {
            helper.assertTrue(bot.getInventory().countItem(Items.DIAMOND) == 1,
                    "Pickup must reach an airborne drop from a walkable cell below it: " + actions.result());
            BlockPos destination = helper.absolutePos(new BlockPos(5, 2, 5));
            helper.setBlock(5, 2, 5, Blocks.SNOW.defaultBlockState()
                    .setValue(net.minecraft.world.level.block.SnowLayerBlock.LAYERS, 6));
            actions.start(decision("goto", "", destination));
        });
        helper.runAtTickTime(320, () -> {
            BlockPos destination = helper.absolutePos(new BlockPos(5, 2, 5));
            helper.assertTrue(bot.blockPosition().distManhattan(destination) <= 1,
                    "Fake player must physically walk to destination: " + bot.blockPosition() + " " + actions.result());
            helper.assertTrue(bot.getY() > destination.getY() + 0.5,
                    "Player must actually stand on the layered snow, not just plan a path");
            helper.setBlock(6, 1, 5, Blocks.LAVA);
            helper.assertFalse(Navigator.safe(helper.getLevel(), helper.absolutePos(new BlockPos(6, 2, 5))),
                    "Navigator rejects lava");
            helper.assertTrue(Navigator.find(helper.getLevel(), bot.blockPosition(),
                    helper.absolutePos(new BlockPos(6, 2, 5))).isEmpty(), "Cannot path into lava");
            bot.getFoodData().setFoodLevel(10);
            bot.getInventory().add(new ItemStack(Items.BREAD));
            actions.start(decision("eat", "", BlockPos.ZERO));
        });
        helper.runAtTickTime(400, () -> {
            helper.assertTrue(bot.getFoodData().getFoodLevel() > 10, "Eating must restore hunger");
            helper.assertTrue(bot.getInventory().countItem(Items.BREAD) == 0, "Eating must consume food");
            actions.start(decision("goto", "", bot.blockPosition().offset(100, 0, 0)));
            helper.assertTrue(actions.result().get("status").getAsString().equals("failed"),
                    "Out-of-range action must fail explicitly");
            for (int x = 1; x <= 3; x++) {
                for (int z = 1; z <= 3; z++) helper.setBlock(x, 6, z, Blocks.OAK_LEAVES);
            }
            BlockPos tree = helper.absolutePos(new BlockPos(2, 7, 2));
            bot.teleportTo(helper.getLevel(), tree.getX() + 0.5, tree.getY(), tree.getZ() + 0.5,
                    Set.of(), 0, 0, true);
            bot.setHealth(20);
            BlockPos ground = helper.absolutePos(new BlockPos(4, 2, 2));
            helper.assertTrue(Navigator.find(helper.getLevel(), tree, ground).isEmpty(),
                    "Normal navigation must not silently take a damaging five-block drop");
            helper.assertFalse(Navigator.find(helper.getLevel(), tree, ground, 12).isEmpty(),
                    "Experimental recovery must permit a damaging descent");
            actions.start(decision("escape", "", BlockPos.ZERO));
        });
        helper.runAtTickTime(560, () -> {
            helper.assertTrue(bot.getY() < helper.absolutePos(new BlockPos(2, 4, 2)).getY(),
                    "Willow must physically leave the treetop: " + actions.result());
            helper.assertTrue(bot.getHealth() < 20, "Risky recovery must allow real fall damage");
            helper.assertTrue(actions.result().get("action").getAsString().equals("escape")
                            && actions.result().get("status").getAsString().equals("success")
                            && actions.result().get("damage").getAsFloat() > 0,
                    "Record landing and damage, not just a planned recovery: " + actions.result());
            for (int x = 5; x <= 7; x++) {
                for (int z = 5; z <= 7; z++) {
                    for (int y = 2; y <= 5; y++) {
                        helper.setBlock(x, y, z, x == 6 && z == 6 ? Blocks.AIR : Blocks.STONE);
                    }
                }
            }
            helper.setBlock(6, 2, 6, Blocks.OAK_LEAVES);
            BlockPos trapped = helper.absolutePos(new BlockPos(6, 3, 6));
            bot.teleportTo(helper.getLevel(), trapped.getX() + 0.5, trapped.getY(), trapped.getZ() + 0.5,
                    Set.of(), 0, 0, true);
            actions.start(decision("escape", "", BlockPos.ZERO));
        });
        helper.runAtTickTime(680, () -> {
            helper.assertTrue(helper.getBlockState(new BlockPos(6, 2, 6)).isAir(),
                    "When no descent exists, recovery must physically break natural support");
            helper.assertTrue(actions.result().get("status").getAsString().equals("success"),
                    "Recovery must finish after settling: " + actions.result());
            var server = helper.getLevel().getServer();
            BlockPos beforeReconnect = bot.blockPosition();
            server.getPlayerList().remove(bot);
            ServerPlayer restored = LocalAiMod.spawn(server);
            helper.assertTrue(restored.getInventory().countItem(Items.WOODEN_PICKAXE) == 1,
                    "Inventory must survive disconnect/reconnect");
            helper.assertTrue(restored.blockPosition().distManhattan(beforeReconnect) <= 1,
                    "Position must survive disconnect/reconnect");
            if (Boolean.getBoolean("localai.liveTest")) {
                liveTick.set(liveBrain(helper, restored));
            } else {
                server.getPlayerList().remove(restored);
                verifyLifecycleAndCamera(helper);
                helper.succeed();
            }
        });
    }

    @GameTest(maxTicks = 400, skyAccess = true)
    public void travelTowardFarDestination(GameTestHelper helper) {
        for (int x = 0; x < 18; x++) {
            for (int z = 0; z < 5; z++) helper.setBlock(x, 1, z, Blocks.STONE);
        }
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_Travel_Test".getBytes()), "Willow_Travel"));
        BlockPos start = helper.absolutePos(new BlockPos(2, 2, 2));
        bot.teleportTo(helper.getLevel(), start.getX() + 0.5, start.getY(), start.getZ() + 0.5, Set.of(), 0, 0, true);
        Actions actions = new Actions(bot);
        BlockPos far = helper.absolutePos(new BlockPos(300, 2, 2));
        helper.runAfterDelay(5, () -> actions.start(decision("explore", "", far)));
        helper.onEachTick(actions::tick);
        helper.runAtTickTime(200, () -> {
            helper.assertTrue(start.distSqr(far) - bot.blockPosition().distSqr(far) >= 4 * 2 * 296,
                    "Explore with a far heading must walk toward it: " + bot.blockPosition() + " " + actions.result());
            helper.assertTrue(actions.result().get("detail").getAsString().startsWith("Heading toward"),
                    "Result must say where Willow was heading: " + actions.result());
            helper.getLevel().getServer().getPlayerList().remove(bot);
            helper.succeed();
        });
    }

    @GameTest
    public void detectCircularProgress(GameTestHelper helper) {
        ProgressTracker tracker = new ProgressTracker();
        for (int i = 0; i < 7; i++) tracker.record("explore", "success", new BlockPos(i % 2, 70, i % 3));
        helper.assertTrue(tracker.problem().isEmpty(), "Do not diagnose a loop before the threshold");
        tracker.record("explore", "success", BlockPos.containing(0, 70, 0));
        helper.assertFalse(tracker.problem().isEmpty(), "Successful movement in circles is still stuck");
        tracker.record("escape", "success", new BlockPos(4, 65, 0));
        helper.assertTrue(tracker.problem().isEmpty(), "Recovery success clears the old movement loop");
        for (int i = 0; i < 3; i++) tracker.record("mine", "interrupted", new BlockPos(4, 65, 0));
        helper.assertTrue(tracker.problem().isEmpty(), "Eating/chat interruptions are not failures");
        for (int i = 0; i < 3; i++) tracker.record("goto", "failed", new BlockPos(4, 65, 0));
        helper.assertFalse(tracker.problem().isEmpty(), "Three failures must diagnose a problem");
        helper.assertTrue(new LocalAiMod.Config().thinkIntervalTicks == 0,
                "No artificial scheduling wait after an action");
        helper.succeed();
    }

    @GameTest
    public void craftingCacheValidation(GameTestHelper helper) {
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_Craft_Test".getBytes()), "Willow_Craft"));
        Crafting.invalidateCache();
        var first = Crafting.available(bot);
        var second = Crafting.available(bot);
        helper.assertTrue(first == second, "Consecutive Crafting.available calls must return identical cached list reference");
        Crafting.invalidateCache();
        var third = Crafting.available(bot);
        helper.assertTrue(first != third, "Crafting.invalidateCache must force recreation of available recipes list");
        helper.getLevel().getServer().getPlayerList().remove(bot);
        helper.succeed();
    }

    @GameTest
    public void willowViewDirtyCheck(GameTestHelper helper) {
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_View_Test".getBytes()), "Willow_View"));
        WillowView.clear();
        WillowView.capture(bot);
        var first = WillowView.current();
        helper.assertTrue(first != null, "WillowView.capture must populate snapshot");
        WillowView.capture(bot);
        var second = WillowView.current();
        helper.assertTrue(first == second, "Unchanged player state must reuse previous snapshot reference");
        WillowView.clear();
        helper.assertTrue(WillowView.current() == null, "WillowView.clear must null snapshot");
        helper.getLevel().getServer().getPlayerList().remove(bot);
        helper.succeed();
    }

    @GameTest
    public void unifiedWorkstationScan(GameTestHelper helper) {
        helper.setBlock(2, 2, 2, Blocks.CRAFTING_TABLE);
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_Scan_Test".getBytes()), "Willow_Scan"));
        BlockPos start = helper.absolutePos(new BlockPos(1, 2, 2));
        bot.teleportTo(helper.getLevel(), start.getX() + 0.5, start.getY(), start.getZ() + 0.5,
                Set.of(), 0, 0, true);
        Actions actions = new Actions(bot);
        var obs = actions.observe(java.util.List.of());
        helper.assertTrue(obs.has("nearby_workstations"), "Observation must include nearby_workstations");
        var ws = obs.getAsJsonObject("nearby_workstations");
        helper.assertTrue(ws.has("crafting_table"), "Nearby crafting table must be scanned and registered in workstations: " + ws);
        var sight = obs.getAsJsonObject("sight");
        helper.assertTrue(sight.has("horizon") && sight.getAsJsonArray("horizon").size() == 8,
                "Sight perception must expose 8-sector multi-chunk horizon: " + sight);
        helper.getLevel().getServer().getPlayerList().remove(bot);
        helper.succeed();
    }

    @GameTest(maxTicks = 200, skyAccess = true)
    public void frontierExploreIgnoresZeroOriginSentinel(GameTestHelper helper) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 5; z++) helper.setBlock(x, 1, z, Blocks.STONE);
        }
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_Frontier_Test".getBytes()), "Willow_Frontier"));
        BlockPos start = helper.absolutePos(new BlockPos(2, 2, 2));
        bot.teleportTo(helper.getLevel(), start.getX() + 0.5, start.getY(), start.getZ() + 0.5, Set.of(), -90.0f, 0.0f, true);
        Actions actions = new Actions(bot);
        helper.runAfterDelay(2, () -> actions.start(decision("explore", "", BlockPos.ZERO)));
        helper.onEachTick(actions::tick);
        helper.runAtTickTime(120, () -> {
            String detail = actions.result().has("detail") ? actions.result().get("detail").getAsString() : "";
            helper.assertFalse(detail.contains("Heading toward 0, 0, 0"),
                    "Open explore (0,0,0) must never pull Willow toward world origin (0,0,0): " + actions.result());
            helper.assertTrue(bot.blockPosition().distSqr(start) >= 16.0,
                    "Frontier explore must smoothly traverse across the platform: " + bot.blockPosition() + " " + actions.result());
            helper.getLevel().getServer().getPlayerList().remove(bot);
            helper.succeed();
        });
    }

    @GameTest
    public void layeredTerrain(GameTestHelper helper) {
        for (int x = 1; x <= 5; x++) helper.setBlock(x, 1, 2, Blocks.STONE);
        helper.setBlock(4, 1, 2, Blocks.DIRT_PATH);
        helper.setBlock(2, 2, 2, Blocks.SNOW.defaultBlockState()
                .setValue(net.minecraft.world.level.block.SnowLayerBlock.LAYERS, 4));
        helper.setBlock(3, 2, 2, Blocks.OAK_SLAB);
        BlockPos snow = helper.absolutePos(new BlockPos(2, 2, 2));
        BlockPos slab = helper.absolutePos(new BlockPos(3, 2, 2));
        BlockPos dirtPath = helper.absolutePos(new BlockPos(4, 2, 2));
        helper.assertTrue(Navigator.safe(helper.getLevel(), snow), "Layered snow is walkable");
        helper.assertTrue(Navigator.safe(helper.getLevel(), slab), "Bottom slabs are walkable");
        helper.assertTrue(Navigator.safe(helper.getLevel(), dirtPath), "Village dirt paths are walkable");
        helper.assertTrue(SightPerception.GAZE_MAX_DISTANCE >= 96.0 && Navigator.MAX_SEARCH_RADIUS >= 64,
                "Vision and navigation cover at least 4-6 chunks");
        helper.assertFalse(Navigator.find(helper.getLevel(), helper.absolutePos(new BlockPos(1, 2, 2)),
                helper.absolutePos(new BlockPos(5, 2, 2))).isEmpty(), "Path crosses snow, slabs, and dirt paths");
        helper.setBlock(2, 2, 2, Blocks.POWDER_SNOW);
        helper.assertFalse(Navigator.safe(helper.getLevel(), snow), "Powder snow remains unsafe");
        helper.succeed();
    }

    @GameTest(maxTicks = 200, skyAccess = true)
    public void placeUnderFeetToGetHigher(GameTestHelper helper) {
        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) helper.setBlock(x, 1, z, Blocks.STONE);
        }
        ServerPlayer bot = LocalAiMod.spawn(helper.getLevel().getServer(), new com.mojang.authlib.GameProfile(
                java.util.UUID.nameUUIDFromBytes("Willow_Pillar_Test".getBytes()), "Willow_Pillar"));
        BlockPos start = helper.absolutePos(new BlockPos(2, 2, 2));
        bot.teleportTo(helper.getLevel(), start.getX() + 0.5, start.getY(), start.getZ() + 0.5, Set.of(), 0, 0, true);
        bot.getInventory().clearContent();
        bot.getInventory().add(new ItemStack(Items.DIRT, 5));
        Actions actions = new Actions(bot);
        var observation = actions.observe(java.util.List.of());
        var sites = observation.getAsJsonObject("crafting").getAsJsonArray("placement_sites");
        boolean hasFeetSite = false;
        for (var el : sites) {
            var arr = el.getAsJsonArray();
            if (arr.get(0).getAsInt() == start.getX() && arr.get(1).getAsInt() == start.getY() && arr.get(2).getAsInt() == start.getZ()) {
                hasFeetSite = true;
                break;
            }
        }
        helper.assertTrue(hasFeetSite, "Placement sites must include feet position for getting higher");
        actions.start(decision("place", "minecraft:dirt", start));
        helper.onEachTick(actions::tick);
        helper.runAtTickTime(60, () -> {
            helper.assertTrue(helper.getLevel().getBlockState(start).is(Blocks.DIRT),
                    "Dirt block must be placed at feet position: " + actions.result());
            helper.assertTrue(bot.getBlockY() == start.getY() + 1,
                    "Bot must be elevated one block higher: " + bot.blockPosition());
            helper.assertTrue(bot.getInventory().countItem(Items.DIRT) == 4,
                    "Pillaring consumes one dirt block");
            helper.getLevel().getServer().getPlayerList().remove(bot);
            helper.succeed();
        });
    }

    private static void verifyCommandPermissions(GameTestHelper helper, ServerPlayer bot) {
        var noPermissions = net.minecraft.commands.Commands.createCompilationContext(
                net.minecraft.server.permissions.PermissionSet.NO_PERMISSIONS);
        var operator = net.minecraft.commands.Commands.createCompilationContext(
                net.minecraft.server.permissions.PermissionSet.ALL_PERMISSIONS);
        helper.assertTrue(noPermissions.getServer() == null, "Reproduce the login packet's serverless permission probe");
        for (var environment : java.util.List.of(net.minecraft.commands.Commands.CommandSelection.INTEGRATED,
                net.minecraft.commands.Commands.CommandSelection.DEDICATED)) {
            var commands = new net.minecraft.commands.Commands(environment,
                    net.minecraft.commands.CommandBuildContext.simple(helper.getLevel().registryAccess(),
                            helper.getLevel().enabledFeatures()));
            var requirement = commands.getDispatcher().getRoot().getChild("brain").getRequirement();
            helper.assertTrue(requirement.test(noPermissions)
                            == (environment == net.minecraft.commands.Commands.CommandSelection.INTEGRATED),
                    "Singleplayer commands need no cheats; dedicated commands require permission");
            helper.assertTrue(requirement.test(operator), "Operators may control Willow");
            for (String name : java.util.List.of("START", "STOP")) {
                var alias = commands.getDispatcher().getRoot().getChild(name).getRequirement();
                helper.assertTrue(alias.test(noPermissions)
                                == (environment == net.minecraft.commands.Commands.CommandSelection.INTEGRATED),
                        name + " must be safe during serverless login permission probing");
                helper.assertTrue(alias.test(operator), name + " allows operators");
            }
            commands.sendCommands(bot);
        }
    }

    private static void verifyLifecycleAndCamera(GameTestHelper helper) {
        var server = helper.getLevel().getServer();
        var controller = LocalAiMod.instance();
        var configFile = net.fabricmc.loader.api.FabricLoader.getInstance().getConfigDir().resolve("localai/config.json");
        try {
            java.nio.file.Files.createDirectories(configFile.getParent());
            java.nio.file.Files.writeString(configFile, """
                    {"enabled":false,"allowDedicatedServer":true,"model":"gemma4:12b-it-qat",
                     "thinkIntervalTicks":0,"cinematicCamera":false}
                    """);
            java.nio.file.Files.deleteIfExists(server.getWorldPath(net.minecraft.world.level.storage.LevelResource.ROOT)
                    .resolve("localai/camera-viewer.json"));
            controller.started(server);
            controller.launch("gemma4:12b-it-qat");
            var first = server.getPlayerList().getPlayer(LocalAiMod.PLAYER_ID);
            helper.assertTrue(first != null, "Start creates Willow without reloading the world");
            first.getInventory().clearContent();
            first.getInventory().add(new ItemStack(Items.DIAMOND, 3));
            var director = new CameraDirector();
            director.load(server);
            var profile = new com.mojang.authlib.GameProfile(java.util.UUID.randomUUID(), "Camera_Test");
            var info = net.minecraft.server.level.ClientInformation.createDefault();
            var viewer = carpet.patches.EntityPlayerMPFake.respawnFake(server, helper.getLevel(), profile, info);
            viewer.setPos(first.getX(), first.getY(), first.getZ());
            server.getPlayerList().placeNewPlayer(new carpet.patches.FakeClientConnection(
                    net.minecraft.network.protocol.PacketFlow.SERVERBOUND), viewer,
                    new net.minecraft.server.network.CommonListenerCookie(profile, 0, info, false));
            viewer.setGameMode(GameType.SURVIVAL);
            BlockPos origin = helper.absolutePos(new BlockPos(1, 2, 1));
            viewer.teleportTo(helper.getLevel(), origin.getX() + 0.5, origin.getY(), origin.getZ() + 0.5,
                    Set.of(), 35, 10, true);
            var before = viewer.position();
            director.follow(viewer, first);
            helper.assertTrue(viewer.isSpectator() && viewer.getCamera() == first,
                    "Follow switches the viewer to a chunk-tracked spectator camera");
            var recovered = new CameraDirector();
            recovered.load(server);
            recovered.restore(server);
            helper.assertTrue(viewer.gameMode.getGameModeForPlayer() == GameType.SURVIVAL
                            && viewer.getCamera() == viewer && viewer.position().distanceToSqr(before) < 0.0001,
                    "Persistent return state restores the original position and game mode");
            server.getPlayerList().remove(viewer);
            var commands = server.getCommands().getDispatcher();
            commands.execute("STOP", first.createCommandSourceStack().withPermission(
                    net.minecraft.server.permissions.PermissionSet.ALL_PERMISSIONS));
            helper.assertTrue(server.getPlayerList().getPlayer(LocalAiMod.PLAYER_ID) == null,
                    "STOP must remove Carpet's fake player, not call its no-op disconnect");
            controller.launch("qwen3.5:4b");
            var second = server.getPlayerList().getPlayer(LocalAiMod.PLAYER_ID);
            helper.assertTrue(second != null && second != first
                            && second.getInventory().countItem(Items.DIAMOND) == 3
                            && controller.selectedModel().equals("qwen3.5:4b"),
                    "Restart switches models and preserves Willow's inventory");
            commands.execute("STOP", second.createCommandSourceStack().withPermission(
                    net.minecraft.server.permissions.PermissionSet.ALL_PERMISSIONS));
            helper.assertTrue(server.getPlayerList().getPlayer(LocalAiMod.PLAYER_ID) == null, "Second STOP removes Willow");
        } catch (java.io.IOException | com.mojang.brigadier.exceptions.CommandSyntaxException e) {
            throw new IllegalStateException("Lifecycle/camera regression failed", e);
        }
    }

    @GameTest
    public void cinematicMotionAndModelSelection(GameTestHelper helper) {
        var motion = new CameraMotion();
        var focus = new net.minecraft.world.phys.Vec3(10, 70, 20);
        var initial = motion.update(focus, 179, 0);
        helper.assertTrue(initial.position().distanceTo(focus) > 4, "Camera starts behind and above its subject");
        var moved = motion.update(focus.add(1, 0, 0), -179, 1.0 / 60);
        helper.assertTrue(moved.position().distanceTo(initial.position()) < 0.5,
                "Camera eases position and takes the short path across yaw wraparound");
        var teleported = motion.update(focus.add(100, 0, 0), 0, 1.0 / 60);
        helper.assertTrue(teleported.position().distanceTo(focus.add(100, 0, 0)) < 6,
                "Long teleports reset smoothing rather than flying through unloaded terrain");
        var zoomFocus = focus.add(100, 0, 0);
        CameraMotion.Pose zoomed = teleported;
        for (int frame = 0; frame < 120; frame++) zoomed = motion.update(zoomFocus, 0, 1.0 / 60, 16);
        helper.assertTrue(Math.abs(motion.distance() - 16) < 0.5 && zoomed.position().distanceTo(zoomFocus) > 15,
                "Zoom eases the camera out to the requested distance");
        motion.update(zoomFocus, 0, 1.0 / 60, 500);
        for (int frame = 0; frame < 240; frame++) motion.update(zoomFocus, 0, 1.0 / 60, 0);
        helper.assertTrue(Math.abs(motion.distance() - CameraMotion.MIN_DISTANCE) < 0.1,
                "Zoom is clamped to safe minimum/maximum distances");
        try {
            var models = OllamaModels.parse(com.google.gson.JsonParser.parseString("""
                    {"models":[{"name":"gemma4:12b-it-qat","capabilities":["completion","thinking"]},
                    {"name":"qwen3.5:4b"},{"name":"gemma4:12b-it-qat"},{"name":"gemma4:cloud"},
                    {"name":"remote:latest","remote_host":"https://example.invalid"},
                    {"name":"embed:latest","capabilities":["embedding"]}]}
                    """).getAsJsonObject());
            helper.assertTrue(models.equals(java.util.List.of("gemma4:12b-it-qat", "qwen3.5:4b")),
                    "Picker lists only distinct local completion models");
            helper.assertFalse(OllamaModels.validName("bad\nmodel"), "Reject control characters in model names");
        } catch (java.io.IOException e) {
            throw new IllegalStateException(e);
        }
        helper.succeed();
    }

    @GameTest
    public void modelSettingsAndConsole(GameTestHelper helper) {
        var settings = new ModelSettings();
        helper.assertTrue(settings.isEmpty() && settings.toJson().equals("{}"), "Blank settings mean fully automatic");
        settings.contextLength = 8192;
        settings.temperature = 0.6;
        settings.think = "auto";
        settings.validate();
        var json = com.google.gson.JsonParser.parseString(settings.toJson()).getAsJsonObject();
        helper.assertTrue(json.get("contextLength").getAsInt() == 8192 && !json.has("think")
                && json.get("temperature").getAsDouble() == 0.6, "Settings JSON must match the Python brain keys");
        settings.contextLength = 100;
        boolean refused = false;
        try {
            settings.validate();
        } catch (IllegalArgumentException e) {
            refused = true;
        }
        helper.assertTrue(refused, "A context below 4096 must be refused");
        var konsole = OllamaConsole.terminalCommand("konsole", java.util.List.of("python3", "monitor.py"));
        helper.assertTrue(konsole.getFirst().equals("konsole") && konsole.get(konsole.size() - 3).equals("-e")
                && konsole.getLast().equals("monitor.py"), "Konsole must run the monitor command: " + konsole);
        helper.succeed();
    }

    @GameTest
    public void ollamaStopsWithGame(GameTestHelper helper) {
        com.sun.net.httpserver.HttpServer fake = null;
        try {
            var unloaded = new java.util.concurrent.CopyOnWriteArrayList<String>();
            fake = com.sun.net.httpserver.HttpServer.create(
                    new java.net.InetSocketAddress(java.net.InetAddress.getLoopbackAddress(), 0), 0);
            fake.createContext("/api/version", exchange -> reply(exchange, "{\"version\":\"test\"}"));
            fake.createContext("/api/ps", exchange -> reply(exchange,
                    "{\"models\":[{\"name\":\"gemma4:12b-it-qat\",\"model\":\"gemma4:12b-it-qat\"},{\"model\":\"bad\\nname\"}]}"));
            fake.createContext("/api/generate", exchange -> {
                var body = com.google.gson.JsonParser.parseString(new String(exchange.getRequestBody().readAllBytes(),
                        java.nio.charset.StandardCharsets.UTF_8)).getAsJsonObject();
                if (body.get("keep_alive").getAsInt() == 0) unloaded.add(body.get("model").getAsString());
                reply(exchange, "{\"done\":true}");
            });
            fake.start();
            var base = java.net.URI.create("http://127.0.0.1:" + fake.getAddress().getPort());
            var service = new OllamaService(base, java.util.List.of("true"), "ollama.service");
            helper.assertTrue(service.ensureRunning(), "Running Ollama is detected without restarting it");
            helper.assertTrue(service.unloadAll() == 1 && unloaded.equals(java.util.List.of("gemma4:12b-it-qat")),
                    "Every loaded model is unloaded with keep_alive 0, ignoring invalid names");
            helper.assertTrue(service.runSystemctl("stop", 5), "Service control command succeeds");
            helper.assertFalse(new OllamaService(base, java.util.List.of("false"), "ollama.service").runSystemctl("stop", 5),
                    "Denied service control is reported, not thrown");
            fake.stop(0);
            fake = null;
            helper.assertFalse(service.reachable(), "Stopped Ollama is detected");
            helper.assertTrue(service.unloadAll() == 0, "Unloading a stopped Ollama is harmless");
            var config = java.nio.file.Files.createTempFile("localai-config", ".json");
            try {
                java.nio.file.Files.writeString(config, "{\"stopOllamaOnExit\":false}");
                helper.assertFalse(OllamaService.stopOnExitEnabled(config), "Config can keep Ollama running");
                java.nio.file.Files.writeString(config, "{\"model\":\"x\"}");
                helper.assertTrue(OllamaService.stopOnExitEnabled(config), "Stopping Ollama on exit is the default");
            } finally {
                java.nio.file.Files.deleteIfExists(config);
            }
        } catch (java.io.IOException e) {
            throw new IllegalStateException(e);
        } finally {
            if (fake != null) fake.stop(0);
        }
        helper.succeed();
    }

    private static void reply(com.sun.net.httpserver.HttpExchange exchange, String json) throws java.io.IOException {
        byte[] bytes = json.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        try (var out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    @GameTest
    public void rejectedPlanIsNotBrainOutage(GameTestHelper helper) {
        try {
            BrainBridge.decodeResponse(422, """
                    {"error":"decision_rejected","detail":"Placement target is blocked"}
                    """.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            helper.fail("Expected a typed rejected-plan response");
        } catch (BrainBridge.DecisionRejectedException expected) {
            helper.assertTrue(expected.getMessage().contains("blocked"), "Preserve the rejection reason");
        } catch (java.io.IOException e) {
            helper.fail("Plan rejection must not become a generic brain-service outage");
        }
        try {
            BrainBridge.decodeResponse(503, new byte[0]);
            helper.fail("A real outage must remain an error");
        } catch (BrainBridge.DecisionRejectedException e) {
            helper.fail("An outage must not be classified as a model mistake");
        } catch (java.io.IOException expected) {
            helper.assertTrue(expected.getMessage().contains("503"), "Preserve service outage status");
        }
        helper.succeed();
    }

    private Runnable liveBrain(GameTestHelper helper, ServerPlayer bot) {
        bot.getInventory().clearContent();
        bot.getFoodData().setFoodLevel(20);
        for (int x = 0; x <= 7; x++) {
            for (int z = 0; z <= 7; z++) {
                helper.setBlock(x, 1, z, Blocks.STONE);
                for (int y = 2; y <= 7; y++) helper.setBlock(x, y, z, Blocks.AIR);
            }
        }
        helper.setBlock(5, 2, 5, Blocks.AIR);
        helper.setBlock(6, 1, 5, Blocks.STONE);
        helper.setBlock(3, 2, 2, Blocks.AIR);
        helper.setBlock(2, 2, 5, Blocks.OAK_LOG);
        helper.setBlock(2, 3, 5, Blocks.OAK_LOG);
        helper.setBlock(2, 4, 5, Blocks.OAK_LOG);
        helper.assertFalse(Crafting.tableReachable(bot), "Live bootstrap must start without an existing table");
        java.nio.file.Path directory = java.nio.file.Path.of("live-brain-" + java.util.UUID.randomUUID());
        try {
            BrainBridge.extractScripts(directory);
        } catch (java.io.IOException e) {
            throw new IllegalStateException("Live brain setup failed", e);
        }
        BrainBridge brain = new BrainBridge(directory.resolve("server.py").toAbsolutePath(),
                directory.toAbsolutePath(), new LocalAiMod.Config().model);
        Actions liveActions = new Actions(bot);
        long started = System.nanoTime();
        return new Runnable() {
            private java.util.concurrent.CompletableFuture<JsonObject> pending;
            private int decisions;
            private boolean done;

            @Override
            public void run() {
                if (done) return;
                try {
                    if (System.nanoTime() - started > java.time.Duration.ofSeconds(600).toNanos()) {
                        throw new IllegalStateException("Live autonomous loop exceeded 600 seconds");
                    }
                    liveActions.tick();
                    if (bot.getInventory().countItem(Items.WOODEN_PICKAXE) > 0) {
                        done = true;
                        liveActions.stop("Live test complete");
                        brain.close();
                        helper.getLevel().getServer().getPlayerList().remove(bot);
                        LocalAiMod.LOG.info("LIVE OLLAMA PASS: empty inventory to crafted and placed table and wooden pickaxe in {} decisions", decisions);
                        helper.succeed();
                        return;
                    }
                    if (pending != null && pending.isDone()) {
                        JsonObject decision = pending.join();
                        pending = null;
                        LocalAiMod.LOG.info("LIVE OLLAMA DECISION: {}", decision);
                        liveActions.start(decision);
                        if (++decisions > 36) throw new IllegalStateException("No autonomous pickaxe after 36 decisions");
                    }
                    if (pending == null && !liveActions.busy()) {
                        pending = brain.decide(liveActions.observe(java.util.List.of()));
                    }
                } catch (RuntimeException e) {
                    done = true;
                    brain.close();
                    helper.fail("Live brain: " + e);
                }
            }
        };
    }
}
