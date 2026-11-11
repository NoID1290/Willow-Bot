package local.minecraft.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.imageio.ImageIO;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.BlockTags;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Multi-chunk visual perception engine for Willow (up to 6 chunks / 96 blocks).
 * Emits long-range gaze raycasts, wide-FOV entity tracking, panoramic heightmap landmarks,
 * 8-sector terrain horizon analysis, ASCII retina depth grid, and depth-shaded 64x64 snapshots.
 */
public final class SightPerception {

    static final double GAZE_MAX_DISTANCE = 96.0;
    static final double ENTITY_FOV_DISTANCE = 64.0;
    static final int LANDMARK_SCAN_RADIUS = 80;
    static final double RETINA_MAX_DISTANCE = 96.0;
    static final double SNAPSHOT_MAX_DISTANCE = 96.0;

    record LandmarkHit(BlockPos pos, String id, String category, double distance, String direction, boolean inFov) {}

    private static volatile String lastLookingAt = "Sky";
    private static volatile String lastVisible = "Clear FOV";
    private static volatile boolean snapshotEnabled = true;
    private volatile List<LandmarkHit> lastLandmarks = List.of();

    private static final ExecutorService ENCODER_POOL = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "SightPerception-PNG-Encoder");
        t.setDaemon(true);
        return t;
    });
    private static volatile CompletableFuture<String> pendingSnapshot = null;
    private static volatile String lastSnapshotBase64 = null;

    public static CompletableFuture<String> getPendingSnapshot() {
        return pendingSnapshot;
    }

    public static String getLastSnapshotBase64() {
        return lastSnapshotBase64;
    }

    public static String currentLookingAt() { return lastLookingAt; }
    public static String currentVisibleSummary() { return lastVisible; }

    /** The brain reports per decision whether it sends images; text-only profiles skip the raycast PNG. */
    public static void setSnapshotEnabled(boolean enabled) {
        snapshotEnabled = enabled;
    }

    List<LandmarkHit> recentLandmarks() {
        return lastLandmarks;
    }

    public JsonObject observe(ServerPlayer player) {
        JsonObject sight = new JsonObject();
        if (player == null || player.isRemoved()) {
            return sight;
        }

        Vec3 eyePos = player.getEyePosition();
        Vec3 viewVec = player.getViewVector(1.0f);
        float yaw = player.getYRot();
        float pitch = player.getXRot();

        sight.addProperty("yaw", Math.round(yaw * 10) / 10.0f);
        sight.addProperty("pitch", Math.round(pitch * 10) / 10.0f);
        sight.addProperty("facing", compassDirection(yaw));
        sight.addProperty("elevation", pitch < -25 ? "looking_up" : (pitch > 25 ? "looking_down" : "level"));
        sight.addProperty("vision_range_blocks", (int) GAZE_MAX_DISTANCE);

        int skyLight = player.level().getBrightness(LightLayer.SKY, player.blockPosition().above());
        int blockLight = player.level().getBrightness(LightLayer.BLOCK, player.blockPosition().above());
        boolean isDay = (player.level().getGameTime() % 24000) < 13000 && skyLight > 4;
        JsonObject lighting = new JsonObject();
        lighting.addProperty("sky", skyLight);
        lighting.addProperty("block", blockLight);
        lighting.addProperty("daylight", isDay);
        sight.add("lighting", lighting);

        // 1. Long-range crosshair raycast (up to 96 blocks / 6 chunks)
        JsonObject lookingAt = raycastGaze(player, eyePos, viewVec, GAZE_MAX_DISTANCE);
        sight.add("looking_at", lookingAt);
        if (lookingAt.has("id") && !"minecraft:air".equals(lookingAt.get("id").getAsString())) {
            double d = lookingAt.has("distance") ? lookingAt.get("distance").getAsDouble() : 0.0;
            String name = lookingAt.get("id").getAsString().replaceAll("^.*:", "");
            lastLookingAt = name + String.format(" (%.1fm)", d);
        } else {
            lastLookingAt = "Horizon (" + compassDirection(yaw) + ")";
        }

        // 2. Visible entities in wide Field of View (up to 64 blocks / 4 chunks, 130° horizontal cone)
        JsonArray visibleEntities = new JsonArray();
        List<EntitySight> seenEntities = scanVisibleEntities(player, eyePos, yaw, pitch, ENTITY_FOV_DISTANCE);
        for (EntitySight es : seenEntities.stream().limit(12).toList()) {
            JsonObject eo = new JsonObject();
            eo.addProperty("id", es.entity.getId());
            eo.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(es.entity.getType()).toString());
            eo.addProperty("distance", Math.round(es.distance * 10) / 10.0);
            eo.addProperty("bearing", es.horizontalBearing);
            eo.addProperty("elevation", es.elevation);
            eo.addProperty("hostile", es.entity instanceof Monster);
            if (es.entity instanceof ItemEntity item) {
                eo.addProperty("item", BuiltInRegistries.ITEM.getKey(item.getItem().getItem()).toString());
                eo.addProperty("count", item.getItem().getCount());
            }
            JsonArray at = new JsonArray();
            at.add(es.entity.blockPosition().getX());
            at.add(es.entity.blockPosition().getY());
            at.add(es.entity.blockPosition().getZ());
            eo.add("at", at);
            visibleEntities.add(eo);
        }
        sight.add("visible_entities", visibleEntities);

        // 3. Multi-chunk key blocks, distant landmarks (up to 80 blocks / 5 chunks), and 8-sector terrain horizon
        PanoramaScan panorama = scanPanoramaAndLandmarks(player, eyePos, viewVec);
        lastLandmarks = panorama.landmarks();
        sight.add("visible_blocks", panorama.visibleBlocks());
        sight.add("distant_landmarks", panorama.distantLandmarks());
        sight.add("horizon", panorama.horizon());

        // 4. 7x5 Long-Range ASCII Retina Depth Grid (up to 96 blocks / 6 chunks)
        String[] retina = generateAsciiRetina(player, eyePos, yaw, pitch, seenEntities);
        JsonArray retinaArray = new JsonArray();
        for (String row : retina) {
            retinaArray.add(row);
        }
        sight.add("retina", retinaArray);

        // 5. Natural language visual summary
        String summary = buildVisualSummary(yaw, pitch, lookingAt, seenEntities, panorama.visibleBlocks(),
                panorama.landmarks(), skyLight);
        sight.addProperty("visual_summary", summary);
        if (!seenEntities.isEmpty()) {
            lastVisible = seenEntities.size() + " entity/entities (" + seenEntities.getFirst().entity.getType().toShortString() + ")";
        } else if (!panorama.landmarks().isEmpty()) {
            LandmarkHit lm = panorama.landmarks().getFirst();
            lastVisible = lm.id().replaceAll("^.*:", "") + String.format(" %.0fm %s", lm.distance(), lm.direction());
        } else {
            lastVisible = "Open horizon (" + (int) GAZE_MAX_DISTANCE + "m)";
        }

        // 6. Base64 Visual Frame Snapshot (64x64 PNG, 6-chunk depth-shaded) for multimodal VL / Omni
        if (snapshotEnabled) {
            String frameBase64 = renderVisualSnapshot(player, eyePos, yaw, pitch, seenEntities);
            if (frameBase64 != null) {
                sight.addProperty("snapshot_png_base64", frameBase64);
            }
        }

        return sight;
    }

    private static BlockHitResult clipLoaded(ServerPlayer player, Vec3 start, Vec3 dir, double maxDist,
                                             ClipContext.Block blockMode, ClipContext.Fluid fluidMode) {
        ServerLevel level = player.level();
        double usableDist = 0.0;
        for (double step = 8.0; step <= maxDist; step += 8.0) {
            Vec3 p = start.add(dir.scale(step));
            if (!level.isLoaded(BlockPos.containing(p))) {
                break;
            }
            usableDist = step;
        }
        if (usableDist < maxDist) {
            Vec3 full = start.add(dir.scale(maxDist));
            if (level.isLoaded(BlockPos.containing(full))) {
                usableDist = maxDist;
            }
        }
        if (usableDist <= 0.5) {
            usableDist = Math.min(maxDist, 8.0);
        }
        Vec3 end = start.add(dir.scale(usableDist));
        return level.clip(new ClipContext(start, end, blockMode, fluidMode, player));
    }

    private JsonObject raycastGaze(ServerPlayer player, Vec3 eyePos, Vec3 viewVec, double maxDist) {
        JsonObject target = new JsonObject();
        BlockHitResult hit = clipLoaded(player, eyePos, viewVec, maxDist,
                ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY);

        double blockDist = maxDist;
        if (hit.getType() != HitResult.Type.MISS) {
            blockDist = eyePos.distanceTo(hit.getLocation());
        }

        // Entity check in crosshairs (up to 64 blocks)
        double entityMax = Math.min(maxDist, ENTITY_FOV_DISTANCE);
        Vec3 entityEnd = eyePos.add(viewVec.scale(entityMax));
        AABB box = player.getBoundingBox().expandTowards(viewVec.scale(entityMax)).inflate(1.0);
        Entity closestEntity = null;
        double closestDist = Math.min(blockDist, entityMax);

        for (Entity e : player.level().getEntities(player, box,
                ent -> ent instanceof LivingEntity || ent instanceof ItemEntity)) {
            AABB eBox = e.getBoundingBox().inflate(0.35);
            var opt = eBox.clip(eyePos, entityEnd);
            if (opt.isPresent()) {
                double d = eyePos.distanceTo(opt.get());
                if (d < closestDist) {
                    closestDist = d;
                    closestEntity = e;
                }
            }
        }

        if (closestEntity != null) {
            target.addProperty("kind", "entity");
            target.addProperty("id", BuiltInRegistries.ENTITY_TYPE.getKey(closestEntity.getType()).toString());
            target.addProperty("entity_id", closestEntity.getId());
            target.addProperty("distance", Math.round(closestDist * 10) / 10.0);
            target.addProperty("hostile", closestEntity instanceof Monster);
            JsonArray pos = new JsonArray();
            pos.add(closestEntity.blockPosition().getX());
            pos.add(closestEntity.blockPosition().getY());
            pos.add(closestEntity.blockPosition().getZ());
            target.add("pos", pos);
        } else if (hit.getType() == HitResult.Type.BLOCK) {
            BlockPos bp = hit.getBlockPos();
            BlockState state = player.level().getBlockState(bp);
            target.addProperty("kind", "block");
            target.addProperty("id", BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());
            target.addProperty("distance", Math.round(blockDist * 10) / 10.0);
            JsonArray pos = new JsonArray();
            pos.add(bp.getX());
            pos.add(bp.getY());
            pos.add(bp.getZ());
            target.add("pos", pos);
        } else {
            target.addProperty("kind", "air");
            target.addProperty("id", "minecraft:air");
            target.addProperty("distance", maxDist);
        }
        return target;
    }

    private record EntitySight(Entity entity, double distance, String horizontalBearing, String elevation) {}

    private List<EntitySight> scanVisibleEntities(ServerPlayer player, Vec3 eyePos, float yaw, float pitch, double maxDist) {
        List<EntitySight> result = new ArrayList<>();
        AABB area = player.getBoundingBox().inflate(maxDist, Math.min(32.0, maxDist), maxDist);
        List<Entity> list = player.level().getEntities(player, area,
                e -> (e instanceof LivingEntity || e instanceof ItemEntity) && !e.isSpectator());

        for (Entity e : list) {
            Vec3 targetPt = e.getEyePosition();
            Vec3 toEntity = targetPt.subtract(eyePos);
            double dist = toEntity.length();
            if (dist > maxDist || dist < 0.1) continue;
            if (!player.level().isLoaded(e.blockPosition())) continue;

            double entityYaw = Math.toDegrees(Math.atan2(-toEntity.x, toEntity.z));
            double relYaw = Mth.wrapDegrees(entityYaw - yaw);

            double entityPitch = -Math.toDegrees(Math.atan2(toEntity.y, Math.hypot(toEntity.x, toEntity.z)));
            // Use horizon-stabilized pitch when head was tilted slightly so walking gaze doesn't blind FOV
            double effectivePitch = Mth.clamp(pitch, -30.0f, 25.0f);
            double relPitch = entityPitch - effectivePitch;

            // Wide human FOV cone: ±65° horizontal (±85° peripheral within 14m), ±50° vertical
            double maxYawCone = dist <= 14.0 ? 85.0 : 65.0;
            if (Math.abs(relYaw) > maxYawCone || Math.abs(relPitch) > 50.0) {
                continue;
            }

            // Occlusion raycast along loaded chunks (ignoring leaves for large silhouettes within 32m)
            BlockHitResult occ = player.level().clip(new ClipContext(eyePos, targetPt,
                    ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, player));
            if (occ.getType() == HitResult.Type.BLOCK && eyePos.distanceTo(occ.getLocation()) < dist - 0.8) {
                BlockState blocker = player.level().getBlockState(occ.getBlockPos());
                if (!(dist <= 24.0 && blocker.is(BlockTags.LEAVES))) {
                    continue;
                }
            }

            String hBearing = Math.abs(relYaw) < 18.0 ? "center" : (relYaw < 0 ? "left" : "right");
            String vElevation = Math.abs(relPitch) < 15.0 ? "level" : (relPitch < 0 ? "above" : "below");
            result.add(new EntitySight(e, dist, hBearing, vElevation));
        }

        result.sort(Comparator.comparingInt((EntitySight es) -> {
            if (es.entity instanceof Monster && es.distance < 20.0) return 0;
            if (es.entity instanceof ItemEntity && es.distance < 16.0) return 1;
            return 2;
        }).thenComparingDouble(EntitySight::distance));
        return result;
    }

    private record PanoramaScan(JsonArray visibleBlocks, JsonArray distantLandmarks, JsonArray horizon,
                                List<LandmarkHit> landmarks) {}

    private static String classifyKeyBlock(BlockState s) {
        if (s.is(BlockTags.LOGS)) return "tree";
        if (s.is(Blocks.CRAFTING_TABLE) || s.is(Blocks.FURNACE) || s.is(Blocks.BLAST_FURNACE)
                || s.is(Blocks.SMOKER) || s.is(Blocks.CHEST) || s.is(Blocks.BARREL)
                || s.is(BlockTags.BEDS) || s.is(Blocks.BELL) || s.is(Blocks.HAY_BLOCK)
                || s.is(Blocks.NETHER_PORTAL)) return "structure";
        if (s.is(Blocks.COAL_ORE) || s.is(Blocks.DEEPSLATE_COAL_ORE)
                || s.is(Blocks.IRON_ORE) || s.is(Blocks.DEEPSLATE_IRON_ORE)
                || s.is(Blocks.COPPER_ORE) || s.is(Blocks.DEEPSLATE_COPPER_ORE)
                || s.is(Blocks.GOLD_ORE) || s.is(Blocks.EMERALD_ORE)
                || s.is(Blocks.DIAMOND_ORE)) return "ore";
        if (s.is(Blocks.STONE) || s.is(Blocks.COBBLESTONE) || s.is(Blocks.MOSSY_COBBLESTONE)
                || s.is(Blocks.OBSIDIAN) || s.is(Blocks.CRYING_OBSIDIAN)) return "stone";
        if (s.is(Blocks.LAVA)) return "hazard";
        if (s.is(Blocks.WATER)) return "water";
        return null;
    }

    private PanoramaScan scanPanoramaAndLandmarks(ServerPlayer player, Vec3 eyePos, Vec3 viewVec) {
        ServerLevel level = player.level();
        BlockPos center = player.blockPosition();
        int cx = center.getX();
        int cy = center.getY();
        int cz = center.getZ();

        Vec3 horizForward = new Vec3(viewVec.x, 0, viewVec.z);
        if (horizForward.lengthSqr() < 1e-4) {
            float fYaw = (float) Math.toRadians(player.getYRot());
            horizForward = new Vec3(-Math.sin(fYaw), 0, Math.cos(fYaw));
        } else {
            horizForward = horizForward.normalize();
        }

        List<LandmarkHit> candidates = new ArrayList<>();

        // Near 3D scan (radius 14) for immediate caves/interiors/overhangs
        for (int dx = -14; dx <= 14; dx += 2) {
            for (int dz = -14; dz <= 14; dz += 2) {
                if (dx == 0 && dz == 0) continue;
                BlockPos col = new BlockPos(cx + dx, cy, cz + dz);
                if (!level.isLoaded(col)) continue;
                for (int dy = -4; dy <= 6; dy++) {
                    BlockPos p = center.offset(dx, dy, dz);
                    BlockState s = level.getBlockState(p);
                    if (s.isAir()) continue;
                    String cat = classifyKeyBlock(s);
                    if (cat == null) continue;
                    addLandmarkCandidate(player, eyePos, horizForward, p, s, cat, true, candidates);
                }
            }
        }

        // Multi-chunk heightmap & cliff scan (15..80 blocks = up to 5 chunks!)
        for (int ring = 16; ring <= LANDMARK_SCAN_RADIUS; ring += 4) {
            int step = ring <= 36 ? 3 : 5;
            for (int d = -ring; d <= ring; d += step) {
                scanSurfaceColumn(player, level, eyePos, horizForward, cx + d, cz - ring, candidates);
                scanSurfaceColumn(player, level, eyePos, horizForward, cx + d, cz + ring, candidates);
                if (Math.abs(d) < ring) {
                    scanSurfaceColumn(player, level, eyePos, horizForward, cx - ring, cz + d, candidates);
                    scanSurfaceColumn(player, level, eyePos, horizForward, cx + ring, cz + d, candidates);
                }
            }
        }

        candidates.sort(Comparator.comparingInt((LandmarkHit h) -> switch (h.category()) {
            case "structure" -> 0;
            case "ore" -> 1;
            case "tree" -> 2;
            case "stone" -> 3;
            case "hazard" -> 4;
            default -> 5;
        }).thenComparingDouble(LandmarkHit::distance));

        JsonArray visibleBlocks = new JsonArray();
        Map<String, Integer> fovCounts = new HashMap<>();
        for (LandmarkHit hit : candidates) {
            if (!hit.inFov()) continue;
            if ("water".equals(hit.category()) && fovCounts.getOrDefault(hit.id(), 0) >= 1) continue;
            if (fovCounts.merge(hit.id(), 1, Integer::sum) > 3) continue;
            JsonObject obj = new JsonObject();
            obj.addProperty("id", hit.id());
            obj.addProperty("distance", hit.distance());
            obj.addProperty("direction", hit.direction());
            JsonArray at = new JsonArray();
            at.add(hit.pos().getX());
            at.add(hit.pos().getY());
            at.add(hit.pos().getZ());
            obj.add("at", at);
            visibleBlocks.add(obj);
            if (visibleBlocks.size() >= 12) break;
        }

        JsonArray distantLandmarks = new JsonArray();
        List<LandmarkHit> selectedLandmarks = new ArrayList<>();
        Map<String, Integer> landmarkSectorCounts = new HashMap<>();
        for (LandmarkHit hit : candidates) {
            if (hit.distance() < 12.0) continue;
            if ("water".equals(hit.category()) && landmarkSectorCounts.getOrDefault("water", 0) >= 2) continue;
            String dedupKey = hit.category() + ":" + hit.direction();
            if (landmarkSectorCounts.containsKey(dedupKey)) continue;
            landmarkSectorCounts.put(dedupKey, 1);
            landmarkSectorCounts.merge(hit.category(), 1, Integer::sum);
            selectedLandmarks.add(hit);

            JsonObject lm = new JsonObject();
            lm.addProperty("id", hit.id());
            lm.addProperty("category", hit.category());
            lm.addProperty("distance", hit.distance());
            lm.addProperty("direction", hit.direction());
            lm.addProperty("in_fov", hit.inFov());
            JsonArray at = new JsonArray();
            at.add(hit.pos().getX());
            at.add(hit.pos().getY());
            at.add(hit.pos().getZ());
            lm.add("at", at);
            distantLandmarks.add(lm);
            if (distantLandmarks.size() >= 10) break;
        }

        // 8-sector compass horizon (samples 16m, 32m, 48m, 64m = 4 full chunks in every direction)
        JsonArray horizon = buildHorizonSectors(level, center);
        return new PanoramaScan(visibleBlocks, distantLandmarks, horizon, selectedLandmarks);
    }

    private void scanSurfaceColumn(ServerPlayer player, ServerLevel level, Vec3 eyePos, Vec3 horizForward,
                                   int x, int z, List<LandmarkHit> out) {
        BlockPos check = new BlockPos(x, 64, z);
        if (!level.isLoaded(check)) return;

        int motionY = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, x, z);
        int worldY = level.getHeight(Heightmap.Types.WORLD_SURFACE, x, z);

        // Check surface ground block and block right above ground (stations, ores, stone cliffs, water/lava)
        for (int y : new int[]{motionY - 1, motionY, motionY + 1}) {
            BlockPos p = new BlockPos(x, y, z);
            BlockState s = level.getBlockState(p);
            if (s.isAir()) continue;
            String cat = classifyKeyBlock(s);
            if (cat != null) {
                addLandmarkCandidate(player, eyePos, horizForward, p, s, cat, false, out);
                break;
            }
        }

        // If leaf canopy sits above solid ground (worldY > motionY), check under canopy for tree trunks!
        if (worldY > motionY + 1) {
            for (int dy = 0; dy <= Math.min(5, worldY - motionY); dy++) {
                BlockPos trunkPos = new BlockPos(x, motionY + dy, z);
                BlockState ts = level.getBlockState(trunkPos);
                if (ts.is(BlockTags.LOGS)) {
                    addLandmarkCandidate(player, eyePos, horizForward, trunkPos, ts, "tree", false, out);
                    break;
                }
            }
        }
    }

    private void addLandmarkCandidate(ServerPlayer player, Vec3 eyePos, Vec3 horizForward, BlockPos p,
                                      BlockState s, String category, boolean checkOcclusion,
                                      List<LandmarkHit> out) {
        Vec3 bCenter = Vec3.atCenterOf(p);
        Vec3 toBlock = bCenter.subtract(eyePos);
        double dist = Math.round(toBlock.length() * 10.0) / 10.0;
        if (dist < 1.5 || dist > LANDMARK_SCAN_RADIUS + 8) return;

        Vec3 horizTo = new Vec3(toBlock.x, 0, toBlock.z);
        boolean inFov = horizTo.lengthSqr() > 1e-4 && horizTo.normalize().dot(horizForward) >= 0.35;
        if (checkOcclusion && inFov && dist <= 20.0) {
            BlockHitResult hit = player.level().clip(new ClipContext(eyePos, bCenter,
                    ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, player));
            if (hit.getType() == HitResult.Type.BLOCK && !hit.getBlockPos().equals(p)) {
                BlockState blocker = player.level().getBlockState(hit.getBlockPos());
                if (!blocker.is(BlockTags.LEAVES)) {
                    inFov = false;
                }
            }
        }

        float bearingYaw = (float) Math.toDegrees(Math.atan2(-toBlock.x, toBlock.z));
        String dir = compassDirection(bearingYaw);
        String id = BuiltInRegistries.BLOCK.getKey(s.getBlock()).toString();
        out.add(new LandmarkHit(p.immutable(), id, category, dist, dir, inFov));
    }

    private JsonArray buildHorizonSectors(ServerLevel level, BlockPos origin) {
        JsonArray sectors = new JsonArray();
        String[] names = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
        double[][] dirs = {
            {0, -1}, {0.7071, -0.7071}, {1, 0}, {0.7071, 0.7071},
            {0, 1}, {-0.7071, 0.7071}, {-1, 0}, {-0.7071, -0.7071}
        };

        int baseY = origin.getY();
        for (int i = 0; i < 8; i++) {
            int waterCount = 0;
            int treeCount = 0;
            int maxStepDelta = 0;
            int endElevationDelta = 0;
            int loadedSamples = 0;
            int prevY = baseY;
            BlockPos farthestSafe = null;

            for (int dist : new int[]{16, 32, 48, 64}) {
                int sx = origin.getX() + (int) Math.round(dirs[i][0] * dist);
                int sz = origin.getZ() + (int) Math.round(dirs[i][1] * dist);
                BlockPos col = new BlockPos(sx, baseY, sz);
                if (!level.isLoaded(col)) break;
                loadedSamples++;

                int groundY = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, sx, sz);
                int topY = level.getHeight(Heightmap.Types.WORLD_SURFACE, sx, sz);
                BlockState surface = level.getBlockState(new BlockPos(sx, groundY - 1, sz));
                BlockState atGround = level.getBlockState(new BlockPos(sx, groundY, sz));

                if (surface.is(Blocks.WATER) || atGround.is(Blocks.WATER)) {
                    waterCount++;
                } else if (!surface.is(Blocks.LAVA) && !atGround.is(Blocks.LAVA)) {
                    farthestSafe = new BlockPos(sx, groundY, sz);
                }
                if (topY > groundY + 2) {
                    treeCount++;
                }
                maxStepDelta = Math.max(maxStepDelta, Math.abs(groundY - prevY));
                endElevationDelta = groundY - baseY;
                prevY = groundY;
            }

            String terrainType;
            if (loadedSamples == 0) terrainType = "unloaded";
            else if (waterCount >= 2) terrainType = "water";
            else if (maxStepDelta >= 10) terrainType = "cliff";
            else if (treeCount >= 2) terrainType = "forest";
            else if (Math.abs(endElevationDelta) >= 8) terrainType = "hilly";
            else terrainType = "open";

            JsonObject sec = new JsonObject();
            sec.addProperty("dir", names[i]);
            sec.addProperty("terrain", terrainType);
            sec.addProperty("elev_delta", endElevationDelta);
            sec.addProperty("passable", loadedSamples > 0 && waterCount < 3 && maxStepDelta < 12);
            if (farthestSafe != null) {
                JsonArray target = new JsonArray();
                target.add(farthestSafe.getX());
                target.add(farthestSafe.getY());
                target.add(farthestSafe.getZ());
                sec.add("sample_at", target);
            }
            sectors.add(sec);
        }
        return sectors;
    }

    private String[] generateAsciiRetina(ServerPlayer player, Vec3 eyePos, float yaw, float pitch,
                                         List<EntitySight> seenEntities) {
        // 7 columns x 5 rows covering 110° wide horizontal x 60° vertical FOV out to 96 blocks (6 chunks)
        double[] colAngles = {-54, -36, -18, 0, 18, 36, 54};
        double[] rowAngles = {-28, -14, 0, 14, 28};
        double centerPitch = Mth.clamp(pitch, -25.0f, 20.0f);
        String[] rows = new String[5];

        for (int r = 0; r < 5; r++) {
            StringBuilder sb = new StringBuilder("[");
            for (int c = 0; c < 7; c++) {
                double checkYaw = yaw + colAngles[c];
                double checkPitch = centerPitch + rowAngles[r];

                float fYaw = (float) Math.toRadians(checkYaw);
                float fPitch = (float) Math.toRadians(checkPitch);
                Vec3 dir = new Vec3(
                        -Math.sin(fYaw) * Math.cos(fPitch),
                        -Math.sin(fPitch),
                        Math.cos(fYaw) * Math.cos(fPitch)
                );

                BlockHitResult hit = clipLoaded(player, eyePos, dir, RETINA_MAX_DISTANCE,
                        ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY);

                char symbol = '.';
                if (hit.getType() == HitResult.Type.BLOCK) {
                    double hitDist = eyePos.distanceTo(hit.getLocation());
                    BlockState bs = player.level().getBlockState(hit.getBlockPos());
                    if (bs.is(BlockTags.LOGS)) symbol = 'T';
                    else if (bs.is(BlockTags.LEAVES)) symbol = hitDist > 32.0 ? 't' : 'L';
                    else if (bs.is(Blocks.CRAFTING_TABLE) || bs.is(Blocks.FURNACE) || bs.is(Blocks.CHEST)) symbol = 'P';
                    else if (bs.is(Blocks.COAL_ORE) || bs.is(Blocks.IRON_ORE) || bs.is(Blocks.COPPER_ORE)) symbol = '*';
                    else if (bs.is(Blocks.LAVA)) symbol = '!';
                    else if (bs.is(Blocks.WATER)) symbol = '~';
                    else symbol = hitDist > 40.0 ? ':' : '#';
                }

                for (EntitySight es : seenEntities) {
                    Vec3 toE = es.entity.getEyePosition().subtract(eyePos);
                    if (toE.length() <= ENTITY_FOV_DISTANCE && toE.normalize().dot(dir) > 0.96) {
                        symbol = es.entity instanceof Monster ? 'Z' : (es.entity instanceof Player ? '@' : 'S');
                        break;
                    }
                }

                if (r == 2 && c == 3 && symbol == '.') {
                    symbol = '+';
                }

                sb.append(' ').append(symbol);
            }
            sb.append(" ]");
            rows[r] = sb.toString();
        }
        return rows;
    }

    private String renderVisualSnapshot(ServerPlayer player, Vec3 eyePos, float yaw, float pitch,
                                        List<EntitySight> seenEntities) {
        try {
            int width = 64;
            int height = 64;
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = image.createGraphics();

            boolean isDay = (player.level().getGameTime() % 24000) < 13000;
            Color skyColor = isDay ? new Color(115, 170, 255) : new Color(15, 20, 45);
            g.setColor(skyColor);
            g.fillRect(0, 0, width, height);

            // 32x32 multi-chunk raycast grid (2x2px cells) out to 96 blocks (6 chunks) with atmospheric depth fog
            int gridW = 32;
            int gridH = 32;
            int cellW = width / gridW;
            int cellH = height / gridH;
            double centerPitch = Mth.clamp(pitch, -30.0f, 25.0f);

            for (int gx = 0; gx < gridW; gx++) {
                double colAngle = ((double) gx / gridW - 0.5) * 100.0;
                double rayYaw = yaw + colAngle;
                float fYaw = (float) Math.toRadians(rayYaw);

                for (int gy = 0; gy < gridH; gy++) {
                    double rowAngle = ((double) gy / gridH - 0.5) * 70.0;
                    double rayPitch = centerPitch + rowAngle;
                    float fPitch = (float) Math.toRadians(rayPitch);

                    Vec3 dir = new Vec3(
                            -Math.sin(fYaw) * Math.cos(fPitch),
                            -Math.sin(fPitch),
                            Math.cos(fYaw) * Math.cos(fPitch)
                    );

                    BlockHitResult hit = clipLoaded(player, eyePos, dir, SNAPSHOT_MAX_DISTANCE,
                            ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY);

                    if (hit.getType() == HitResult.Type.BLOCK) {
                        BlockState bs = player.level().getBlockState(hit.getBlockPos());
                        Color base;
                        if (bs.is(BlockTags.LOGS)) base = new Color(102, 68, 34);
                        else if (bs.is(BlockTags.LEAVES)) base = new Color(38, 139, 38);
                        else if (bs.is(Blocks.GRASS_BLOCK)) base = new Color(85, 155, 45);
                        else if (bs.is(Blocks.STONE) || bs.is(Blocks.COBBLESTONE)) base = new Color(130, 130, 135);
                        else if (bs.is(Blocks.COAL_ORE)) base = new Color(45, 45, 50);
                        else if (bs.is(Blocks.IRON_ORE)) base = new Color(185, 150, 125);
                        else if (bs.is(Blocks.SAND) || bs.is(Blocks.SANDSTONE)) base = new Color(218, 205, 150);
                        else if (bs.is(Blocks.SNOW) || bs.is(Blocks.SNOW_BLOCK)) base = new Color(240, 245, 250);
                        else if (bs.is(Blocks.DIRT) || bs.is(Blocks.COARSE_DIRT)) base = new Color(134, 96, 67);
                        else if (bs.is(Blocks.WATER)) base = new Color(40, 85, 205);
                        else if (bs.is(Blocks.LAVA)) base = new Color(255, 100, 0);
                        else if (bs.is(Blocks.CRAFTING_TABLE) || bs.is(BlockTags.PLANKS)) base = new Color(175, 125, 65);
                        else base = new Color(105, 105, 105);

                        double dist = eyePos.distanceTo(hit.getLocation());
                        double fog = Mth.clamp((dist - 16.0) / (SNAPSHOT_MAX_DISTANCE * 1.15), 0.0, 0.62);
                        int r = (int) Math.round(base.getRed() * (1.0 - fog) + skyColor.getRed() * fog);
                        int gr = (int) Math.round(base.getGreen() * (1.0 - fog) + skyColor.getGreen() * fog);
                        int b = (int) Math.round(base.getBlue() * (1.0 - fog) + skyColor.getBlue() * fog);

                        g.setColor(new Color(Mth.clamp(r, 0, 255), Mth.clamp(gr, 0, 255), Mth.clamp(b, 0, 255)));
                        g.fillRect(gx * cellW, gy * cellH, cellW, cellH);
                    }
                }
            }

            // Project visible entities onto the 64x64 viewport
            for (EntitySight es : seenEntities.stream().limit(8).toList()) {
                Vec3 toE = es.entity.getEyePosition().subtract(eyePos);
                double eYaw = Mth.wrapDegrees(Math.toDegrees(Math.atan2(-toE.x, toE.z)) - yaw);
                double ePitch = -Math.toDegrees(Math.atan2(toE.y, Math.hypot(toE.x, toE.z))) - centerPitch;
                if (Math.abs(eYaw) <= 50.0 && Math.abs(ePitch) <= 35.0) {
                    int px = (int) Math.round((eYaw / 100.0 + 0.5) * width);
                    int py = (int) Math.round((ePitch / 70.0 + 0.5) * height);
                    int sz = es.distance < 16.0 ? 3 : 2;
                    g.setColor(es.entity instanceof Monster ? new Color(235, 45, 45) : new Color(250, 230, 80));
                    g.fillRect(Mth.clamp(px - 1, 0, width - sz), Mth.clamp(py - 1, 0, height - sz), sz, sz);
                }
            }

            // Draw center crosshair
            g.setColor(Color.WHITE);
            g.drawLine(width / 2 - 2, height / 2, width / 2 + 2, height / 2);
            g.drawLine(width / 2, height / 2 - 2, width / 2, height / 2 + 2);
            g.dispose();

            pendingSnapshot = CompletableFuture.supplyAsync(() -> {
                try {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(image, "png", baos);
                    String b64 = Base64.getEncoder().encodeToString(baos.toByteArray());
                    lastSnapshotBase64 = b64;
                    return b64;
                } catch (IOException | RuntimeException e) {
                    return null;
                }
            }, ENCODER_POOL);

            return lastSnapshotBase64;
        } catch (RuntimeException e) {
            return null;
        }
    }

    private String buildVisualSummary(float yaw, float pitch, JsonObject lookingAt,
                                      List<EntitySight> seenEntities, JsonArray visibleBlocks,
                                      List<LandmarkHit> landmarks, int skyLight) {
        StringBuilder sb = new StringBuilder();
        sb.append("Looking ").append(compassDirection(yaw));
        if (pitch < -25) sb.append(" upward");
        else if (pitch > 25) sb.append(" downward");
        else sb.append(" across horizon");

        if (lookingAt.has("id") && !"minecraft:air".equals(lookingAt.get("id").getAsString())) {
            String id = lookingAt.get("id").getAsString().replaceAll("^.*:", "");
            double dist = lookingAt.has("distance") ? lookingAt.get("distance").getAsDouble() : 0.0;
            sb.append(". Crosshair on ").append(id).append(" (").append(dist).append("m)");
        }

        if (!seenEntities.isEmpty()) {
            sb.append(". Entities (up to 4 chunks): ");
            List<String> entStrings = new ArrayList<>();
            for (EntitySight es : seenEntities) {
                String type = BuiltInRegistries.ENTITY_TYPE.getKey(es.entity.getType()).toString().replaceAll("^.*:", "");
                entStrings.add(type + " " + (Math.round(es.distance * 10) / 10.0) + "m " + es.horizontalBearing);
                if (entStrings.size() >= 4) break;
            }
            sb.append(String.join(", ", entStrings));
        }

        if (!landmarks.isEmpty()) {
            sb.append(". Landmarks (up to 5 chunks): ");
            List<String> lmStrings = new ArrayList<>();
            for (LandmarkHit lm : landmarks) {
                String shortId = lm.id().replaceAll("^.*:", "");
                lmStrings.add(shortId + " " + Math.round(lm.distance()) + "m " + lm.direction()
                        + " [" + lm.pos().getX() + "," + lm.pos().getY() + "," + lm.pos().getZ() + "]");
                if (lmStrings.size() >= 4) break;
            }
            sb.append(String.join("; ", lmStrings));
        } else if (visibleBlocks.size() > 0) {
            sb.append(". Visible features: ").append(visibleBlocks.size()).append(" key blocks in FOV");
        }
        sb.append(". Sunlight: ").append(skyLight).append("/15.");
        return sb.toString();
    }

    static String compassDirection(float yaw) {
        float rot = (yaw % 360 + 360) % 360;
        if (rot >= 337.5 || rot < 22.5) return "South";
        if (rot >= 22.5 && rot < 67.5) return "South-West";
        if (rot >= 67.5 && rot < 112.5) return "West";
        if (rot >= 112.5 && rot < 157.5) return "North-West";
        if (rot >= 157.5 && rot < 202.5) return "North";
        if (rot >= 202.5 && rot < 247.5) return "North-East";
        if (rot >= 247.5 && rot < 292.5) return "East";
        return "South-East";
    }
}
