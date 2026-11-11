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
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.imageio.ImageIO;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
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
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Visual perception engine for Willow.
 * Emits real-time visual streams: line-of-sight gaze targeting, field-of-view entities,
 * 2D ASCII retina depth map, and base64 visual frame snapshots for LFM3-VL / LFM3-Omni.
 */
public final class SightPerception {

    private static volatile String lastLookingAt = "Air";
    private static volatile String lastVisible = "Clear FOV";
    private static volatile boolean snapshotEnabled = true;

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

    /** The brain reports per decision whether it sends images; text-only profiles skip the 256-raycast PNG. */
    public static void setSnapshotEnabled(boolean enabled) {
        snapshotEnabled = enabled;
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

        int skyLight = player.level().getBrightness(LightLayer.SKY, player.blockPosition().above());
        int blockLight = player.level().getBrightness(LightLayer.BLOCK, player.blockPosition().above());
        boolean isDay = (player.level().getGameTime() % 24000) < 13000 && skyLight > 4;
        JsonObject lighting = new JsonObject();
        lighting.addProperty("sky", skyLight);
        lighting.addProperty("block", blockLight);
        lighting.addProperty("daylight", isDay);
        sight.add("lighting", lighting);

        // 1. Staring gaze / crosshair raycast (up to 12 blocks)
        JsonObject lookingAt = raycastGaze(player, eyePos, viewVec, 12.0);
        sight.add("looking_at", lookingAt);
        if (lookingAt.has("id")) {
            double d = lookingAt.has("distance") ? lookingAt.get("distance").getAsDouble() : 0.0;
            String name = lookingAt.get("id").getAsString().replaceAll("^.*:", "");
            lastLookingAt = name + String.format(" (%.1fm)", d);
        } else {
            lastLookingAt = "Air";
        }

        // 2. Visible entities in Field of View (within 24 blocks, ~90° horizontal, ~70° vertical cone)
        JsonArray visibleEntities = new JsonArray();
        List<EntitySight> seenEntities = scanVisibleEntities(player, eyePos, yaw, pitch, 24.0);
        for (EntitySight es : seenEntities) {
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
            visibleEntities.add(eo);
        }
        sight.add("visible_entities", visibleEntities);

        // 3. Significant visible blocks in FOV line-of-sight
        JsonArray visibleBlocks = scanVisibleKeyBlocks(player, eyePos, viewVec, 12.0);
        sight.add("visible_blocks", visibleBlocks);

        // 4. 7x5 ASCII Retina Depth Grid
        String[] retina = generateAsciiRetina(player, eyePos, yaw, pitch, seenEntities);
        JsonArray retinaArray = new JsonArray();
        for (String row : retina) {
            retinaArray.add(row);
        }
        sight.add("retina", retinaArray);

        // 5. Natural language visual summary
        String summary = buildVisualSummary(yaw, pitch, lookingAt, seenEntities, visibleBlocks, skyLight);
        sight.addProperty("visual_summary", summary);
        lastVisible = seenEntities.isEmpty() ? "Clear FOV"
                : seenEntities.size() + " entity/entities (" + seenEntities.get(0).entity.getType().toShortString() + ")";

        // 6. Base64 Visual Frame Snapshot (64x64 PNG) for multimodal LFM3-VL / Omni
        if (snapshotEnabled) {
            String frameBase64 = renderVisualSnapshot(player, eyePos, yaw, pitch, seenEntities);
            if (frameBase64 != null) {
                sight.addProperty("snapshot_png_base64", frameBase64);
            }
        }

        return sight;
    }

    private JsonObject raycastGaze(ServerPlayer player, Vec3 eyePos, Vec3 viewVec, double maxDist) {
        JsonObject target = new JsonObject();
        Vec3 end = eyePos.add(viewVec.scale(maxDist));

        // Block raycast
        BlockHitResult hit = player.level().clip(new ClipContext(eyePos, end,
                ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY, player));

        double blockDist = maxDist;
        if (hit.getType() != HitResult.Type.MISS) {
            blockDist = eyePos.distanceTo(hit.getLocation());
        }

        // Entity check in crosshairs
        AABB box = player.getBoundingBox().expandTowards(viewVec.scale(maxDist)).inflate(1.0);
        Entity closestEntity = null;
        double closestDist = blockDist;

        for (Entity e : player.level().getEntities(player, box,
                ent -> ent instanceof LivingEntity || ent instanceof ItemEntity)) {
            AABB eBox = e.getBoundingBox().inflate(0.3);
            var opt = eBox.clip(eyePos, end);
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
        AABB area = player.getBoundingBox().inflate(maxDist);
        List<Entity> list = player.level().getEntities(player, area,
                e -> (e instanceof LivingEntity || e instanceof ItemEntity) && !e.isSpectator());

        for (Entity e : list) {
            Vec3 toEntity = e.getEyePosition().subtract(eyePos);
            double dist = toEntity.length();
            if (dist > maxDist || dist < 0.1) continue;

            // Calculate angle relative to head yaw and pitch
            double entityYaw = Math.toDegrees(Math.atan2(-toEntity.x, toEntity.z));
            double relYaw = Mth.wrapDegrees(entityYaw - yaw);

            double entityPitch = -Math.toDegrees(Math.atan2(toEntity.y, Math.sqrt(toEntity.x * toEntity.x + toEntity.z * toEntity.z)));
            double relPitch = entityPitch - pitch;

            // FOV cone: ±50° horizontal, ±40° vertical
            if (Math.abs(relYaw) > 50.0 || Math.abs(relPitch) > 40.0) {
                continue;
            }

            // Occlusion raycast to ensure line of sight is clear
            BlockHitResult occ = player.level().clip(new ClipContext(eyePos, e.getEyePosition(),
                    ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, player));
            if (occ.getType() == HitResult.Type.BLOCK && eyePos.distanceTo(occ.getLocation()) < dist - 0.5) {
                continue; // Occluded by solid terrain
            }

            String hBearing = Math.abs(relYaw) < 15.0 ? "center" : (relYaw < 0 ? "left" : "right");
            String vElevation = Math.abs(relPitch) < 12.0 ? "level" : (relPitch < 0 ? "above" : "below");
            result.add(new EntitySight(e, dist, hBearing, vElevation));
        }

        result.sort(Comparator.comparingDouble(EntitySight::distance));
        return result;
    }

    private JsonArray scanVisibleKeyBlocks(ServerPlayer player, Vec3 eyePos, Vec3 viewVec, double maxDist) {
        JsonArray array = new JsonArray();
        BlockPos center = player.blockPosition();
        int r = (int) Math.min(maxDist, 10);

        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -3; dy <= 4; dy++) {
                for (int dz = -r; dz <= r; dz++) {
                    BlockPos p = center.offset(dx, dy, dz);
                    BlockState s = player.level().getBlockState(p);
                    if (s.isAir()) continue;

                    boolean isKey = s.is(BlockTags.LOGS) || s.is(Blocks.CRAFTING_TABLE) || s.is(Blocks.FURNACE)
                            || s.is(Blocks.COAL_ORE) || s.is(Blocks.IRON_ORE) || s.is(Blocks.COPPER_ORE)
                            || s.is(Blocks.CHEST) || s.is(Blocks.LAVA) || s.is(Blocks.WATER);
                    if (!isKey) continue;

                    Vec3 bCenter = Vec3.atCenterOf(p);
                    Vec3 toBlock = bCenter.subtract(eyePos);
                    double dist = toBlock.length();
                    if (dist > maxDist) continue;

                    // Dot product to check if block is in forward hemisphere
                    if (toBlock.normalize().dot(viewVec) < 0.5) continue;

                    // Occlusion check
                    BlockHitResult hit = player.level().clip(new ClipContext(eyePos, bCenter,
                            ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, player));
                    if (hit.getType() == HitResult.Type.BLOCK && !hit.getBlockPos().equals(p)) {
                        continue;
                    }

                    JsonObject obj = new JsonObject();
                    obj.addProperty("id", BuiltInRegistries.BLOCK.getKey(s.getBlock()).toString());
                    obj.addProperty("distance", Math.round(dist * 10) / 10.0);
                    JsonArray at = new JsonArray();
                    at.add(p.getX());
                    at.add(p.getY());
                    at.add(p.getZ());
                    obj.add("at", at);
                    array.add(obj);
                    if (array.size() >= 8) return array;
                }
            }
        }
        return array;
    }

    private String[] generateAsciiRetina(ServerPlayer player, Vec3 eyePos, float yaw, float pitch,
                                         List<EntitySight> seenEntities) {
        // 7 columns x 5 rows
        // Cols: -45, -30, -15, 0, +15, +30, +45 degrees rel yaw
        // Rows: +30 (up), +15, 0 (eye level), -15, -30 (down) degrees rel pitch
        double[] colAngles = {-45, -30, -15, 0, 15, 30, 45};
        double[] rowAngles = {-30, -15, 0, 15, 30}; // -30 is looking up in pitch convention
        String[] rows = new String[5];

        for (int r = 0; r < 5; r++) {
            StringBuilder sb = new StringBuilder("[");
            for (int c = 0; c < 7; c++) {
                double checkYaw = yaw + colAngles[c];
                double checkPitch = pitch + rowAngles[r];

                float fYaw = (float) Math.toRadians(checkYaw);
                float fPitch = (float) Math.toRadians(checkPitch);
                Vec3 dir = new Vec3(
                        -Math.sin(fYaw) * Math.cos(fPitch),
                        -Math.sin(fPitch),
                        Math.cos(fYaw) * Math.cos(fPitch)
                );

                Vec3 end = eyePos.add(dir.scale(16.0));
                BlockHitResult hit = player.level().clip(new ClipContext(eyePos, end,
                        ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY, player));

                char symbol = '.';
                if (hit.getType() == HitResult.Type.BLOCK) {
                    BlockState bs = player.level().getBlockState(hit.getBlockPos());
                    if (bs.is(BlockTags.LOGS)) symbol = 'T';
                    else if (bs.is(BlockTags.LEAVES)) symbol = 'L';
                    else if (bs.is(Blocks.CRAFTING_TABLE) || bs.is(Blocks.FURNACE) || bs.is(Blocks.CHEST)) symbol = 'P';
                    else if (bs.is(Blocks.COAL_ORE) || bs.is(Blocks.IRON_ORE)) symbol = '*';
                    else if (bs.is(Blocks.LAVA)) symbol = '!';
                    else if (bs.is(Blocks.WATER)) symbol = '~';
                    else symbol = '#';
                }

                // Check entities near this direction
                for (EntitySight es : seenEntities) {
                    Vec3 toE = es.entity.getEyePosition().subtract(eyePos);
                    if (toE.length() <= 16.0 && toE.normalize().dot(dir) > 0.95) {
                        symbol = es.entity instanceof Monster ? 'Z' : (es.entity instanceof Player ? '@' : 'S');
                        break;
                    }
                }

                // Center crosshairs marker
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

            // Background sky based on time
            boolean isDay = (player.level().getGameTime() % 24000) < 13000;
            Color skyColor = isDay ? new Color(110, 165, 255) : new Color(15, 20, 45);
            g.setColor(skyColor);
            g.fillRect(0, 0, width, height);

            // Fast 16x16 raycast grid rendered to 64x64 canvas
            int gridW = 16;
            int gridH = 16;
            int cellW = width / gridW;
            int cellH = height / gridH;

            for (int gx = 0; gx < gridW; gx++) {
                double colAngle = ((double) gx / gridW - 0.5) * 80.0;
                double rayYaw = yaw + colAngle;
                float fYaw = (float) Math.toRadians(rayYaw);

                for (int gy = 0; gy < gridH; gy++) {
                    double rowAngle = ((double) gy / gridH - 0.5) * 60.0;
                    double rayPitch = pitch + rowAngle;
                    float fPitch = (float) Math.toRadians(rayPitch);

                    Vec3 dir = new Vec3(
                            -Math.sin(fYaw) * Math.cos(fPitch),
                            -Math.sin(fPitch),
                            Math.cos(fYaw) * Math.cos(fPitch)
                    );

                    Vec3 end = eyePos.add(dir.scale(16.0));
                    BlockHitResult hit = player.level().clip(new ClipContext(eyePos, end,
                            ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY, player));

                    if (hit.getType() == HitResult.Type.BLOCK) {
                        BlockState bs = player.level().getBlockState(hit.getBlockPos());
                        Color blockColor;
                        if (bs.is(BlockTags.LOGS)) blockColor = new Color(102, 68, 34);
                        else if (bs.is(BlockTags.LEAVES)) blockColor = new Color(34, 139, 34);
                        else if (bs.is(Blocks.GRASS_BLOCK)) blockColor = new Color(85, 150, 40);
                        else if (bs.is(Blocks.STONE)) blockColor = new Color(128, 128, 128);
                        else if (bs.is(Blocks.DIRT)) blockColor = new Color(134, 96, 67);
                        else if (bs.is(Blocks.WATER)) blockColor = new Color(40, 80, 200);
                        else if (bs.is(Blocks.LAVA)) blockColor = new Color(255, 100, 0);
                        else if (bs.is(Blocks.CRAFTING_TABLE)) blockColor = new Color(175, 120, 60);
                        else blockColor = new Color(100, 100, 100);

                        g.setColor(blockColor);
                        g.fillRect(gx * cellW, gy * cellH, cellW, cellH);
                    }
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
                                      List<EntitySight> seenEntities, JsonArray visibleBlocks, int skyLight) {
        StringBuilder sb = new StringBuilder();
        sb.append("Looking ").append(compassDirection(yaw));
        if (pitch < -20) sb.append(" upward");
        else if (pitch > 20) sb.append(" downward");
        else sb.append(" straight ahead");

        if (lookingAt.has("id")) {
            String id = lookingAt.get("id").getAsString().replaceAll("^.*:", "");
            double dist = lookingAt.has("distance") ? lookingAt.get("distance").getAsDouble() : 0.0;
            sb.append(". Crosshair directly on ").append(id).append(" (").append(dist).append("m)");
        }

        if (!seenEntities.isEmpty()) {
            sb.append(". In sight: ");
            List<String> entStrings = new ArrayList<>();
            for (EntitySight es : seenEntities) {
                String type = BuiltInRegistries.ENTITY_TYPE.getKey(es.entity.getType()).toString().replaceAll("^.*:", "");
                entStrings.add(type + " " + (Math.round(es.distance * 10) / 10.0) + "m " + es.horizontalBearing);
                if (entStrings.size() >= 3) break;
            }
            sb.append(String.join(", ", entStrings));
        }

        if (visibleBlocks.size() > 0) {
            sb.append(". Visible features: ").append(visibleBlocks.size()).append(" key resources");
        }
        sb.append(". Sunlight: ").append(skyLight).append("/15.");
        return sb.toString();
    }

    private static String compassDirection(float yaw) {
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
