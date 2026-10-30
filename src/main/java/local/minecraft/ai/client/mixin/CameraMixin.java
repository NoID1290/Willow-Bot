package local.minecraft.ai.client.mixin;

import local.minecraft.ai.client.CinematicCamera;
import net.minecraft.client.Camera;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Camera.class)
abstract class CameraMixin {
    @Shadow private Entity entity;
    @Shadow private boolean detached;
    @Shadow protected abstract void setPosition(Vec3 position);
    @Shadow protected abstract void setRotation(float yaw, float pitch);

    // This runs before Camera.update constructs projection and culling matrices.
    @Inject(method = "alignWithEntity", at = @At("TAIL"))
    private void willow$cinematic(float partialTick, CallbackInfo ci) {
        var pose = CinematicCamera.sample(entity, partialTick);
        if (pose == null) return;
        detached = true;
        setPosition(pose.position());
        setRotation(pose.yaw(), pose.pitch());
    }
}
