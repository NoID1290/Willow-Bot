package local.minecraft.ai.mixin;

import local.minecraft.ai.AcousticSensor;
import net.minecraft.core.Holder;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.sounds.SoundEvent;
import net.minecraft.sounds.SoundSource;
import net.minecraft.world.entity.Entity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerLevel.class)
abstract class ServerLevelSoundMixin {
    @Inject(method = "playSeededSound(Lnet/minecraft/world/entity/Entity;DDDLnet/minecraft/core/Holder;Lnet/minecraft/sounds/SoundSource;FFJ)V",
            at = @At("HEAD"))
    private void willow$recordSoundAtPos(Entity except, double x, double y, double z,
                                         Holder<SoundEvent> sound, SoundSource source,
                                         float volume, float pitch, long seed, CallbackInfo ci) {
        AcousticSensor.recordSound((ServerLevel) (Object) this, sound, source, x, y, z, volume, pitch);
    }

    @Inject(method = "playSeededSound(Lnet/minecraft/world/entity/Entity;Lnet/minecraft/world/entity/Entity;Lnet/minecraft/core/Holder;Lnet/minecraft/sounds/SoundSource;FFJ)V",
            at = @At("HEAD"))
    private void willow$recordSoundAtEntity(Entity except, Entity entity,
                                            Holder<SoundEvent> sound, SoundSource source,
                                            float volume, float pitch, long seed, CallbackInfo ci) {
        if (entity != null) {
            AcousticSensor.recordSound((ServerLevel) (Object) this, sound, source,
                    entity.getX(), entity.getY(), entity.getZ(), volume, pitch);
        }
    }
}
