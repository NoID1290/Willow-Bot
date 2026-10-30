package local.minecraft.ai;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/** User overrides for one Ollama model; null fields mean "auto-detect from the model". */
public final class ModelSettings {
    private static final Gson GSON = new Gson();
    public Integer contextLength;
    public Integer actorTokens;
    public Integer planTokens;
    public Double temperature;
    public String think;
    public String vision;

    public ModelSettings copy() {
        var copy = new ModelSettings();
        copy.contextLength = contextLength;
        copy.actorTokens = actorTokens;
        copy.planTokens = planTokens;
        copy.temperature = temperature;
        copy.think = think;
        copy.vision = vision;
        return copy;
    }

    public boolean isEmpty() {
        return contextLength == null && actorTokens == null && planTokens == null && temperature == null
                && (think == null || think.equals("auto"))
                && (vision == null || vision.equals("auto"));
    }

    /** Mirrors brain/modelprofile.py validate_settings so bad values are refused before saving. */
    public void validate() {
        if (contextLength != null && (contextLength < 4096 || contextLength > 262144))
            throw new IllegalArgumentException("Context length must be 4096-262144");
        if (actorTokens != null && (actorTokens < 256 || actorTokens > 16384))
            throw new IllegalArgumentException("Actor output must be 256-16384 tokens");
        if (planTokens != null && (planTokens < 256 || planTokens > 16384))
            throw new IllegalArgumentException("Strategist output must be 256-16384 tokens");
        if (temperature != null && (temperature.isNaN() || temperature < 0 || temperature > 2))
            throw new IllegalArgumentException("Temperature must be 0-2");
        if (think != null && !think.equals("auto") && !think.equals("on") && !think.equals("off"))
            throw new IllegalArgumentException("Thinking must be auto, on or off");
        if (vision != null && !vision.equals("auto") && !vision.equals("on") && !vision.equals("off"))
            throw new IllegalArgumentException("Vision must be auto, on or off");
    }

    public String toJson() {
        JsonObject json = GSON.toJsonTree(this).getAsJsonObject();
        if ("auto".equals(think)) json.remove("think");
        if ("auto".equals(vision)) json.remove("vision");
        return json.toString();
    }
}
