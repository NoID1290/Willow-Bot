import unittest

import modelprofile

LLAMA3 = {
    "capabilities": ["completion"],
    "details": {"family": "llama", "parameter_size": "8.0B", "quantization_level": "Q4_0"},
    "parameters": 'num_keep                       24\nstop                           "<|eot_id|>"',
    "model_info": {"general.architecture": "llama", "llama.context_length": 8192, "llama.block_count": 32,
                   "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
                   "llama.embedding_length": 4096},
}
GEMMA4 = {
    "capabilities": ["completion", "vision", "tools", "thinking"],
    "details": {"family": "gemma4", "parameter_size": "11.9B", "quantization_level": "Q4_0"},
    "parameters": "temperature                    1\ntop_k                          64\ntop_p                          0.95",
    "model_info": {"general.architecture": "gemma4", "gemma4.context_length": 262144, "gemma4.block_count": 48,
                   "gemma4.attention.head_count": 16, "gemma4.attention.key_length": 512,
                   "gemma4.attention.value_length": 512, "gemma4.attention.sliding_window": 1024,
                   "gemma4.embedding_length": 3840},
}
QWEN_HYBRID = {
    "capabilities": ["completion", "thinking"],
    "details": {"family": "qwen35"},
    "parameters": "presence_penalty 1.5\ntemperature 1\ntop_k 20",
    "model_info": {"general.architecture": "qwen35", "qwen35.context_length": 262144, "qwen35.block_count": 32,
                   "qwen35.attention.head_count": 16, "qwen35.attention.key_length": 256,
                   "qwen35.full_attention_interval": 4},
}
MOE = {
    "capabilities": ["completion"],
    "model_info": {"general.architecture": "qwen3moe", "qwen3moe.context_length": 40960,
                   "qwen3moe.block_count": 48, "qwen3moe.attention.head_count": 32,
                   "qwen3moe.attention.head_count_kv": 4, "qwen3moe.embedding_length": 2048,
                   "qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8},
}
LFM3_1_7B = {
    "capabilities": ["completion", "tools"],
    "details": {"family": "lfm", "parameter_size": "1.7B", "quantization_level": "Q4_K_M"},
    "parameters": "top_p 0.9",
    "model_info": {
        "general.architecture": "lfm",
        "lfm.context_length": 131072,
        "lfm.block_count": 28,
        "lfm.attention.head_count": 16,
        "lfm.attention.head_count_kv": 4,
        "lfm.embedding_length": 2048,
        "lfm.attention.sliding_window": 512,
    },
}
LFM3_VL = {
    "capabilities": ["completion", "vision"],
    "details": {"family": "lfm", "parameter_size": "2.0B", "quantization_level": "Q4_K_M"},
    "model_info": {
        "general.architecture": "lfm3",
        "lfm3.context_length": 131072,
        "lfm3.block_count": 28,
        "lfm3.attention.head_count": 16,
        "lfm3.attention.head_count_kv": 4,
        "lfm3.embedding_length": 2048,
    },
}
LFM3_OMNI = {
    "capabilities": ["completion", "vision", "audio", "tools"],
    "details": {"family": "lfm", "parameter_size": "2.2B", "quantization_level": "Q4_K_M"},
    "model_info": {
        "general.architecture": "liquid",
        "liquid.context_length": 262144,
        "liquid.block_count": 32,
        "liquid.attention.head_count": 16,
        "liquid.attention.head_count_kv": 4,
        "liquid.embedding_length": 2048,
    },
}


class ModelProfileTests(unittest.TestCase):
    def test_small_context_non_reasoning_model_fits_its_trained_window(self):
        profile = modelprofile.build("llama3:8b", LLAMA3)
        self.assertEqual(profile["context"], 8192)
        self.assertFalse(profile["thinking"])
        self.assertIsNone(profile["think_value"], "Never send think to a model without the capability")
        self.assertEqual(profile["kv_cache_mib"], 1024, "Matches llama.cpp's reported 1024 MiB cache")
        self.assertLessEqual(profile["actor_tokens"], 1024)
        tokens = profile["prompt_bytes"] / modelprofile.BYTES_PER_TOKEN
        self.assertLessEqual(tokens + profile["plan_tokens"] + modelprofile.TEMPLATE_TOKENS, profile["context"])
        self.assertEqual(profile["options"]["temperature"], 0.6)

    def test_long_context_reasoning_model_uses_its_own_sampling(self):
        profile = modelprofile.build("gemma4:12b-it-qat", GEMMA4)
        self.assertEqual(profile["context"], 32768)
        self.assertIs(profile["think_value"], True)
        self.assertNotIn("temperature", profile["options"], "Model-tuned defaults are not overridden")
        self.assertEqual(profile["model_defaults"]["top_k"], 64)

    def test_hybrid_attention_and_moe_are_detected(self):
        hybrid = modelprofile.build("qwen3.5:4b", QWEN_HYBRID)
        self.assertEqual(hybrid["kv_bytes_per_token"], 8 * 16 * 512 * 2)
        moe = modelprofile.build("qwen3:30b-a3b", MOE)
        self.assertEqual((moe["moe_experts"], moe["moe_active_experts"]), (128, 8))
        self.assertEqual(moe["context"], 32768)

    def test_overrides_are_validated_and_clamped_to_the_model(self):
        profile = modelprofile.build("llama3:8b", LLAMA3, {"contextLength": 65536, "actorTokens": 512,
                                                           "temperature": 0.2, "think": "on"})
        self.assertEqual(profile["context"], 8192, "Cannot exceed the trained context")
        self.assertEqual(profile["actor_tokens"], 512)
        self.assertEqual(profile["options"]["temperature"], 0.2)
        self.assertIsNone(profile["think_value"])
        off = modelprofile.build("gemma4:12b-it-qat", GEMMA4, {"think": "off", "contextLength": 16384})
        self.assertIs(off["think_value"], False)
        self.assertEqual((off["context"], off["actor_tokens"]), (16384, 1024))
        for bad in ({"contextLength": 100}, {"temperature": 3}, {"think": "maybe"}, {"shell": 1},
                    {"actorTokens": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                modelprofile.build("llama3:8b", LLAMA3, bad)

    def test_unusable_models_are_refused(self):
        with self.assertRaises(ValueError):
            modelprofile.build("embed", {"capabilities": ["embedding"]})
        with self.assertRaises(ValueError):
            modelprofile.build("cloud", dict(GEMMA4, remote_host="https://example.invalid"))
        gptoss = modelprofile.build("gpt-oss:20b", {"capabilities": ["completion", "thinking"],
                                                    "model_info": {"general.architecture": "gptoss",
                                                                   "gptoss.context_length": 131072}})
        self.assertEqual(gptoss["think_value"], "medium", "gpt-oss takes a reasoning level, not a boolean")

    def test_negative_modelfile_options_are_sanitized(self):
        show = {
            "capabilities": ["completion"],
            "details": {"family": "granite"},
            "parameters": "repeat_last_n                  -1\ntop_k                          -1\n",
            "model_info": {"general.architecture": "granite", "granite.context_length": 8192},
        }
        profile = modelprofile.build("mineslayer", show)
        self.assertEqual(profile["options"]["repeat_last_n"], 64, "Negative repeat_last_n must be clamped to 64")
        self.assertEqual(profile["options"]["top_k"], 0, "Negative top_k must be clamped to 0")

    def test_lfm3_models_configured_properly(self):
        profile = modelprofile.build("lfm3:1.7b", LFM3_1_7B)
        self.assertTrue(profile["is_lfm"])
        self.assertEqual(profile["options"]["temperature"], 0.2, "LFM defaults to 0.2 temperature for precise tool calling")
        self.assertEqual(profile["context"], 16384, "Caps trained context at 16k auto-context default for fast edge startup")
        self.assertEqual(profile["multimodal_capabilities"]["text"], True)

    def test_lfm3_multimodal_and_omni_support(self):
        vl_profile = modelprofile.build("lfm3-vl-2b", LFM3_VL)
        self.assertTrue(vl_profile["is_lfm"])
        self.assertTrue(vl_profile["multimodal_capabilities"]["vision"])

        omni_profile = modelprofile.build("lfm3:omni", LFM3_OMNI)
        self.assertTrue(omni_profile["is_lfm"])
        self.assertTrue(omni_profile["multimodal_capabilities"]["vision"])
        self.assertTrue(omni_profile["multimodal_capabilities"]["audio"])
        self.assertTrue(omni_profile["multimodal_capabilities"]["omni"])

    def test_hardware_detection_and_presets(self):
        hw = modelprofile.detect_hardware()
        self.assertGreaterEqual(hw["cpu_cores"], 1)
        self.assertGreaterEqual(hw["num_thread"], 2)
        self.assertGreaterEqual(hw["vram_total_mib"], 1024)
        self.assertIn(hw["preset"], modelprofile.PRESETS)
        self.assertEqual(modelprofile.select_preset(hw), hw["preset"])

        low_hw = {"preset": "low_vram", "vram_total_mib": 8192, "vram_used_mib": 1200,
                  "model_weights_mib": 4300, "cpu_cores": 6, "num_thread": 4}
        low_prof_f16 = modelprofile.build("qwen3.5:4b", QWEN_HYBRID, hardware=low_hw)
        self.assertEqual(low_prof_f16["context"], 4096, "Without q8_0 active, 1024 MiB f16 KV cache Halves to 4096 to fit 564 MiB budget")
        self.assertEqual(low_prof_f16["planner_mode"], "disabled")
        self.assertEqual(low_prof_f16["options"]["num_thread"], 4)

        low_prof_q8 = modelprofile.build("qwen3.5:4b", QWEN_HYBRID, hardware=dict(low_hw, kv_cache_type="q8_0"))
        self.assertEqual(low_prof_q8["context"], 8192, "With q8_0 KV cache active, 512 MiB fits inside 564 MiB budget")

        high_hw = {"preset": "high_vram", "vram_total_mib": 24576, "vram_used_mib": 2000,
                   "model_weights_mib": 6823, "cpu_cores": 12, "num_thread": 8}
        high_prof = modelprofile.build("gemma4:12b-it-qat", GEMMA4, hardware=high_hw)
        self.assertEqual(high_prof["context"], 32768)
        self.assertEqual(high_prof["planner_mode"], "idle_only")

    def test_rtx4070ti_5800x3d_vram_budget_and_measured_formulas(self):
        hw = {
            "preset": "rtx4070ti_5800x3d",
            "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
            "gpu_source": "nvml",
            "vram_total_mib": 12282,
            "vram_used_mib": 1315,
            "model_weights_mib": 6823,
            "ram_total_mib": 64205,
            "cpu_model": "AMD Ryzen 7 5800X3D 8-Core Processor",
            "cpu_cores": 8,
            "cpu_threads": 16,
            "num_thread": 6,
        }
        bench = {
            "prompt_tps": 960.3,
            "gen_tps": 47.5,
            "bytes_per_token": 4.08,
            "model_weights_mib": 6823,
            "ps_vram_mib": 7614,
            "source": "ollama_live_benchmark",
        }
        profile = modelprofile.build("gemma4:12b-it-qat", GEMMA4, hardware=hw, benchmark=bench)
        # 12282 - 3200 (MC/Iris) - 6823 (weights) - 1024 (margin) = 1235 MiB KV budget -> 16384 context
        self.assertEqual(profile["free_kv_budget_mib"], 1235)
        self.assertEqual(profile["context"], 16384)
        self.assertIs(profile["think_value"], False, "Auto-tuned actor disables CoT on 12B to meet 1-3s latency target")
        self.assertEqual(profile["options"]["temperature"], 0.2)
        self.assertEqual(profile["options"]["num_thread"], 6)
        self.assertEqual(profile["actor_tokens"], 448)
        # The strategist keeps chain-of-thought on a thinking model, so its budget must hold a reasoning trace.
        self.assertEqual(profile["plan_tokens"], 2048)
        self.assertIs(profile["roles"]["actor"]["think"], False)
        self.assertIs(profile["roles"]["strategist"]["think"], True)
        self.assertEqual(profile["roles"]["strategist"]["num_predict"], 2048)
        self.assertEqual(
            profile["prompt_tokens"],
            profile["context"] - profile["json_output_tokens"] - profile["thinking_reserve_tokens"] - profile["safety_margin_tokens"],
        )
        self.assertEqual(profile["prompt_bytes"], int(profile["prompt_tokens"] * 4.08))
        self.assertEqual(profile["timeouts"]["actor_timeout"], 60)
        self.assertEqual(profile["timeouts"]["plan_timeout"], 90)
        self.assertEqual(profile["sources"]["gameplay_limits"], "hardware_and_latency_derived")
        self.assertGreaterEqual(profile["gameplay_limits"]["lessons_cap"], 2000)
        self.assertGreaterEqual(profile["gameplay_limits"]["decisions_cap"], 20000)

        log_line = modelprofile.format_startup_log(dict(profile, hardware=hw, benchmark=bench))
        self.assertIn("[AUTOTUNE] preset=rtx4070ti_5800x3d", log_line)
        self.assertIn("NVIDIA GeForce RTX 4070 Ti", log_line)
        self.assertIn("num_thread=6", log_line)
        self.assertIn("ctx=16384", log_line)
        self.assertIn("limits=(", log_line)
        self.assertEqual(profile["planner_mode"], "idle_only")
        self.assertTrue(profile["vision_enabled"])

        # Regression test: VRAM is sampled before the benchmark together with /api/ps. A model still
        # resident from the previous session (ps_vram_mib) must not be counted as Minecraft's share, and a
        # benchmark that timed out mid-load must not inflate it either (bench ps is ignored when hw has one).
        lfm_hw = dict(hw, vram_used_mib=6768, model_weights_mib=4375, ps_vram_mib=4643)
        lfm_bench = dict(bench, model_weights_mib=4375, ps_vram_mib=0, source="fallback(timed out)")
        lfm_prof = modelprofile.build("lfm3-all:latest", GEMMA4, hardware=lfm_hw, benchmark=lfm_bench)
        self.assertEqual(lfm_prof["minecraft_reserve_mib"], 3200)
        self.assertEqual(lfm_prof["planner_mode"], "idle_only")
        self.assertTrue(lfm_prof["vision_enabled"])
        # Without a ps sample the whole pre-load usage is non-model usage (Minecraft + shaders).
        heavy = modelprofile.build("lfm3-all:latest", GEMMA4, hardware=dict(hw, vram_used_mib=4100, ps_vram_mib=0),
                                   benchmark=bench)
        self.assertEqual(heavy["minecraft_reserve_mib"], 4100)

    def test_role_profile_derivation_and_budget_consistency(self):
        # Verify role-aware configuration and consistency between tokens, bytes, and num_ctx
        profile = modelprofile.build("gemma4:12b-it-qat", GEMMA4, {
            "strategistTemperature": 0.3,
            "roles": {
                "actor": {"num_predict": 1024, "think": "off"},
                "strategist": {"num_predict": 2048, "think": "on", "temperature": 0.25}
            }
        })
        self.assertIn("roles", profile)
        actor = profile["roles"]["actor"]
        strat = profile["roles"]["strategist"]

        # 1. Both roles MUST share the exact same num_ctx
        self.assertEqual(actor["num_ctx"], strat["num_ctx"])
        self.assertEqual(actor["num_ctx"], profile["context"])

        # 2. Actor has thinking OFF and output cap 1024
        self.assertFalse(actor["think"])
        self.assertIs(actor["think_value"], False)
        self.assertEqual(actor["num_predict"], 1024)

        # 3. Strategist has thinking ON and specified output cap and temperature
        self.assertTrue(strat["think"])
        self.assertEqual(strat["num_predict"], 2048)
        self.assertEqual(strat["temperature"], 0.25)

        # 4. Token/byte budgets are consistent
        bpt = profile["bytes_per_token"]
        self.assertEqual(actor["prompt_bytes"], int(min(modelprofile.PROMPT_BYTES_CAP, actor["prompt_tokens"] * bpt)))
        self.assertEqual(strat["prompt_bytes"], int(min(modelprofile.PROMPT_BYTES_CAP, strat["prompt_tokens"] * bpt)))
        self.assertLessEqual(actor["prompt_tokens"] + actor["num_predict"], actor["num_ctx"])
        self.assertLessEqual(strat["prompt_tokens"] + strat["num_predict"], strat["num_ctx"])


if __name__ == "__main__":
    unittest.main()
