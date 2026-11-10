"""Module 0: Hardware, Memory & 12GB VRAM Optimization Layer."""

from .llm_client import (
    DEFAULT_RESIDENT_MODEL,
    MAX_AI_VRAM_BUDGET_MIB,
    MAX_DYNAMIC_SUFFIX_TOKENS,
    NONE_TOKEN,
    RECOMMENDED_NUM_BATCH,
    RECOMMENDED_NUM_CTX,
    RECOMMENDED_NUM_PREDICT,
    CancellationToken,
    CancellationTokenSource,
    InferencePriority,
    LocalInferenceOptimizer,
    LocalLlmClient,
    OperationCanceledError,
    compute_safe_num_threads,
    estimate_tokens,
)
from .memory_pool import (
    MAX_COMPILED_SKILLS_IN_RAM,
    SNAPSHOT_MIN_INTERVAL_TICKS,
    CompiledSkillCache,
    CompiledSkillEntry,
    LiveReflexState,
    MemoryPoolManager,
)
from .telemetry_formatter import CompactTelemetryFormatter

__all__ = [
    "DEFAULT_RESIDENT_MODEL",
    "MAX_AI_VRAM_BUDGET_MIB",
    "MAX_DYNAMIC_SUFFIX_TOKENS",
    "MAX_COMPILED_SKILLS_IN_RAM",
    "SNAPSHOT_MIN_INTERVAL_TICKS",
    "NONE_TOKEN",
    "RECOMMENDED_NUM_BATCH",
    "RECOMMENDED_NUM_CTX",
    "RECOMMENDED_NUM_PREDICT",
    "CancellationToken",
    "CancellationTokenSource",
    "CompiledSkillCache",
    "CompiledSkillEntry",
    "CompactTelemetryFormatter",
    "InferencePriority",
    "LiveReflexState",
    "LocalInferenceOptimizer",
    "LocalLlmClient",
    "MemoryPoolManager",
    "OperationCanceledError",
    "compute_safe_num_threads",
    "estimate_tokens",
]
