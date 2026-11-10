# SYSTEM SPECIFICATION: WILLOW-BOT V2 — HIERARCHICAL RL, SURVIVAL CURRICULUM, VOYAGER SKILL SYNTHESIS & 12GB VRAM LOCAL LLM OPTIMIZATION ARCHITECTURE

You are a Principal AI Systems Architect, High-Performance Systems Engineer, and Senior Minecraft Engine Developer. Your task is to upgrade the **Willow-Bot** autonomous Minecraft agent codebase to incorporate a **Hierarchical Reinforcement Learning (HRL) Reward/Punishment Engine**, an **Automated Minecraft Advancement & Survival Curriculum (DAG)**, a **Voyager-inspired Dynamic Code Synthesis & RAG Skill Library**, and a **Strict Hardware Resource Optimization Layer engineered for Local LLMs on <= 12GB VRAM**.

Do not write placeholder code, `TODO` comments, or stubbed methods. Implement production-grade, zero-allocation/low-GC, thread-safe, fully typed code across all modules.

---

## 1. ARCHITECTURAL OVERVIEW, HARDWARE BUDGET & DUAL-LOOP RUNTIME

Willow-Bot V2 runs locally alongside the Minecraft client/server on consumer hardware with a **strict maximum of 12GB GPU VRAM** (e.g., RTX 3060 12GB, RTX 4070 12GB, ARC B580 12GB), **16GB–32GB System RAM**, and a shared **Multi-Core CPU**. 

Because Minecraft itself consumes ~2GB–3.5GB of VRAM (textures, shaders, chunk buffers) and ~4GB–6GB of System RAM, **the AI stack must never exceed 7.5GB–8.0GB of VRAM** and must never starve the Minecraft main render/server thread of CPU cycles.

### 1.1 Dual-Loop Runtime & Zero-LLM Fast Path
1. **Fast Reactive Reflex Loop (20 Hz / 50ms tick — Deterministic, 0% GPU/LLM Usage):**
   - Runs synchronously with the game tick on a dedicated lightweight CPU thread using pre-allocated memory buffers (zero GC pressure).
   - Monitors immediate survival hazards:
     - Hostile entity distance $< 6.0$ blocks (Creeper fuse state, Skeleton line-of-sight, Zombie swarm).
     - Environmental hazards (Lava within $2$ blocks, fire ticks $> 0$, suffocation in block, oxygen $< 60$, fall distance $> 3$).
     - Critical vitals (`Health <= 8.0` or `FoodLevel <= 6`).
   - Holds a master `CancellationTokenSource` that immediately preempts and pauses the Slow Cognitive Loop when a critical survival reflex is triggered (e.g., auto-equip shield and block, consume highest-saturation food, place water bucket to MLG/extinguish fire, or sprint/pillar away).
   - **While the Reflex Loop is actively fighting or fleeing, pause new LLM inference requests** so the GPU/CPU is 100% dedicated to frame rate and tick stability.

2. **Slow Cognitive, Curriculum & RL Loop (Asynchronous Background Worker):**
   - Orchestrated by `WillowBotOrchestrator` on a low-priority background thread (`ThreadPriority.BelowNormal`).
   - **90% Deterministic / 10% LLM Execution Target:** Only invoke the local LLM when a task has no verified high-Q skill in `SkillStore` or when a skill fails and requires mutation. Verification (`SelfVerificationCritic`) and Curriculum scheduling (`CurriculumManager`) must execute 100% deterministically in native code first, falling back to the LLM only on ambiguous failures.

---

## 2. MODULE 0: LOCAL LLM, 12GB VRAM, CPU & RAM OPTIMIZATION ENGINE (`hardware_opt/`)

Implement a dedicated hardware and inference optimization layer (`LocalInferenceOptimizer` & `MemoryPoolManager`) enforcing the following constraints:

### 2.1 12GB VRAM Budget & Single-Model Resident Policy
Never load multiple LLMs (e.g., separate Planner, Coder, and Critic models) into VRAM simultaneously—model swapping causes massive PCIe bus thrashing and multi-second freezes.
- **Recommended Local Model Spec:** Single resident 7B–8B parameter coding model quantized to **4-bit (`Q4_K_M` or `EXL2 4.0bpw`)** such as `Qwen2.5-Coder-7B-Instruct-Q4_K_M` (~4.7GB weights) or `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M`.
- **VRAM Allocation Breakdown (Max 8.0GB for AI + 4.0GB for Minecraft/OS = 12.0GB):**
  - Model Weights (`Q4_K_M`): `~4.6 GB`
  - KV Cache (`num_ctx = 4096`, **FP8 / `q8_0` quantized KV cache**): `~0.6 GB`
  - Compute Buffer & CUDA/Vulkan Overhead: `~0.8 GB`
  - **Total AI VRAM Footprint:** `~6.0 GB` (leaves a safe `6.0 GB` buffer for Minecraft and OS display manager).
- **Ollama / Llama.cpp / vLLM Request Payload Parameters:**
  Every inference call from `LocalLlmClient` MUST explicitly pass hardware-bounding options:
  ```json
  {
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "keep_alive": "30m",
    "stream": true,
    "options": {
      "num_ctx": 4096,
      "num_predict": 768,
      "temperature": 0.15,
      "top_p": 0.9,
      "num_batch": 256,
      "num_thread": 4,
      "use_mmap": true,
      "use_mlock": false,
      "flash_attn": true
    }
  }
  ```
  - Enforce `num_thread = Math.Max(2, Environment.ProcessorCount / 2 - 1)` so LLM prompt processing never starves Minecraft's Server/Render threads.
  - Enforce **Single-Flight Inference Semaphores (`SemaphoreSlim(1, 1)`)**: Strictly forbid concurrent/parallel LLM requests. Queue requests by priority (`SkillMutation > NewSkillSynthesis > TaskDecomposition`).

### 2.2 Prompt Prefix Caching & Ultra-Compact Context Compression (`num_ctx <= 4096`)
To achieve instant Time-To-First-Token (TTFT $< 250\text{ ms}$) on a 12GB GPU, structure all prompts so the **first 65% of tokens are 100% static across calls**, allowing llama.cpp / Ollama KV-cache prefix reuse:
1. **Static Prefix (Cached in GPU KV Cache — Never changes between episodes):**
   - System Role + `IWillowBotActions` compact TypeScript/C# header signatures + Output JSON Grammar.
2. **Dynamic Suffix (Appended at the end — Keep under 1,200 tokens):**
   - **Delta/Compact Telemetry Serializer (`CompactTelemetryFormatter`):** NEVER serialize raw 36-slot inventory arrays or full chunk grids into JSON. Serialize compact token-dense strings:
     `POS:124,64,-82(overworld,day,light:15) | HP:18/20 FOOD:16/20 ARMOR:6 | EQUIP:stone_pickaxe(98/131),shield | INV:cobblestone:24,oak_log:5,torch:12,raw_iron:4 | WORKSTATIONS:crafting_table@(125,64,-81) | THREATS:skeleton@11m`
   - **Dynamic API Pruning:** Only inject `SurvivalMechanicsRegistry` rules relevant to the current task's tier and target items (saves ~800 tokens per prompt).
   - **Grammar-Constrained Output:** Enforce structured JSON schema (`format: "json"` or GBNF grammar) and set `reasoning` max length to 2 sentences so the model spends tokens on executable code, not verbose essays.

### 2.3 Zero-VRAM CPU Embedding & Hybrid Lexical/Vector RAG
Do **NOT** load a heavy neural embedding model onto the GPU alongside the LLM.
- Implement a **Zero-VRAM Hybrid Retriever** in `SkillStore`:
  1. **BM25 + Precondition Tag Matcher (CPU, $< 0.2\text{ ms}$):** Exact keyword, item ID, and precondition/postcondition graph matching.
  2. **Lightweight CPU Embedding (Optional/ONNX Int8):** Use an ultra-lightweight CPU-only embedder (`all-MiniLM-L6-v2` quantized to Int8 ONNX, ~22MB RAM, 0MB VRAM) or Feature Hashing (MurmurHash3 $N$-gram vectors in a flat `float[]` array).
- Store all vectors in a flat memory-mapped binary file (`skills/embeddings.bin`) using `Span<float>` SIMD dot-product (`System.Numerics.Vector` / `numpy` BLAS) instead of running an external vector database server (no Docker, no Chroma/Milvus background RAM bloat).

### 2.4 System RAM & Garbage Collection (GC) Optimization
Minecraft mods and bots suffer severe tick lag spikes if they allocate megabytes of objects every tick:
- **Object Pooling & Structs:** `PerceivedEntity`, `BlockInfo`, and `Vector3` must be `readonly record struct` (value types) or pooled via `ArrayPool<T>.Shared`.
- **Snapshot Throttling:** Only allocate a full `BotTelemetrySnapshot` on cognitive episode boundaries ($S_t$ and $S_{t+1}$) or once every $20$ ticks ($1\text{ Hz}$). The $20\text{ Hz}$ Fast Reflex Loop must read directly from a pre-allocated, mutable `LiveReflexState` struct updated in-place.
- **Script Compilation Caching & Unloading:**
  - Compute a `SHA256` hash of every script's source code. Cache compiled delegates/bytecode in an LRU cache (max 64 compiled skills in RAM).
  - If using C# Roslyn (`CSharpScript`), compile into collectible `AssemblyLoadContext(isCollectible: true)` so failed/deprecated scripts are unloaded from RAM and never cause permanent memory leaks.
- **Bounded Ring Buffers:** Cap `ReflexionMemoryLog` to a fixed-size circular buffer of $5$ entries per task in RAM, flushing older analytics asynchronously to an append-only `.jsonl` file on disk.

---

## 3. MODULE 1: REINFORCEMENT LEARNING & REWARD/PUNISHMENT ENGINE (`rl_rewards/`)

Implement a **Hybrid Extrinsic/Intrinsic Reward Engine** combined with **Tabular Q-Learning over Abstracted Context States** and **Verbal Reinforcement (Reflexion)**.

### 3.1 Telemetry State Snapshot (`BotTelemetrySnapshot`)
Create an immutable, compact snapshot structure capturing the state of the bot at episode boundaries $t$ and $t+1$:
- `TimestampMs`: `long` (Unix epoch milliseconds)
- `WorldTick`: `long` and `TimeOfDay`: `long` ($0$ to $24000$, where $12542$–$23460$ is night)
- `Position`: `Vector3(double X, double Y, double Z)`
- `Dimension`: `"minecraft:overworld" | "minecraft:the_nether" | "minecraft:the_end"`
- `Biome`: `string` and `LightLevel`: `int` ($0$–$15$)
- `Vitals`:
  - `Health`: `float` ($0.0$–$20.0$), `MaxHealth`: `float` ($20.0$), `Absorption`: `float`
  - `FoodLevel`: `int` ($0$–$20$), `Saturation`: `float` ($0.0$–$20.0$)
  - `ArmorPoints`: `int` ($0$–$20$), `Oxygen`: `int` ($0$–$300$), `IsAlive`: `bool`
- `Inventory`:
  - `ItemCounts`: `Dictionary<string, int>` (Normalized item ID $\rightarrow$ total count across all 36 slots + offhand)
  - `EmptySlotsCount`: `int`
  - `Equipment`: `Dictionary<EquipmentSlot, EquippedItemState>` (`MainHand`, `OffHand`, `Head`, `Chest`, `Legs`, `Feet` $\rightarrow$ ItemID, CurrentDurability, MaxDurability)
- `Progression`:
  - `UnlockedAdvancements`: `HashSet<string>` (All completed vanilla advancement IDs)
  - `DiscoveredUniqueItems`: `HashSet<string>` (Cumulative set of item IDs ever held)
  - `CraftedUniqueItems`: `HashSet<string>` (Cumulative set of item IDs ever crafted)
- `EnvironmentPerception`:
  - `NearbyEntities`: `PerceivedEntity[]` (Pre-filtered to top 12 closest entities within $16$ blocks: EntityID, Type, Position, Distance, Health, IsHostile)
  - `NearbyWorkstations`: `Dictionary<string, Vector3>` (Closest `crafting_table`, `furnace`, `chest`, `bed`, `nether_portal` within $16$ blocks)
- `ExecutionTelemetry`:
  - `DurationMs`: `long`
  - `CompilationFailed`: `bool`
  - `RuntimeExceptions`: `List<string>` (Capped at 3)
  - `PathfindingTimeouts`: `int`
  - `BlocksMinedCount`: `int`
  - `ImproperToolMineAttempts`: `int`
  - `ToolsBroken`: `List<string>`

### 3.2 Exact Mathematical Reward Function ($R_{\text{total}}$)
Implement `RewardCalculator.CalculateEpisodeReward(BotTelemetrySnapshot s_prev, BotTelemetrySnapshot s_curr, CurriculumNode task, CriticEvaluation critic)` using the composite equation:

$$R_{\text{total}} = R_{\text{adv}} + R_{\text{task}} + R_{\text{shaping}} + R_{\text{intrinsic}} - P_{\text{vital}} - P_{\text{exec}} - P_{\text{time}}$$

#### 1. Extrinsic Advancement Rewards ($R_{\text{adv}}$)
Let $\Delta A = S_{t+1}.\text{UnlockedAdvancements} \setminus S_t.\text{UnlockedAdvancements}$. For each $a \in \Delta A$:
$$R_{\text{adv}} = \sum_{a \in \Delta A} \left( \text{BaseValue}(a) + \mathbb{I}(a == \text{task.AdvancementId}) \cdot 50.0 \right)$$
- **Tier 1 Advancements** (`story/root`, `story/mine_stone`, `husbandry/root`, `adventure/root`): $+50.0$
- **Tier 2 Advancements** (`story/upgrade_tools`, `story/smelt_iron`, `husbandry/plant_seed`, `adventure/sleep_in_bed`, `adventure/kill_a_mob`): $+100.0$
- **Tier 3 Advancements** (`story/obtain_armor`, `story/iron_tools`, `story/deflect_arrow`, `story/lava_bucket`): $+200.0$
- **Tier 4 Advancements** (`story/mine_diamond`, `story/shiny_gear`, `story/form_obsidian`, `story/enchant_item`): $+350.0$
- **Tier 5 Advancements** (`story/enter_the_nether`, `nether/obtain_blaze_rod`, `story/follow_ender_eye`, `story/enter_the_end`): $+600.0$

#### 2. Task Completion Reward ($R_{\text{task}}$)
$$R_{\text{task}} = \begin{cases} 
\text{task.BaseReward} & \text{if } \text{task.VerificationPredicate}(S_t, S_{t+1}) \land \text{critic.Success} \\
0.25 \cdot \text{task.BaseReward} \cdot \text{ProgressRatio}(S_t, S_{t+1}, \text{task}) & \text{if partial target item delta achieved} \\
0.0 & \text{otherwise}
\end{cases}$$

#### 3. Dense Inventory & Gear Shaping ($R_{\text{shaping}}$)
- **Target Resource Delta:** For each required item $(i, \text{reqCount})$ in `task.TargetItems`:
  $$\Delta n_i = \min\left(\text{reqCount}, S_{t+1}.\text{ItemCounts}[i]\right) - \min\left(\text{reqCount}, S_t.\text{ItemCounts}[i]\right)$$
  Add $+5.0 \times \max(0, \Delta n_i)$.
- **Armor & Tool Tier Upgrade:** $+15.0$ if $\text{ArmorPoints}(S_{t+1}) > \text{ArmorPoints}(S_t)$ or highest pickaxe/sword material tier in $S_{t+1}$ exceeds $S_t$ ($\text{Wood} < \text{Stone} < \text{Iron} < \text{Diamond} < \text{Netherite}$).
- **Sustenance Maintenance:** $+5.0$ if $S_t.\text{FoodLevel} \le 12$ and $S_{t+1}.\text{FoodLevel} \ge 18$.

#### 4. Intrinsic Curiosity & Novelty Bonus ($R_{\text{intrinsic}}$)
- $+3.0$ for each newly discovered item ID: $|S_{t+1}.\text{DiscoveredUniqueItems} \setminus S_t.\text{DiscoveredUniqueItems}| \times 3.0$.
- $+8.0$ for each newly crafted item ID: $|S_{t+1}.\text{CraftedUniqueItems} \setminus S_t.\text{CraftedUniqueItems}| \times 8.0$.

#### 5. Vitality & Survival Punishments ($P_{\text{vital}}$)
- **Death Penalty:** If $\neg S_{t+1}.\text{IsAlive}$ or $S_{t+1}.\text{Health} \le 0$:
  $$P_{\text{death}} = 250.0 + 10.0 \times (\text{lost valuable items count})$$
- **Damage Taken Penalty:**
  $$P_{\text{damage}} = 3.5 \times \max\left(0.0, S_t.\text{Health} - S_{t+1}.\text{Health}\right)$$
- **Hunger Neglect Penalty:**
  - $+15.0$ penalty if $S_{t+1}.\text{FoodLevel} == 0$ (taking starvation damage).
  - $+5.0$ penalty if $S_{t+1}.\text{FoodLevel} \le 6$ (cannot sprint/regenerate) AND the bot had edible food in $S_t.\text{Inventory}$ that it failed to consume.

#### 6. Execution, Tool & Pathing Punishments ($P_{\text{exec}}$)
- **Compilation / Syntax Failure:** $+15.0$ if `CompilationFailed == true`.
- **Unhandled Runtime Exception:** $+20.0 \times \min(3, \text{RuntimeExceptions.Count})$.
- **Valuable Tool Breakage:** $+35.0$ for each broken `iron_*`, `diamond_*`, or `netherite_*` tool in `ToolsBroken`.
- **Wrong Tool Usage (Zero-Drop Mining):** $+8.0 \times \text{ImproperToolMineAttempts}$ (e.g., breaking `iron_ore` with a `wooden_pickaxe` or hand).
- **Pathfinding / Stuck Penalty:** $+12.0$ if `PathfindingTimeouts > 0` or if task requires movement and $\lVert S_{t+1}.\text{Position} - S_t.\text{Position} \rVert_2 < 0.5$ over $> 10,000\text{ ms}$.

#### 7. Temporal Cost ($P_{\text{time}}$)
$$P_{\text{time}} = 0.08 \times \left(\frac{S_{t+1}.\text{DurationMs}}{1000.0}\right)$$

### 3.3 Contextual State Abstraction & Tabular Q-Learning (`SkillQTable`)
To allow RL to guide skill selection in microseconds with zero GPU/RAM overhead, implement `ContextStateHash` (packed into a 32-bit `uint` bitmask):
- Bitfields of context state $c$:
  - `ToolTier` (3 bits): `None(0) | Wood(1) | Stone(2) | Iron(3) | Diamond(4)`
  - `ArmorTier` (2 bits): `None(0) | PartialIron(1) | FullIron(2) | Diamond(3)`
  - `HealthBucket` (2 bits): `Critical <8 (0) | Moderate 8-15 (1) | Healthy 16-20 (2)`
  - `TimeBucket` (2 bits): `Day(0) | Dusk(1) | Night(2)`
  - `Dimension` (2 bits): `Overworld_Surface(0) | Overworld_Cave Y<45(1) | Nether(2) | End(3)`
  - `Flags` (3 bits): `HasCraftingTable | HasFurnace | HasShield`
- Update the Q-value for context $c$ and skill/task pair $(n, k)$ after every execution:
  $$Q(c, n, k) \leftarrow Q(c, n, k) + \alpha \left[ R_{\text{total}} + \gamma \max_{k'} Q(c', n', k') - Q(c, n, k) \right]$$
  where $\alpha = 0.25$ and $\gamma = 0.85$.
- **Fast-Path Skill Reuse (Bypassing LLM):** When `CurriculumManager` selects task $n$ in context $c$, if `SkillStore` contains a verified skill $k$ with $Q(c, n, k) \ge +10.0$ and satisfied preconditions, **execute $k$ directly in `ScriptSandbox` with 0 LLM calls!**

### 3.4 Compact Verbal Reinforcement Memory (`ReflexionMemoryLog`)
Whenever $R_{\text{total}} < 0$ or `critic.Success == false`, store a compact `EpisodeReflectionRecord` in the task's ring buffer (max 3 entries injected into prompt to save KV cache tokens):
```text
[FAIL #2 | R=-42.5 (Vital:-28, Exec:-16)] Cause: Mined iron_ore with wooden_pickaxe (0 drops) & took 8 HP from Cave Spider. Fix: Equip stone_pickaxe before mining iron_ore; retreat/shield if hostile < 6m.
```

---

## 4. MODULE 2: MINECRAFT BASE GAME MECHANICS & SURVIVAL CURRICULUM (`curriculum/`)

### 4.1 Deterministic Game Mechanics Knowledge Base (`SurvivalMechanicsRegistry`)
Implement a static, queryable knowledge base that filters and injects **only the rules matching the active task** into the LLM prompt:

1. **Block Harvest Level & Tool Matrix:**
   - `oak_log`, `birch_log`, `spruce_log`, `dirt`, `sand`, `gravel`: Harvestable by `hand` or any tool (Optimal: `axe` for wood, `shovel` for dirt/sand/gravel).
   - `stone`, `cobblestone`, `coal_ore`, `deepslate_coal_ore`: **Requires** `>= wooden_pickaxe`. Mining with hand yields 0 drops.
   - `iron_ore`, `deepslate_iron_ore`, `copper_ore`, `lapis_ore`: **Requires** `>= stone_pickaxe`. Drops `raw_iron` (must be smelted into `iron_ingot`).
   - `gold_ore`, `deepslate_gold_ore`, `diamond_ore`, `deepslate_diamond_ore`, `redstone_ore`, `emerald_ore`: **Requires** `>= iron_pickaxe`.
   - `obsidian`, `crying_obsidian`, `ancient_debris`: **Requires** `>= diamond_pickaxe`.

2. **Exact Crafting Dependency Tree (`RecipeKnowledgeGraph`):**
   - `planks` (x4) $\leftarrow$ $1\times$ `*_log` (2x2 inventory grid)
   - `stick` (x4) $\leftarrow$ $2\times$ `*_planks` (2x2 inventory grid)
   - `crafting_table` (x1) $\leftarrow$ $4\times$ `*_planks` (2x2 inventory grid)
   - `torch` (x4) $\leftarrow$ $1\times$ `coal` (or `charcoal`) $+ 1\times$ `stick` (2x2 inventory grid)
   - `wooden_pickaxe` $\leftarrow$ $3\times$ `*_planks` $+ 2\times$ `stick` (**Requires placed `crafting_table` within 4 blocks**)
   - `stone_pickaxe` $\leftarrow$ $3\times$ `cobblestone` (or `cobbled_deepslate`) $+ 2\times$ `stick` (**Requires `crafting_table`**)
   - `stone_axe` $\leftarrow$ $3\times$ `cobblestone` $+ 2\times$ `stick` (**Requires `crafting_table`**)
   - `stone_sword` $\leftarrow$ $2\times$ `cobblestone` $+ 1\times$ `stick` (**Requires `crafting_table`**)
   - `furnace` $\leftarrow$ $8\times$ `cobblestone` (**Requires `crafting_table`**)
   - `chest` $\leftarrow$ $8\times$ `*_planks` (**Requires `crafting_table`**)
   - `bed` $\leftarrow$ $3\times$ `*_wool` $+ 3\times$ `*_planks` (**Requires `crafting_table`**)
   - `shield` $\leftarrow$ $6\times$ `*_planks` $+ 1\times$ `iron_ingot` (**Requires `crafting_table`**)
   - `bucket` $\leftarrow$ $3\times$ `iron_ingot` (**Requires `crafting_table`**)
   - `iron_pickaxe` $\leftarrow$ $3\times$ `iron_ingot` $+ 2\times$ `stick` (**Requires `crafting_table`**)
   - `iron_sword` $\leftarrow$ $2\times$ `iron_ingot` $+ 1\times$ `stick` (**Requires `crafting_table`**)
   - `iron_helmet` ($5\times$), `iron_chestplate` ($8\times$), `iron_leggings` ($7\times$), `iron_boots` ($4\times$ `iron_ingot`) (**Requires `crafting_table`**)
   - `diamond_pickaxe` $\leftarrow$ $3\times$ `diamond` $+ 2\times$ `stick` (**Requires `crafting_table`**)
   - `diamond_sword` $\leftarrow$ $2\times$ `diamond` $+ 1\times$ `stick` (**Requires `crafting_table`**)
   - `flint_and_steel` $\leftarrow$ $1\times$ `iron_ingot` $+ 1\times$ `flint` (2x2 grid; `flint` drops from mining `gravel`).

3. **World Generation & Safe Mining Heuristics:**
   - **Never dig straight down** ($X, Z$ beneath the bot's feet) or straight up. Always mine a stepped staircase ($\Delta Y = -1, \Delta X = +1$) and check adjacent blocks for `lava` or `water`.
   - **Optimal Ore Y-Levels:** `coal_ore` $Y \in [48, 136]$; `iron_ore` $Y \in [-16, 48]$ (peak $Y = 16$); `diamond_ore` $Y \in [-59, -50]$ (mine at $Y = -53$ above lava aquifers).

### 4.2 Complete 20-Node Curriculum DAG (`curriculum/advancement_dag.json`)
Implement `CurriculumDAG` with the following 20 nodes across 5 progressive Tiers. Each node has a deterministic native verification predicate `Verify(s_prev, s_curr)`:

| Node ID | Tier | Advancement ID | Prerequisites | Preconditions | Verification Postcondition (`S_curr`) | Base Reward |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| `t1_gather_logs` | 1 | `minecraft:story/root` | `[]` | None | `Count("*_log") >= 4` | $+40.0$ |
| `t1_craft_planks_sticks` | 1 | `null` | `["t1_gather_logs"]` | `Count("*_log") >= 3` | `Count("*_planks") >= 8 && Count("stick") >= 4` | $+40.0$ |
| `t1_craft_table` | 1 | `null` | `["t1_craft_planks_sticks"]` | `Count("*_planks") >= 4` | `Count("crafting_table") >= 1 \|\| Nearby("crafting_table")` | $+50.0$ |
| `t1_wooden_pickaxe` | 1 | `null` | `["t1_craft_table"]` | `HasTable && Planks>=3 && Stick>=2` | `Count("wooden_pickaxe") >= 1` | $+60.0$ |
| `t2_mine_cobblestone` | 2 | `minecraft:story/mine_stone` | `["t1_wooden_pickaxe"]` | `HasPickaxe(>=Wood)` | `Count("cobblestone") >= 16` | $+80.0$ |
| `t2_stone_tools` | 2 | `minecraft:story/upgrade_tools` | `["t2_mine_cobblestone"]` | `Cobble >= 11 && Stick >= 6` | `Count("stone_pickaxe") >= 1 && Count("stone_axe") >= 1 && Count("stone_sword") >= 1` | $+100.0$ |
| `t2_acquire_raw_food` | 2 | `minecraft:husbandry/root` | `["t2_stone_tools"]` | `HasSword(>=Stone)` | `FoodItemsCount >= 5` | $+85.0$ |
| `t2_furnace_and_fuel` | 2 | `null` | `["t2_stone_tools"]` | `Cobble >= 8` | `Count("furnace") >= 1 && (Count("coal") + Count("charcoal") >= 4)` | $+90.0$ |
| `t2_cook_food_and_torches` | 2 | `null` | `["t2_acquire_raw_food", "t2_furnace_and_fuel"]` | `HasFurnace && HasFuel` | `CookedFoodCount >= 4 && Count("torch") >= 8` | $+100.0$ |
| `t2_survive_night_or_bed` | 2 | `minecraft:adventure/sleep_in_bed` | `["t2_cook_food_and_torches"]` | `Torches >= 4` | `Unlocked("sleep_in_bed") \|\| EnclosedLitShelterBuilt` | $+120.0$ |
| `t3_mine_iron_ore` | 3 | `null` | `["t2_survive_night_or_bed"]` | `HasPickaxe(>=Stone) && Torches>=8` | `Count("raw_iron") + Count("iron_ingot") >= 12` | $+150.0$ |
| `t3_smelt_iron_ingots` | 3 | `minecraft:story/smelt_iron` | `["t3_mine_iron_ore"]` | `Count("raw_iron") >= 1` | `Count("iron_ingot") >= 12` | $+160.0$ |
| `t3_shield_and_water_bucket` | 3 | `null` | `["t3_smelt_iron_ingots"]` | `IronIngot >= 4 && Planks >= 6` | `HasEquippedOrInv("shield") && Count("water_bucket") >= 1` | $+175.0$ |
| `t3_iron_tools_and_armor` | 3 | `minecraft:story/obtain_armor` | `["t3_shield_and_water_bucket"]` | `IronIngot >= 13 && Stick >= 4` | `Count("iron_pickaxe") >= 1 && Count("iron_sword") >= 1 && ArmorPoints >= 10` | $+220.0$ |
| `t3_slay_hostile_mob` | 3 | `minecraft:adventure/kill_a_mob` | `["t3_iron_tools_and_armor"]` | `HasSword(>=Iron) && HasShield` | `Unlocked("adventure/kill_a_mob")` | $+200.0$ |
| `t4_deep_mining_prep` | 4 | `null` | `["t3_iron_tools_and_armor"]` | `HasPickaxe(>=Iron)` | `Count("torch") >= 32 && CookedFood >= 12 && Count("iron_pickaxe") >= 2` | $+180.0$ |
| `t4_mine_diamonds` | 4 | `minecraft:story/mine_diamond` | `["t4_deep_mining_prep"]` | `HasPickaxe(>=Iron) && WaterBucket>=1` | `Count("diamond") >= 5` | $+400.0$ |
| `t4_craft_diamond_gear` | 4 | `minecraft:story/shiny_gear` | `["t4_mine_diamonds"]` | `Count("diamond") >= 5 && Stick >= 4` | `Count("diamond_pickaxe") >= 1 && Count("diamond_sword") >= 1` | $+350.0$ |
| `t4_harvest_obsidian` | 4 | `minecraft:story/form_obsidian` | `["t4_craft_diamond_gear"]` | `Count("diamond_pickaxe") >= 1 && WaterBucket >= 1` | `Count("obsidian") >= 10` | $+380.0$ |
| `t5_enter_the_nether` | 5 | `minecraft:story/enter_the_nether` | `["t4_harvest_obsidian"]` | `Count("obsidian") >= 10 && Count("flint_and_steel") >= 1` | `Dimension == "minecraft:the_nether"` | $+750.0$ |

### 4.3 Dynamic Regression & Sub-Goal Scheduler (`CurriculumManager`)
Implement `CurriculumManager.GetNextObjective(BotTelemetrySnapshot state)` purely in native code (0 LLM calls):
1. **Dynamic Regression Handling:** Evaluate `VerificationPostcondition(state)` on earlier tool/food nodes every cycle. If the bot died or broke its `iron_pickaxe` while working on Tier 4, the DAG automatically detects that `t3_iron_tools_and_armor` is no longer satisfied and schedules a recovery task before resuming `t4_mine_diamonds`.
2. **Automatic Sub-Task Decomposition:** Only if a `CurriculumNode` accumulates $\ge 3$ consecutive failures (`ConsecutiveFailures >= 3`), invoke `CurriculumDecomposer.SplitNodeAsync(node, state, reflexionHistory)` on the LLM to split node $n$ into $2$–$3$ simpler micro-tasks.

---

## 5. MODULE 3: VOYAGER-STYLE CODE SYNTHESIS & SANDBOX (`execution/` & `skills/`)

### 5.1 High-Level Bot Control API (`IWillowBotActions`)
Because smaller 7B–8B local LLMs on 12GB VRAM are more prone to hallucinating complex low-level vector math, **keep `IWillowBotActions` high-level, defensive, and self-healing**. Every method checks `ct.ThrowIfCancellationRequested()`, logs telemetry to `ExecutionTelemetry`, and returns a structured `ActionResult`:

```csharp
public readonly record struct ActionResult(
    bool Success,
    string ErrorCode, // "NONE", "PATH_BLOCKED", "MISSING_ITEM", "NO_TOOL", "HOSTILE_INTERRUPT", "TIMEOUT"
    string Message
);
```

Mandatory `IWillowBotActions` interface methods:
- **Perception & Queries (Zero-Allocation where possible):**
  - `BotTelemetrySnapshot GetState()`
  - `int GetItemCount(string itemNameOrPattern)`
  - `bool HasItem(string itemNameOrPattern, int minCount = 1)`
  - `IReadOnlyList<BlockInfo> FindBlocks(string blockIdPattern, int searchRadius = 32, int maxResults = 16)` (Uses spiral chunk-section scanning with pre-allocated buffer; never scans $>32$ block radius to protect CPU tick time)
  - `IReadOnlyList<PerceivedEntity> FindEntities(string entityIdPattern, int searchRadius = 24, bool onlyAlive = true)`
  - `Vector3? FindSafeSurfaceOrWater(int searchRadius = 32)`
- **Movement & Exploration (Time-Sliced A* Pathfinding):**
  - `Task<ActionResult> MoveToAsync(Vector3 targetPos, double arrivalRadius = 2.0, int timeoutMs = 20000)` (A* pathfinder MUST be time-sliced to max $2.5\text{ ms}$ per game tick so pathing never lags the server)
  - `Task<ActionResult> ExploreAsync(Vector3 preferredDirection, int distance = 32, int timeoutMs = 25000)`
  - `Task<ActionResult> DigStaircaseToYAsync(int targetY, bool placeTorches = true, CancellationToken ct = default)`
- **Block Harvesting & Placement:**
  - `Task<ActionResult> MineBlocksAsync(string blockIdPattern, int targetCount, int maxSearchRadius = 32)`
    *(Automatically validates harvest level against `SurvivalMechanicsRegistry` before swinging; if lacking required pickaxe tier, immediately aborts and returns `ErrorCode = "NO_TOOL"` and increments `ImproperToolMineAttempts`!)*
  - `Task<ActionResult> MineBlockAtAsync(Vector3 blockPos, bool collectDrops = true)`
  - `Task<ActionResult> PlaceBlockAsync(string itemId, Vector3 placeAtPos)`
  - `Task<ActionResult> UseBucketAtAsync(string bucketType, Vector3 targetBlockPos)`
- **Workstations, Crafting & Smelting:**
  - `Task<ActionResult> EnsureWorkstationPlacedAsync(string stationItemId)`
  - `Task<ActionResult> ReclaimWorkstationAsync(string stationBlockId)`
  - `Task<ActionResult> CraftAsync(string outputItemId, int craftOperationsCount = 1)` (Automatically resolves intermediate 2x2 planks/sticks if missing and logs exist in inventory!)
  - `Task<ActionResult> SmeltAsync(string inputItemId, string fuelItemId, int count = 1)`
- **Equipment, Sustenance & Combat:**
  - `Task<ActionResult> EquipBestToolForBlockAsync(string blockId)`
  - `Task<ActionResult> EquipArmorAndWeaponsAsync()`
  - `Task<ActionResult> EatIfHungryAsync(int hungerThreshold = 15)`
  - `Task<ActionResult> EngageEntityAsync(int entityId, CombatTactic tactic = CombatTactic.SmartAuto)`
  - `Task<ActionResult> BuildSimpleShelterAsync(string blockMaterial = "cobblestone")`
  - `Task WaitTicksAsync(int ticks)`
  - `void Log(string message)`

### 5.2 AST Security, Compilation Caching & Execution Sandbox (`ScriptSandbox`)
1. **Static AST Inspection before Execution:**
   - Parse the LLM-generated script into an Abstract Syntax Tree (AST).
   - **Reject & Penalize (`-15.0`)** any code referencing:
     - File system / OS (`System.IO`, `os`, `subprocess`, `fs`, `child_process`)
     - Network sockets (`System.Net`, `http`, `socket`, `requests`)
     - Process reflection or termination (`System.Reflection`, `Environment.Exit`, `sys.exit`)
     - Unbounded synchronous loops (`while(true)` or `for(;;)` that do not contain an `await` or `ct.ThrowIfCancellationRequested()`)
     - Blocking thread sleep (`Thread.Sleep`, `time.sleep` — must use `await bot.WaitTicksAsync(...)`).
2. **Collectible Compilation & LRU Delegate Cache:**
   - Hash script code with `SHA256`. If already compiled in `CompiledSkillCache` (LRU max 64 entries), execute the cached delegate directly in $< 0.1\text{ ms}$ without invoking the compiler.
   - When compiling new scripts, use a collectible assembly context (`AssemblyLoadContext(isCollectible: true)` in C# or isolated dictionary scope in Python/JS) and unload failed scripts immediately so RAM usage stays flat over 24+ hour runs.

### 5.3 KV-Cache Optimized Prompt Template (`CodeSkillAgent`)
Structure the LLM code-generation prompt into a **Static Prefix** (cached in VRAM across all calls) and a **Compact Dynamic Suffix** (under 1,200 tokens):

```text
=== STATIC SYSTEM PREFIX (100% IDENTICAL ACROSS CALLS FOR KV-CACHE REUSE) ===
You are Willow-Bot's Voyager Code Synthesizer. Write a concise, fault-tolerant async control function using ONLY `IWillowBotActions bot` and `CancellationToken ct`.
API SIGNATURES:
- bot.HasItem(pattern, minCount=1) -> bool | bot.GetItemCount(pattern) -> int
- await bot.MineBlocksAsync(blockPattern, count, radius=32) -> ActionResult
- await bot.EnsureWorkstationPlacedAsync(stationId) -> ActionResult
- await bot.CraftAsync(itemId, count=1) -> ActionResult
- await bot.SmeltAsync(inputId, fuelId, count=1) -> ActionResult
- await bot.DigStaircaseToYAsync(targetY, placeTorches=true, ct) -> ActionResult
- await bot.EatIfHungryAsync(threshold=15) -> ActionResult
- await bot.EquipArmorAndWeaponsAsync() -> ActionResult
- await bot.EngageEntityAsync(entityId) -> ActionResult
- await bot.BuildSimpleShelterAsync(material) -> ActionResult
- await bot.ExploreAsync(dir, dist=32) -> ActionResult
RULES:
1. Check `if (!res.Success) return res;` on critical steps.
2. Never mine stone/ores without checking/crafting required pickaxe tier first (-8.0 penalty).
3. Output strictly valid JSON: {"reasoning": "max 2 short sentences", "skill_name": " string", "description": "string", "code": "string"}

=== DYNAMIC TASK SUFFIX (COMPACT TOKEN FORMAT) ===
[TASK] {task.Id}: {task.Description} | Postcondition: {task.VerificationDescription} (Reward: +{task.BaseReward})
[RULES] {filtered_task_mechanics_one_liner}
[STATE] {compact_telemetry_string}
[RL_HISTORY] {max_3_compact_reflexion_lines}
[RAG_SKILLS] {top_3_retrieved_skill_signatures}
```

### 5.4 Deterministic-First Self-Verification Critic (`SelfVerificationCritic`)
To save GPU cycles on 12GB VRAM:
1. Capture $S_{t+1} = \text{bot.GetState()}$.
2. **Step 1 — Native Deterministic Critic ($0\text{ ms}$ GPU):**
   - Evaluate `task.VerificationPredicate(S_t, S_{t+1})`.
   - Inspect `S_{t+1}.ExecutionTelemetry` and inventory deltas ($\Delta \text{Inventory} = S_{t+1} - S_t$).
   - Deterministically classify common failures without calling the LLM:
     - `CompilationFailed` $\rightarrow$ `SYNTAX_ERROR`
     - `RuntimeExceptions.Any()` $\rightarrow$ `API_MISUSE`
     - `ImproperToolMineAttempts > 0` $\rightarrow$ `WRONG_HARVEST_TOOL`
     - `!S_{t+1}.Vitals.IsAlive` $\rightarrow$ `BOT_DIED`
     - `PathfindingTimeouts > 0` $\rightarrow$ `PATHFINDING_STUCK`
3. **Step 2 — LLM Fallback Critic (Only if failure is ambiguous):**
   - Only if the script ran cleanly with 0 exceptions/timeouts yet `VerificationPredicate` returned `false` and the deterministic classifier cannot pinpoint the missing item prerequisite, issue a fast 150-token diagnostic prompt to the resident LLM.

---

## 6. MODULE 4: ZERO-VRAM HYBRID RAG SKILL LIBRARY (`skills/SkillStore`)

Implement a persistent, low-memory skill repository in `skills/`:
1. **Storage & Memory Footprint:**
   - `skills/skill_registry.json`: Metadata, preconditions, postconditions, and Q-values.
   - `skills/library/<skill_id>.code`: Source code loaded on demand.
   - `skills/embeddings.bin`: Flat binary array of 384-dim `float` vectors (1,000 skills = only `1.5 MB` of RAM!).
2. **Zero-VRAM Hybrid Retrieval (`RetrieveRelevantSkills(task, state, topK = 3)`):**
   - Compute similarity on CPU using SIMD vector intrinsics + BM25 lexical overlap + RL Q-value weighting:
     $$\text{RetrievalScore}(k) = 0.50 \cdot \text{Sim}_{\text{CPU}}(q, k) + 0.30 \cdot \sigma\left(\frac{Q(c, k)}{50.0}\right) + 0.20 \cdot \text{PreconditionMatchRatio}(S_t, k)$$
   - Limit `topK = 3` in the prompt to keep context size minimal.
3. **Skill Composition & Automatic Mutation:**
   - Inject verified skill delegates into the `ScriptSandbox` global scope so newly synthesized scripts can invoke previously learned skills directly.
   - If a stored skill's `QValue` drops below $-15.0$ or fails $2$ times in a row, mark `Status = SkillStatus.NeedsRefinement`, pass its source code + compact failure trace to `CodeSkillAgent.MutateSkillAsync()`, and replace it with `<skill_id>_v2`.

---

## 7. IMPLEMENTATION ROADMAP & DELIVERABLES

Scan the existing **Willow-Bot** workspace first to identify the host language, Minecraft protocol/bot library, and LLM client integration. Then implement the following deliverables cleanly and completely:

1. **Deliverable 0 — Hardware, Memory & 12GB VRAM Optimization Layer (`hardware_opt/`):**
   - `LocalLlmClient` enforcing single-resident model policy, `num_ctx: 4096`, `num_predict: 768`, CPU thread capping, single-flight `SemaphoreSlim(1, 1)`, and static prefix ordering for KV-cache reuse.
   - `CompactTelemetryFormatter` compressing bot state into $< 150$ tokens.
   - `MemoryPoolManager` & `LiveReflexState` ensuring zero-allocation 20Hz reflex checks and collectible script compilation caching.

2. **Deliverable 1 — Telemetry & RL Reward Engine (`rl_rewards/`):**
   - `BotTelemetrySnapshot` data model & state diffing utility.
   - `RewardCalculator` implementing the exact $R_{\text{total}}$ mathematical equations ($R_{\text{adv}}, R_{\text{task}}, R_{\text{shaping}}, R_{\text{intrinsic}}, P_{\text{vital}}, P_{\text{exec}}, P_{\text{time}}$).
   - `SkillQTable` with 32-bit bitmask context hashing, TD-learning updates, fast-path LLM bypass when $Q(c, n, k) \ge +10.0$, and JSON persistence.
   - `ReflexionMemoryLog` with bounded ring buffers.

3. **Deliverable 2 — Survival Mechanics & Advancement Curriculum (`curriculum/`):**
   - `SurvivalMechanicsRegistry` with task-filtered rule injection, tool harvest tiers, crafting recipes, smelting fuels, and safe Y-level mining rules.
   - `CurriculumDAG` & `advancement_dag.json` containing all 20 nodes across Tiers 1–5 with deterministic verification predicates and regression recovery.
   - `CurriculumManager` with $\epsilon$-greedy / Softmax utility scheduling and automatic sub-task decomposition after 3 failures.

4. **Deliverable 3 — Voyager Code Sandbox & Time-Sliced Bot Actions API (`execution/`):**
   - `IWillowBotActions` interface and concrete implementation with $2.5\text{ ms}$/tick time-sliced A* pathfinding, spiral block scanning, and automatic harvest-tool validation.
   - `ScriptSandbox` with static AST security validation, SHA256 LRU compiled delegate caching, collectible assembly unloading, cancellation token injection, and timeout enforcement.

5. **Deliverable 4 — Code Synthesizer, Deterministic Critic & Zero-VRAM RAG Store (`skills/`):**
   - `SkillStore` with zero-VRAM CPU hybrid retrieval (BM25 + SIMD vectors + Q-value weighting) and flat binary storage.
   - `SelfVerificationCritic` performing native deterministic verification first and falling back to the LLM only on ambiguous failures.
   - `CodeSkillAgent` orchestrating KV-cache-friendly prompt assembly, syntax retry loops, and skill mutation.

6. **Deliverable 5 — Dual-Loop Master Orchestrator (`WillowBotOrchestrator`):**
   - The 20Hz Fast Reactive Survival Reflex loop (0% GPU usage, auto-eat, auto-shield/flee, lava avoidance) with preemption over the cognitive loop and LLM inference pausing during active combat.
   - The Slow Cognitive RL + Voyager loop tying together `CurriculumManager` $\rightarrow$ `SkillQTable` Fast-Path $\rightarrow$ `SkillStore` $\rightarrow$ `CodeSkillAgent` $\rightarrow$ `ScriptSandbox` $\rightarrow$ `SelfVerificationCritic` $\rightarrow$ `RewardCalculator`.
   - Rich structured telemetry logging (`[VRAM-OPT]`, `[RL-REWARD]`, `[Q-FASTPATH]`, `[CURRICULUM]`, `[VOYAGER-SYNTH]`, `[CRITIC]`, `[REFLEX-INTERRUPT]`).

Start by outputting a brief summary of the existing files you detect in the Willow-Bot repository, and then begin implementing Deliverable 0 through Deliverable 5 in full.