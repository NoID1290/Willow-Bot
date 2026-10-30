# Willow: local autonomous Minecraft player

Fabric **Minecraft Java 26.3**, Fabric Loader **0.19.5**, Carpet
**26.3+v260915**, Fabric API **0.161.0+26.3**, Java **25**, Python **3.10+**,
and Ollama. This is a working experimental survival agent, not a trained
Minecraft expert or a promise to finish the game.

## Using it

The installer places three JARs in `~/.minecraft/mods`. Restart Minecraft using
your existing **fabric-loader-0.19.5-26.3** installation, then open a **new test
world**. A **Choose Willow's AI** screen lists the text-generation models already
installed in local Ollama. Select one and click **Start Willow**. The last selected
model is remembered; **Refresh** reloads the list and **Not now** leaves Willow
stopped. No model is downloaded by this screen.
Once started, there is no separate Python terminal or per-action approval.
Ollama must be running; its system service is already enabled on this computer.
The first inference after a cold model load can take substantially longer.

**Back up valuable worlds.** Willow really mines blocks, consumes resources,
places blocks, attacks non-player mobs, and can die. The mod is enabled in
singleplayer worlds by default, not just newly created worlds.

Communicate through Minecraft chat:

```text
/brain say Hello Willow, build a small wooden shelter.
/brain status
/brain pause
/brain resume
/START
/START gemma4:12b-it-qat
/STOP
/brain camera off
/brain camera on
```

`/brain say` is available without enabling cheats in singleplayer. Messages
are queued, not prerequisites for autonomous activity. Responses appear as
`<Willow_AI>` in chat. Pause stops movement/actions and prevents a late model
response from restarting them. Resume restarts autonomous planning.

**`/STOP`** cancels pending decisions, terminates Willow's Python child, saves and
disconnects Willow, and restores your camera/player. It does not stop Minecraft,
kill the shared Ollama service, cause a survival death, or erase inventory/memory.
**`/START`** opens the model picker again; **`/START <exact-model-name>`** selects an
installed model directly. Stop first before switching models. Both work without
quitting the world or enabling cheats in singleplayer. Lowercase aliases are
`/brain start [model]` and `/brain stop`; `/brain model` also opens the picker while
stopped. **Use uppercase `/STOP`: vanilla's lowercase `/stop` is not overridden.**
Unlike STOP/START, `/brain pause` and `/brain resume` retain the existing player
and brain process.

### Automatic cinematic follow

When Willow connects, the singleplayer host automatically watches from a smooth,
slightly elevated over-the-shoulder camera. Position and heading ease toward
Willow and sharp turns wrap correctly. The camera is **noclip**: it passes through
blocks instead of being pulled in by walls (you see through terrain from inside
it, as in spectator mode). Respawns/reconnections reacquire Willow. The camera
resets across long teleports rather than sweeping through unloaded terrain.

While following Willow:

| Key (default) | Action |
| --- | --- |
| `=` (hold) | Zoom in, down to 1.5 blocks |
| `-` (hold) | Zoom out, up to 32 blocks |
| `0` | Reset zoom to 4.5 blocks |
| `I` | Show/hide Willow's inventory window |
| `O` | Open Willow AI settings (works any time in a world) |

The keys can be rebound under **Options → Controls → Key Binds → Willow AI**.
The screen shows a live copy of Willow's survival HUD: hotbar with his selected
slot and offhand, hearts, armor, food, air and experience. It also shows a
vanilla-style inventory window in the top-right with his model, armor, offhand,
main inventory and hotbar. This is copied from the server every tick (20 times per second).
It is view-only: you cannot move his items.

To keep Willow loaded even far from your original position, **your player is
temporarily placed in spectator mode** and follows through Minecraft's normal
server camera system. Willow itself remains a survival player. The host's
original dimension, position, orientation and game mode are saved to
`<world>/localai/camera-viewer.json` **before** switching. `/STOP`,
`/brain camera off`, and normal world shutdown restore that state; an unexpected
exit leaves the return-state file for recovery on the next login. Camera mode
and view bobbing are restored on the client when following ends.

Use `/brain camera off` to play normally while Willow continues autonomously;
`/brain camera on` resumes automatic viewing. This preference persists. Automatic
viewing and the startup picker are for the **singleplayer host only**, not other
LAN players or remote dedicated-server clients. On a dedicated server, use
`/START <model>` with operator permission; opt-in configuration is still required.

The **world must remain open and ticking**. Closing the world stops its Python
brain. Vanilla's Escape/pause screen can still pause the whole singleplayer
world. `pauseOnLostFocus:false` in `~/.minecraft/options.txt` lets the world
continue when you switch windows; it does not override explicit pausing.
When cinematic viewing is disabled, your own survival player is exposed to mobs
while the world keeps running. Spectator viewing protects the host, not Willow.
This is not a separately running 24/7 Minecraft server.

### Ollama starts and stops with the game

When you **quit Minecraft**, the mod first unloads every loaded Ollama model,
which frees VRAM/RAM right away and needs no extra permission. It then stops the
`ollama` system service. When Willow is needed again (model picker or `/START`),
the mod starts the service if it is not running. Leaving a world without quitting
the game keeps Ollama running, so you can reopen a world quickly.

Ollama is a root-owned system service, so allow stop/start **once**:

```bash
bash ~/Projects/minecraft-local-ai/scripts/setup-ollama-control.sh   # asks for your sudo password
```

This installs a polkit rule allowing only your user to start/stop only
`ollama.service`. Remove it with `--remove`. Without it, models are still unloaded
on quit, but the service stays up. Set `"stopOllamaOnExit": false` in the config
to leave Ollama untouched. Ollama still starts at boot unless you run
`sudo systemctl disable ollama`. The mod starts it when needed either way.

## Any model: automatic per-model configuration & LFM3 Local AI

Willow supports all modern local AI models including **Liquid Foundation Models (LFM3 & LFM2.5)**:
- **LFM3-1.7B (Text / Agent)**: High-throughput tri-hybrid architecture (gated short convolutions, sliding-window attention, and global attention) with up to 131K-256K context and rapid structured JSON tool calling.
- **LFM3-VL-2B (Vision-Language)**: Processes real-time visual streams and visual scene snapshots.
- **LFM3-Audio-2B (Audio)**: Real-time speech and acoustic sound recognition.
- **LFM3-Omni-2B (Omni Multimodal)**: Simultaneous visual, auditory, and textual perception.
- **d1-2B & d1-Omni-2B (Decision Models)**: High-speed single-pass survival decision models.

### Concurrent Multi-Stream Sensory Pipeline (See, Listen, Decide simultaneously)

Willow runs a synchronized concurrent sensory pipeline where Seeing, Listening, and Deciding execute in parallel:
1. **Live Sight (Vision Stream)**:
   - **Crosshair Gaze**: Raycast targeting up to 12m ahead identifying the exact block or entity Willow is staring at.
   - **Field of View (FOV)**: Scans entities and landmarks in Willow's visual cone (~90° horizontal, ~70° vertical) with line-of-sight occlusion checks.
   - **7x5 ASCII Retina**: Visual depth and semantic map showing blocks (`#`), ores (`*`), tree logs (`T`), water (`~`), lava (`!`), entities (`Z`/`S`), and crosshairs (`+`).
   - **Visual Frame Snapshots**: 64x64 color raster PNG snapshots (base64) delivered directly to multimodal vision models (LFM3-VL / Omni).
2. **Live Sound (Auditory Stream)**:
   - Real-time sound capture hook on `ServerLevel.playSeededSound`.
   - Records spatial distance, relative head bearing (`ahead`, `right`, `behind`, `left`), and elevation (`level`, `above`, `below`).
   - Categorizes sound urgency (`CRITICAL`, `WARNING`, `NOTICE`, `AMBIENT`).
   - **Reflex Reaction**: Immediate survival reflexes when critical acoustic threats (e.g. creeper fuse priming within 5m) are heard, dodging danger instantly without waiting for long planning cycles.
3. **Live Doing (Activity Stream)**:
   - Tracks ongoing action (`mining`, `crafting`, `navigating`, `pillaring`, `attacking`), target coordinates, progress %, duration ticks, held items, stance, and active goal.
4. **Decision Engine**:
   - Synthesizes sight, sound, doing, and memory in parallel to drive autonomous survival actions.

Before its first decision (and whenever
the model changes) the brain reads the model's architecture from Ollama
(`/api/show`) and builds a **profile** (`brain/modelprofile.py`), logged as
`Model profile ...` in `brain.log` and saved to `<world>/localai/model-profile.json`:

- **Context length**: the model's *trained* context, capped at 32768, then halved
  until the estimated KV cache (layers × KV heads × head size, counting only
  full-attention layers for sliding-window/hybrid models) fits ~8 GiB. A context
  bigger than the model was trained on is never requested.
- **Thinking**: enabled only when Ollama reports the `thinking` capability
  (gpt-oss uses its `medium` level). Non-thinking models are not asked to think.
- **Output limits**: thinking models get up to 8192 tokens to reason; plain models
  get 1024 (actor) / 2048 (strategist), always a fraction of the context.
- **Prompt budget**: observation/history size scales with what is left of the context.
- **Sampling**: the model's own Modelfile defaults (temperature, top_k...) are kept;
  0.6 is used only when a model ships none.
- Embedding size, layers, MoE expert counts, quantization and capabilities are shown
  in the settings screen.

**Why Willow got stuck with `llama3:8b`:** the old fixed config asked for a 32K
context (llama3 is trained on 8K) and allowed 8192 output tokens, equal to the
whole context. llama3 has no thinking mode, so it sometimes rambled ~4000 tokens
(almost 2 minutes per decision) until it was truncated, and its long free-text
fields and out-of-range coordinates were rejected. It now gets 8K context, 1024
output tokens, overlong text is trimmed instead of rejected, stale mining
coordinates fall back to automatic selection and far `goto` targets become
`explore`. A replay against the real model gave valid decisions in 1.3-3.6 s.

### Settings screen (`O` or `/brain settings`)

Shows what was detected for each installed model (`<` / `>` to switch) and lets you
override **context length**, **actor output**, **strategist output**,
**temperature** and **thinking** (auto/on/off). Blank fields mean auto; the
right-hand hints show the value in use. Changes are checked against the real
model before saving. **Save & apply** on the running model restarts only
Willow's brain process (the player stays connected, the brain memory is kept).
The model picker also has a **Settings...** button. Overrides are stored per
model under `modelSettings` in the config.

### Ollama / Willow console (`/OLLAMA` or `/brain console`)

Opens a terminal window (Konsole on KDE; also gnome-terminal, xterm, kitty, ...)
running `monitor.py`: a live panel with the Ollama version, loaded models (size,
% on GPU, context, unload time), Willow's active profile, last prompt/generation
tokens per second and calls per minute, followed by Willow's colour-coded brain
log (actor actions, strategist goals, warnings) and Ollama service warnings
(context shift, truncation, out of memory). It can also be run by hand:

```bash
python3 ~/.minecraft/config/localai/runtime/monitor.py --log "$HOME/.minecraft/saves/<world>/localai/brain.log"
```

## Hardware and model choice

Detected hardware: Ryzen 7 5800X3D (8 cores / 16 threads), approximately
64 GB installed RAM, NVIDIA RTX 4070 Ti with 12 GB VRAM.

Initially selected: **`gemma4:12b-it-qat`**, the official Gemma 4 12B QAT **Q4_0** model
requested for this computer. Ollama reports 11.9B parameters, Q4_0 quantization,
and thinking support. The exact tag matters: the bare `gemma4:12b` manifest
currently points to Q4_K_M, not Q4_0.

At **32K context**, the local Gemma runner reported 7,686,900,939 bytes (about
**7.2 GiB**) resident entirely on the GPU. Total GPU use was about 9.3 GiB,
leaving about 2.4 GiB free in that session. The machine's 64 GB system RAM leaves
room for offloading if other GPU workloads require it, but RAM capacity is not
GPU bandwidth, and forcing offload is not a speed improvement.

Gemma uses its recommended temperature 1.0, top-k 64 and top-p 0.95, with thinking
enabled. The brain checks each selected model's local metadata once at startup;
models without thinking support omit the thinking option rather than failing
on an unsupported feature. Embedding-only and remote/cloud models are excluded.
A measured warm startup decision used 1,681 generated tokens and took
about 31 seconds. This is deliberately a more deliberative setup, not a speed
upgrade. The initial smaller reasoning budget exhausted before some responses
finished; the final allowance is 8,192 tokens including reasoning and JSON output.

The earlier **`qwen3.5:9b-q4_K_M`** and **`qwen3.5:4b`** remain downloaded.
All inference stays local; there is no paid/cloud endpoint.

The original 4K-context five-scenario decision smoke benchmark favored 9B over 4B:
9B passed 4/5, 4B passed 3/5 with the tested startup prompt.
These tiny synthetic cases are **not an exhaustive model ranking** or a
long-horizon gameplay benchmark. The 9B run occupied about 5.1 GiB of GPU
memory as reported by Ollama, versus about 2.9 GiB for 4B. Warm responses in
those cases were around one second; loading, other GPU workloads, and real
world observations change latency. See `benchmarks/*.json`.

Both models missed a hunger-priority case. Therefore **automatic eating is a
deterministic game-side reflex**, not something entrusted solely to the LLM.
The selection is a practical, measured starting point for this machine, not a
claim that any model is universally the "best Minecraft AI."

Bounded resource settings:

| Setting | Value |
| --- | --- |
| Ollama context | 32,768 tokens per role |
| Maximum generated output | 8,192 tokens per role, including reasoning |
| Prompt plus JSON-schema budget | 20,000 UTF-8 bytes, with template/output headroom |
| Concurrent requests | One actor + at most one background strategist per world |
| Reasoning/thinking mode | Enabled if supported by the selected model |
| CPU inference threads | 6 |
| Temperature | Gemma 1.0; Qwen fallback 0.25 |
| Artificial planning delay | None; one request at a time, after the action finishes |
| Model keep-alive | 10 minutes |
| Strategy refresh | At most once per 60 seconds; 180-second request timeout |
| Actor + corrective retry deadline | 240 seconds total |
| Action deadline | 2,400 game ticks / 120 seconds |
| Follow-up plan | Up to eight actions, expires after five minutes |
| Navigation search | 3,000 nodes, 24-block horizontal radius, loaded chunks only |

There is no ever-growing chat transcript sent to Ollama. Observations and
selected memories are rebuilt for every request and pruned to the byte budget.
Ollama also enforces the context/output limits. Avoid running ComfyUI or other
large GPU models alongside Minecraft and Willow if GPU memory becomes tight;
Minecraft shaders/high-resolution texture packs need additional headroom.
No system-wide Ollama settings were changed.

### Actor, strategist, and saved skill code

Willow now has **two separate prompts**: an actor that chooses immediate actions,
and a background strategist that reviews observations/outcomes, proposes a goal,
and writes reusable skill programs. The strategist is scheduled after an action
is chosen, so it can think while Willow moves or mines. Its queue never grows
beyond one request. Each role has its own bounded 32K context, sharing the same
local Gemma model; this is not two independently loaded models.

**Ollama currently serializes inference on this computer.** Two roles can have
requests in flight, but this does not promise two simultaneous GPU generations.
A new actor request can still wait for an unfinished strategy generation. Ordinary
action execution and local skill evaluation do not wait for the strategist.
Strategy snapshots older than four minutes, from a different dimension, or preceding
new player chat are discarded. Only the main service thread owns the SQLite
connection. Strategist errors are logged explicitly; the independent actor can
continue without its advice.

Accepted programs are saved under each world's `localai/skills/<hash>.py`,
with their canonical source, success/failure counts, and disabled state in
`localai/brain.sqlite3`. They use a **restricted Python-shaped language**:
`def skill():`, local variable assignments, `if/elif/else`, comparisons, bounded
integer addition/subtraction, `return`, `pass`, docstrings, and a small set of
observation/action helpers. The source is parsed and interpreted, **never passed
to Python `exec` or a shell**. No imports, file/network access, attributes,
arbitrary functions, or loops are permitted. Programs are limited to 4,096 bytes,
256 AST nodes and 16 local variables; at most 16 versions are retained. This is reusable game-skill
code, not unrestricted rewriting of the mod or model-weight training.

For example, a learned program can inspect `count("#pickaxes")`,
`nearby("minecraft:stone")`, and `drops("minecraft:cobblestone")`, then choose
`mine("minecraft:stone")` or `pickup_item("minecraft:cobblestone")`. It can also
query `failures("mine", "minecraft:stone")` and `successes(...)` from its learned
lookup table. Every returned action is checked against
the **current**, not remembered, observation and Minecraft still applies its
survival rules. Invalid programs are disabled and logged; three observed action
failures disable a version. A failed action returns control to the actor before
trying routines again. Chat bypasses routines for at least 60 seconds, and
hunger/stuck handling takes priority. Skills may still be wrong: these are
experiments with recorded consequences, not trusted Minecraft expertise.

### Longer plans and situation lookup

The actor can return up to eight follow-up steps, not just an isolated command.
These execute without additional LLM calls, but each step is checked against a
fresh observation. A failed action, missing prerequisite, changed dimension,
low health/hunger, nearby threat or player chat interrupts the plan. Old mining
coordinates are not replayed after a block disappears. Execution is still
interruptible by the game-side survival reflexes.

Willow builds an `action_lookup` table in its existing brain database from
**measured outcomes**: situation, action, target, attempts, successes, failures,
damage and net inventory changes. Situations include dimension, an eight-block
horizontal/four-block vertical region and currently owned tool types. Two
consecutive failures put that action/target on a two-minute cooldown in that
situation; changing position or equipment allows a new experiment. Other
targets remain available. At most 10,000 lookup rows are retained.
Placement choices exclude cooldown-blocked materials and non-block inventory
items, and constrain coordinates to currently reachable, unoccupied support sites.
This applies to slabs and other blocks, not only crafting tables. Queued steps
and learned routines are checked against the same fresh placement information.

Both prompts now receive the current lookup, inventory changes, owned-tool
milestones, up to 20 recent action outcomes and 12 lessons. This distinguishes
"I already have two stone pickaxes" from "I still need a stone pickaxe."
The actor normally stays quiet during routine work; repeated identical
unsolicited speech is suppressed for 60 seconds. Direct chat still gets replies.

### Crafting awareness

Observations distinguish an owned crafting table from a reachable placed one,
explain the current 2x2/3x3 grid, count wood/sticks/tools, list actual craftable
recipe IDs, and provide reachable unoccupied placement sites. They also include
ground/water/air state, weather, biome, eight neighboring terrain cells with
standability and head clearance, nearby entity distance/hostility, and item-drop
counts. Up to three positions per block type are retained (36 blocks total),
with tool requirements rather than only one vaguely nearby resource.
Remembered table locations are not suggested when a table is already in inventory.

A supplied foundation routine handles **logs → planks → table → place table →
sticks → wooden pickaxe**, choosing each next step from Minecraft's actual
recipes and inventory. It also collects wood-related drops before mining more.
These known steps bypass LLM inference; gathering resources and adapting beyond
the foundation still use the actor or learned programs. An unavailable recipe
is rejected even if some other recipe is craftable. The actor's structured-output
schema constrains craft targets to up to 32 current recipe IDs, prioritizing the
bootstrap step and recipes requested in chat; learned routines are validated
against the full list. If a model response still fails validation, the actor
receives the rejection reason and gets exactly one corrective attempt within the
same 240-second deadline. Invalid attempts never execute or count as completed
actions. A second invalid response is logged and returned as a distinct rejected
plan (HTTP 422), not a brain-service outage. Minecraft reports that Willow is
reconsidering, keeps the player and brain running, and requests a fresh plan after
a short bounded backoff. It does not invent an action or record success. Actual
Ollama/service failures still report a brain error (HTTP 503).

Pickup looks for walkable positions within collection reach instead of requiring
the player to stand in the item's exact block cell, and briefly allows airborne
drops to settle. Success requires an actual inventory increase, not merely a
disappearing item entity. A requested item ID is honored rather than silently
collecting another drop. Mining can target an exact observed block; approach
positions allow visibility margin, and mining can reposition up to three times
instead of immediately failing at an obstructed corner.

### Responsiveness tuning

The gameplay log showed roughly 0.7-1.5 second model responses, but a five-second
controller interval and repeated blocked actions made progress slow. The interval
was reduced to one second, then removed entirely. That earlier Qwen configuration
used **8K**; the current Gemma experiment uses **32K**. The prompt retains the full
inventory and up to six recent action outcomes instead of discarding all history
to fit an overly small observation budget. Commands that are clearly unavailable
(e.g. crafting with no craftable recipes, or pickup with no drops) are excluded
from that request's action schema. Navigation handles layered snow and bottom
slabs, considers short escape routes, and checks visibility from mining destinations.

More context improves information retention; it does not inherently speed up
inference. At 8K, the 9B model still fits entirely on this GPU (about 5.3 GiB reported
by Ollama). `benchmarks/qwen3.5-9b-8k.json` records the new configuration.

MoE active-parameter counts are not total memory requirements. Official Ollama
manifests list approximately **18.62 GB** of weights for
`qwen3:30b-a3b-q4_K_M` and **23.87 GB** for `qwen3.5:35b-a3b`.
Those do not fit entirely into a 12 GB GPU alongside Minecraft; they require
RAM offload or a different quantization/runtime. That might be worth a separate
benchmark, but is not an established speed upgrade over this GPU-resident 9B model.
Neither large MoE model was downloaded or benchmarked.

### Experimental recovery and mistakes

Willow now diagnoses three consecutive failures or eight non-resource actions
circling within five blocks as stuck, even when each tiny movement technically
succeeded. Its next model decision must use `escape`, rather than endlessly
choosing `explore` or `wait`. Recovery searches once (up to 3,000 nodes) for a
lower landing within six horizontal blocks, allowing drops of up to twelve
blocks. If that fails, it can mine nearby natural obstructions, including the
leaves/log/dirt block supporting its own feet. This can cause injury or death,
and may change terrain. It is intentionally not perfect-safety navigation.
If `escape` itself fails, other actions are allowed on the next decision instead
of forcing that same failed recovery forever.

Ordinary navigation allows up to three-block descents; riskier descents are
reserved for the explicit recovery action. Both modes still refuse unloaded
space, lava/fire landing surfaces, and player attacks. Recovery does not teleport,
grant items, make Willow invulnerable, or erase the consequences.

Action feedback includes before/after health, net health lost, final position,
and the attempted recovery method. These measured outcomes are saved as lessons
and fed back to the planner. A successful escape can also record damage; a death
is recorded as a failure. This provides useful experience, not model-weight
training or a guarantee that every mistake will teach a durable new skill.

There is no remaining artificial one-second wait between actions. Normal
inference and survival action time still apply; increasing VRAM allocation by
itself cannot speed a model that is already fully GPU-resident.

## What Willow can do

- Exist as a real server-side survival player with a name, inventory, hunger,
  health, physics, item pickup, and saved player data, using Carpet's fake-player
  implementation. No second Minecraft account or online skin lookup is needed.
- Walk, swim out of water, jump, sprint, sneak, and navigate via bounded local A*
  plus multi-step global heading pathfinding toward far waypoints; explore without
  instructions. Ordinary navigation avoids lava, fire, large drops, cactus, and
  similar hazards; explicit recovery permits damaging falls, pillaring, and support
  mining.
- Mine visible/reachable blocks using survival break timing and appropriate tools,
  automatically ignoring decorative ground clutter (`leaf_litter`, `wildflowers`)
  when gathering resources.
- Collect item drops and craft ordinary shaped/shapeless recipes, consuming the
  actual inventory ingredients. Recipes larger than 2x2 need a visible nearby
  crafting table.
- Smelt ores, raw metals, and food (`smelt`) using a reachable or inventory furnace
  and fuel (`coal`, `charcoal`, `planks`, `logs`, `sticks`).
- Manage storage containers (`chest`) to deposit excess building materials or
  retrieve needed supplies from nearby chests and barrels.
- Sleep in or place beds (`sleep`) to set respawn points and pass the night; trade
  with nearby villagers and wandering traders (`trade`); place, mount, and row boats
  across water (`boat`).
- Dig 1x2 tunnels through underground or blocking terrain (`tunnel`), bridge across
  gaps or fluids with solid blocks (`bridge`), construct protective walls and roofs
  (`shelter`), and light or enter Nether portals (`portal`).
- Place inventory blocks against visible support faces; eat real food; attack
  observed non-player living entities; defend against nearby hostile mobs.
- Replan from action results, remember goals and observations, and resume after
  ordinary deaths. Hardcore deaths pause the controller instead of bypassing
  hardcore rules.

Limitations: no unrestricted self-modification outside the validated skill AST.
Complex multi-room architectural blueprints are not guaranteed. Unsupported
requests should be treated as experiments, not assured capabilities, and the
model can still make suboptimal decisions under novel conditions.

## Saved "brain" and learning

Each world has its own state:

```text
~/.minecraft/saves/<world>/localai/
    brain.sqlite3       goals, strategy, decisions, outcomes, skills, action_lookup, permanent memory
    memory.json         readable export of the permanent memory (every 10 decisions and on shutdown)
    skills/<hash>.py    validated generated programs and supplied foundation routine
    brain.log           rotating planner diagnostics, 2 MB plus two backups
    brain-startup.log   current Python startup failures, if any
    camera-viewer.json  host return position/game mode while cinematic viewing is active
```

SQLite retains at most 20,000 decisions, 2,000 observed lessons and 10,000
situation/action lookup rows. Observed successes/failures are stored separately
from the model's speculative claims, so "I crafted a tool" is not treated as
proof that crafting succeeded. Player inventory, position, and
respawn data are saved in Minecraft 26.3's normal `players/data` files.

### Permanent memory (`brain/memory.py`)

Willow never forgets what it did or saw in a world. Every decision shows the
relevant part of this memory back to the model:

- **Placed blocks** (up to 100,000): every successful `place` stores the exact
  block position and id. Placements within 24 blocks are re-checked against the
  real world each turn and marked `present`, `gone`, or `replaced:<id>`; mining
  its own block marks it gone. The 24 nearest are shown with distances.
- **Waypoints** (up to 256): `first_spawn` and `last_position` are automatic;
  Willow saves named spots itself through the decision's `waypoint` field (for
  example `home`, `iron cave`), or when you ask ("save this spot as home").
  Saving the same name again moves it. A far `goto` (more than 24 blocks) walks
  in that direction instead of failing, so waypoints can be travelled back to.
- **Notes** (up to 2,000): anything the model writes in `memory`, de-duplicated.
- **Knowledge**: per action/target attempts, successes, damage taken and items
  gained (for example `mine minecraft:stone: 5/6 ok, gains cobblestone`).
- **Remembered resources** (up to 20,000): the nearest known location of every
  non-filler block kind ever seen (ores, logs, crafting tables, chests...), shown
  when it is no longer in view.
- **Discoveries and events**: first time each item was obtained, waypoint saves,
  blocks that disappeared.

`memory.json` is a human-readable copy; the SQLite database remains the source
of truth. Memory starts accumulating from the first session with this version;
older lessons and lookup rows are kept.

This is **persistent experience and environment memory**, not neural-network
training, weight updates, or guaranteed improvement. Back up the whole world
while it is closed to preserve the player and its brain together.

Startup prompts live in `brain/server.py` (`SYSTEM` and `PLAN_SYSTEM`).
Primitive game actions live in `src/main/java/local/minecraft/ai/Actions.java`;
the restricted skill interpreter lives in `brain/skills.py`, and observed lookup
learning lives in `brain/experience.py`. Adding new primitive
capabilities still requires reviewed code and a rebuild. The model can write
restricted skill programs, not arbitrary Python or shell commands. All packaged
Python modules (including `memory.py`, `modelprofile.py` and `monitor.py`) are extracted to `~/.minecraft/config/localai/runtime/` on startup.

## Configuration

On first world startup the mod creates:

`~/.minecraft/config/localai/config.json`

```json
{
  "enabled": true,
  "allowDedicatedServer": false,
  "model": "gemma4:12b-it-qat",
  "thinkIntervalTicks": 0,
  "chooseModelOnStart": true,
  "cinematicCamera": true,
  "stopOllamaOnExit": true,
  "modelSettings": {
    "gemma4:12b-it-qat": {
      "contextLength": 16384,
      "actorTokens": 1024,
      "planTokens": 4096,
      "temperature": 0.2,
      "strategistTemperature": 0.3,
      "think": "auto",
      "roles": {
        "actor": { "think": "off", "num_predict": 1024 },
        "strategist": { "think": "on", "num_predict": 2048, "temperature": 0.3 }
      }
    },
    "llama3:8b": { "contextLength": 8192, "actorTokens": 1024, "think": "off" }
  }
}
```

`modelSettings` is optional; leave it empty for full auto-detection. Any field may
be omitted. Supported fields include:
- `contextLength`: shared context window (num_ctx) for both actor and strategist. Both roles always share this single window to avoid model reload thrashing in Ollama.
- `actorTokens`: maximum tokens for actor generation (defaults to ~1024 with thinking disabled).
- `planTokens`: maximum tokens for strategist generation (defaults to 2048-4096).
- `temperature`: sampling temperature for actor decisions.
- `strategistTemperature`: separate sampling temperature for strategist script writing (default 0.3).
- `think`: "auto", "on", or "off".
- `roles`: optional fine-grained per-role dictionary (`roles.actor` and `roles.strategist`), configuring role-specific `think`, `num_predict`, `temperature`, and `timeout`.
- `bounded_repetition`: boolean (default `false`). When enabled, permits bounded loops (`for _ in range(N):` up to 4) and returning action lists under strict AST validation.

### Multi-Slot & Parallel Inference (`OLLAMA_NUM_PARALLEL=2`)

When running Ollama with parallel inference (`OLLAMA_NUM_PARALLEL=2`), Ollama allocates two KV cache slots for concurrent actor and strategist requests:
- VRAM needed per slot: $\text{context} \times \text{KV bytes per token}$.
- At 16K context with Q8_0 KV cache, each slot uses ~500-1000 MiB VRAM. With two parallel slots on a 12 GB GPU (such as RTX 4070 Ti), total KV cache is ~1.5-2.0 GB, leaving plenty of headroom for model weights (~7 GB) and Minecraft Iris shaders (~3.2 GB).
- Note: Ollama system-wide environment variables (`OLLAMA_NUM_PARALLEL=2`) must be set in your systemd service or shell profile; Willow never alters system-wide Ollama settings automatically.

Use the model picker or `/STOP` then `/START <model>` to change models without
quitting. For manual file edits, close the world first. Set `chooseModelOnStart`
to `false` to restore fully unattended startup with the saved model, and
`cinematicCamera` to `false` to leave your player/camera alone.
To reduce GPU memory, choose `qwen3.5:4b`. Models must already be downloaded locally; model
downloads are not attempted in the gameplay loop. Dedicated servers require
explicit opt-in, and `/brain` then requires gamemaster/operator permission.
`/START` and `/STOP` use the same permission rule.
Install server dependencies on that server and comply with Minecraft's EULA.
The mod does not expose an unauthenticated game-control HTTP service: its Python
bridge binds to `127.0.0.1` on a random port and requires a per-process random
bearer token. Ollama is contacted only at `http://127.0.0.1:11434`.

If Ollama stops, the controller reports the error and retries with backoff.
Invalid model actions are rejected, not executed as Minecraft commands.
Controller exceptions pause Willow with an explicit message. Check
`~/.minecraft/logs/latest.log` and the world's `localai/brain.log`.

## Build, install, and reproduce

```bash
cd ~/Projects/minecraft-local-ai
bash scripts/build.sh
python3 scripts/install.py
python3 -m unittest discover -s brain -v
JAVA_HOME="$HOME/.minecraft/runtime/java-runtime-epsilon/linux/java-runtime-epsilon" \
  ./gradlew runGametest
JAVA_HOME="$HOME/.minecraft/runtime/java-runtime-epsilon/linux/java-runtime-epsilon" \
  ./gradlew runLiveBrain
python3 brain/benchmark.py --model gemma4:12b-it-qat \
  --output benchmarks/gemma4-12b-q4_0-32k.json
```

The build uses Java 25 from the existing Minecraft runtime unless `JAVA_HOME`
is set. Dependencies are pinned and verified against upstream checksums.
The installer backs up conflicting JARs under `config/localai/backups`.
For a different launcher instance, use `scripts/install.py --minecraft /path/to/instance`.

`runGametest` runs actual survival crafting, placement, continuous mining,
physical navigation, eating, unsafe-target rejection, and save/reconnect checks.
It also reproduces circular motion detection, leaving a treetop with actual fall
damage, and mining the supporting leaf block when there is no walking escape.
It also checks integrated/dedicated command permissions with Minecraft's
serverless login probe and serializes both command trees to catch login failures.
It verifies uppercase START/STOP permissions, immediate fake-player removal,
inventory-preserving model switches, camera smoothing, local-only model lists,
and spectator return-state recovery.
`bash scripts/build.sh runClientSmoke` launches an isolated client and validates
the actual model-picker widgets and camera mixin loading, then exits automatically.
`runClientCameraSmoke` additionally exercises the startup picker, a real
singleplayer spectator/cinematic handoff, stopping during inference, player
restoration and reopening the picker. It requires a disposable fixture at
`build/client-camera-smoke/saves/camera-smoke` (copy an isolated GameTest world,
never a valuable user save) and a local `qwen3.5:4b` model.
`runLiveBrain` additionally starts the real packaged Python bridge and local
configured default model (Gemma), and requires empty-inventory progression through mining, pickup,
crafting and placing a table, and crafting a wooden pickaxe. The survival test
also reproduces an owned table plus spruce logs/sticks, and airborne item pickup.
Python tests cover prompt budgets, overlapping roles, stale strategies, restricted
code rejection, persistence, skill execution, measured lookup learning, bounded
multi-action plans and failure disabling. The standalone
`benchmark.py` isolates actor decisions with background planning disabled; it does
not measure the full multi-role loop. Test worlds are isolated in `build/`, not your saves.

## Upstream references

- [Fabric 26.3 compatibility and Java/build changes](https://fabricmc.net/2026/09/15/263.html)
- [Carpet source and license](https://github.com/gnembon/fabric-carpet)
- [Carpet 26.3 release](https://modrinth.com/mod/carpet/version/yt9oDFOj)
- [Fabric API 26.3 release](https://modrinth.com/mod/fabric-api/version/bNnaTiuM)
- [Official Qwen3.5 9B model](https://ollama.com/library/qwen3.5:9b-q4_K_M)
- [Official Qwen3.5 4B model](https://ollama.com/library/qwen3.5:4b)
- [Official Gemma 4 12B QAT Q4_0 model](https://ollama.com/library/gemma4:12b-it-qat)
