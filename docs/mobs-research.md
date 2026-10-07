# Mobs research (wave 1: natural zombie spawning + living entities)

Extraction notes for the mobs milestone. Every fact below cites the
decompiled 26.3 reference under `scratch/vanilla-decomp/game/net/minecraft/`.
Engine code carries none of these identifiers.

## 1. The natural spawn cycle

### Chunk selection (`server/level/ServerChunkCache.java`)

- `tickChunks` runs EVERY tick; the spawn state is rebuilt every tick
  (`NaturalSpawner.createState`).
- `doMobSpawning` gamerule gates the whole cycle. `spawnPersistent`
  (gameTime % 400 == 0) only gates persistent mobs; the monster category
  is not persistent, so monsters try every tick.
- Spawning chunks = entity-ticking chunks within `SPAWN_DISTANCE_CHUNK`
  (8) of some player. With simulation-distance 4 that is the 9x9 chunk
  square around each player (81 chunks).
- Global cap (`NaturalSpawner.SpawnState.canSpawnForCategoryGlobal`):
  `maxInstancesPerChunk * spawnableChunkCount / 289` where 289 = 17^2
  (the full square at radius 8). MONSTER `maxInstancesPerChunk` = 70
  (`world/entity/MobCategory.java`); with 81 chunks the cap is
  70 * 81 / 289 = 19 monsters.
- `spawnEnemies=false` (peaceful difficulty) excludes non-friendly
  categories via `getFilteredSpawningCategories`.

### Per-chunk cycle (`world/level/NaturalSpawner.java`)

`spawnCategoryForPosition`, the pack loop:

1. One random start column per chunk attempt:
   `x = minX + rand(16)`, `z = minZ + rand(16)`,
   `y = randomBetweenInclusive(minY, WORLD_SURFACE height + 1)`
   (`getRandomPosWithin`). Abort if the start cell is a redstone
   conductor.
2. Up to 3 groups. Per group: `x += rand(6) - rand(6)`,
   `z += rand(6) - rand(6)` per pack member (the +-6 jitter).
   Group size starts `ceil(rand * 4)`, resampled to the weighted entry's
   count provider once an entry is picked (zombie: uniform 4..4, so 4).
3. Per position, in order:
   - nearest player distance squared must be > 576 (24 blocks,
     `MIN_SPAWN_DISTANCE`), and the position must be inside the
     spawning chunk or another spawnable chunk
     (`isRightDistanceToPlayerAndSpawnPoint`); also >= 24 blocks from
     the world respawn point.
   - monsters cannot spawn farther than their despawn distance
     (`canSpawnFarFromPlayer` false, despawn distance 128), so spawn
     positions land within 128 blocks of that nearest player.
   - `SpawnPlacements.isSpawnPositionOk` for the zombie's ON_GROUND
     type: the cell below passes `isValidSpawn` (solid top with a safe
     spawn offset), and the spawn cell + the cell above pass
     `isValidEmptySpawnBlock` (no full collision shape, no signal
     source, no fluid, not in the prevent-mob-spawning tag)
     (`world/entity/SpawnPlacementTypes.java`).
   - `Monster.checkMobSpawnRules` = the darkness test (below).
   - `level.noCollision(spawn AABB)`.
4. `clusterSize` caps the whole chunk attempt at
   `mob.getMaxSpawnClusterSize()` (MONSTER: 4).

### The darkness test (`world/entity/monster/Monster.java`)

`isDarkEnoughToSpawn(level, pos, random)`:

1. `skyLight(pos) > random.nextInt(32)` -> fail.
2. `blockLight(pos) > monsterSpawnBlockLightLimit` -> fail. Overworld
   limit is 0 (`data/worldgen/DimensionTypes.java`).
3. `getMaxLocalRawBrightness(pos) <= monsterSpawnLightTest().sample(random)`
   where overworld's test is `UniformInt.of(0, 7)`.
   `getMaxLocalRawBrightness = max(blockLight, skyLight - skyDarken)`.

Check 1 reads the RAW sky light (undarkened: 15 under an open sky
even at midnight), so under open sky it passes only 17/32 of draws.
Check 3 reads the DARKENED brightness. Combined per-attempt pass rate
under open sky at midnight: 17/32 x 4/8 ~ 0.27.

skyDarken comes from the timeline keyframes
(`world/timeline/Timelines.java`): SKY_LIGHT_LEVEL holds 15.0 between
ticks 133..11867 and interpolates down to 4.0 at 13670, holding until
22330. So skyDarken = 0 through the day, ~7 at tick 13000 (dusk), and
11 across 13670..22330 (night). Under an open sky at midnight:
brightness = 15 - 11 = 4, which passes check 3 when the sampled light
test (0..7) is >= 4 (half the draws). During the day brightness is 15
and the test can never pass, so a fresh world at time 0 never sees
monster spawns. `tick 13000` (dusk) also never passes on the surface:
brightness 15 - 7 = 8 exceeds the sample ceiling 7.

MONSTERS_BURN keyframes: false at 12542, true at 23460 - i.e. undead
burn during 23460..12542 (wrapping through 0), dawn to dusk.

### Spawn weights (plains)

`data/worldgen/BiomeDefaultFeatures.java`, plains ->
`commonSpawnWithZombieHorse` -> `monsters(90, 5, 5, 100, false)`:
SPIDER 100 (4,4), ZOMBIE 90 (4,4), ZOMBIE_VILLAGER 5 (1,1),
ZOMBIE_HORSE 5 (1,1), SKELETON 100 (4,4), CREEPER 100 (4,4),
SLIME 100 (4,4), ENDERMAN 10 (1,4), WITCH 5 (1,1). Wave 1 spawns the
zombie slice only; the other entries need entity implementations.

## 2. Living entities on the wire

### Pairing order (`server/level/ServerEntity.java`, `sendPairingData`)

`add_entity` -> `set_entity_data` (non-default values only) ->
`update_attributes` (syncable attributes with live instances, if any) ->
`set_equipment` (only non-empty slots; a bare zombie sends none) ->
passenger/link packets (none).

For a fresh adult zombie with no equipment the only non-default datum
is health: accessor 9, serializer FLOAT. Everything else (living flags
byte 0, particles empty, ambient false, arrows/stingers 0, sleeping pos
empty, mob flags 0, baby false, special type 0, drowned conversion
false) sits at its default.

Metadata accessors (declaration order of `defineId` in
`world/entity/LivingEntity.java` + `Mob.java` + `monster/zombie/Zombie.java`):

| idx | field | serializer |
|-----|-------|-----------|
| 8 | living flags (byte) | 0 |
| 9 | health (float) | 3 |
| 10 | effect particles | 12 |
| 11 | effect ambient | 10 |
| 12 | arrow count | 1 |
| 13 | stinger count | 1 |
| 14 | sleeping pos (optional block pos) | 15 |
| 15 | mob flags (bit 1 no-ai, 2 lefthanded, 4 aggressive) | 0 |
| 16 | baby | 10 |
| 17 | special type | 1 |
| 18 | drowned conversion | 10 |

Serializer ids follow `network/syncher/EntityDataSerializers.java`
registration order: 0 BYTE, 1 INT, 2 LONG, 3 FLOAT, 4 STRING,
5 COMPONENT, 6 OPTIONAL_COMPONENT, 7 ITEM_STACK, 8 BLOCK_STATE,
9 OPTIONAL_BLOCK_STATE, 10 BOOLEAN, 11 PARTICLE, 12 PARTICLES,
13 ROTATIONS, 14 BLOCK_POS, 15 OPTIONAL_BLOCK_POS.

Packet ids (registration order in
`network/protocol/game/GameProtocols.java`, bundle delimiter at 0),
cross-checked against the repo pins 0x01/0x23/0x4e/0x65/0x7f:
set_equipment 0x68, hurt_animation 0x2b, damage_event 0x19,
entity_event 0x22, rotate_head 0x55, update_attributes 0x86,
move_entity_pos 0x36, move_entity_pos_rot 0x37, move_entity_rot 0x39,
set_health 0x6a, animate 0x02. Entity type registry: ZOMBIE = 154
(same generator-order source as the repo's ITEM = 72).

### Per-tick sync (`ServerEntity.sendChanges`)

Zombie: updateInterval 3 (EntityType default; PLAYER pins 2 explicitly),
clientTrackingRange 8 chunks (128 blocks) - `world/entity/EntityTypes.java`.

The sync block runs when `needsSync || tickCount % interval == 0 ||
entityData dirty`:

- Movement packet selection (`createMovePacket`): a full
  `entity_position_sync` (0x23) when precise positions are required,
  the teleport delay exceeds 400, the entity was riding, or the
  on-ground flag flipped; otherwise a delta packet when the position
  moved (delta^2 >= 7.6293945e-6) or tickCount % 60 == 0:
  `move_entity_pos_rot` (0x37) if the packed yaw/pitch byte changed,
  `move_entity_pos` (0x36) otherwise; `move_entity_rot` (0x39) when
  only rotation changed; nothing otherwise. Deltas are 1/4096-block
  shorts relative to the last synced position.
- Rotation counts as changed when the packed-degree byte differs by >= 1.
- `rotate_head` (0x55) when the packed head-yaw byte changes by >= 1.
- Dirty entity data sends `set_entity_data`, then `update_attributes`
  for attributes marked to sync (values AND modifiers - the aggressive
  flag flips mob-flags accessor 15 to 4 when a zombie acquires a
  target).
- Zombies do not track deltas (`trackDelta` false), so no
  `set_entity_motion`.

### Death (`world/entity/LivingEntity.java`)

`die` -> `broadcastEntityEvent(3)` (entity_event 0x22 with byte 3) and
pose DYING (accessor 6, serializer 1). `tickDeath` increments
deathTime; at 20 it sends `broadcastEntityEvent(60)` and removes the
entity. The tracker then sends `remove_entities`.

## 3. Damage, health, knockback

`hurtServer(level, source, damage)` (`world/entity/LivingEntity.java`):

- Dead or invulnerable -> nothing.
- `noActionTime = 0` (any hit refreshes the despawn clock).
- Invulnerability cooldown: when `damageCooldownTime > 10`, only the
  damage beyond `lastHurt` applies (the partial-damage rule) and no
  animation replays; otherwise `lastHurt = damage`,
  `damageCooldownTime = 20`, apply, `hurtTime = hurtDuration = 10`.
- Full damage only: `broadcastDamageEvent` (damage_event 0x19),
  `markHurt` (hurt_animation 0x2b), default knockback:
  `knockback(0.4, ...)` from the attacker direction.
- Health <= 0 -> death sound + `die` (the entity_event-3 flow above).

`knockback(power, xd, zd, ...)`: power *= 1 - knockback resistance;
`delta = (delta.x / 2 - dir.x * power, onGround ? min(0.4, delta.y / 2
+ power) : delta.y, delta.z / 2 - dir.z * power)`.

`doHurtTarget` (`world/entity/Mob.java`): damage = ATTACK_DAMAGE (3.0
for the zombie) via a mob-attack damage source, then the target's
knockback. Melee reach (`isWithinMeleeAttackRange`):
DEFAULT_ATTACK_REACH = sqrt(2.04) - 0.6 ~ 0.83, used as the
horizontal inflation of the attacker's AABB tested against the
target's hitbox.

## 4. Zombie AI

Goal registration (`world/entity/monster/zombie/Zombie.java`,
`registerGoals` + `addBehaviourGoals`):

- behaviour: 2 spear use, 3 ZombieAttackGoal(1.0, false), 6
  MoveThroughVillage, 7 WaterAvoidingRandomStroll(1.0)
- goals: 4 turtle egg trample, 8 LookAtPlayerGoal(Player, 8.0), 8
  RandomLookAroundGoal
- targets: 1 HurtByTarget, 2 NearestAttackableTarget(Player,
  mustSee=true), 3 villager/golem, 5 turtle

Attributes (`createAttributes` + `Mob.createMobAttributes` +
`LivingEntity.createLivingAttributes`): FOLLOW_RANGE 35.0,
MOVEMENT_SPEED 0.23, ATTACK_DAMAGE 3.0, ARMOR 2.0,
SPAWN_REINFORCEMENTS 0. ARMOR/ATTACK_DAMAGE/FOLLOW_RANGE/
SPAWN_REINFORCEMENTS are not client-syncable
(`ai/attributes/Attributes.java`); of the syncable set only the
instances the constructor touched exist at pairing time (MAX_HEALTH
via `setHealth(getMaxHealth())` = 20.0). The attribute registry is
alphabetical: max_health = id 23, movement_speed = 26, gravity = 18,
step_height = 33.

Dimensions: 0.6 wide, 1.95 tall, eye 1.74, MONSTER category,
`notInPeaceful`.

Goal selector cadence (`world/entity/ai/goal/GoalSelector.java` +
`Mob.serverAiStep`): full cleanup/update tick on ticks where
`(tickCount + entityId) % 2 == 0` (or the first two ticks); other
ticks only tick running goals that require every-tick updates.
Priorities preempt per flag (MOVE, LOOK, JUMP, TARGET): a running
goal holds its flags until it stops or a higher priority goal takes
them. `adjustedTickDelay(t) = ceilDiv(t, 2)`,
`reducedTickDelay(t) = ceilDiv(t, 2)`.

Server AI step order (`Mob.serverAiStep`): sensing -> selectors ->
navigation -> mob-specific -> move control -> look control -> jump
control.

Melee attack (`ai/goal/MeleeAttackGoal.java`; ZombieAttackGoal wraps
it with no extra behavior): flags MOVE+LOOK; canUse gate: a check at
most every 20 game ticks, needs a live target plus a path or melee
range; canContinueToUse: target alive and navigation not done (the
zombie's false follow-even-if-not-seen); start: navigate to target,
aggressive flag on (mob-flags bit 4); tick: look at target (30, 30),
re-path when the target moved >= 1 block or 5% of ticks, counter
`4 + rand(7)` (+10 beyond 32 blocks, +5 beyond 16) halved by
adjustedTickDelay; attack when the cooldown (adjustedTickDelay(20) =
10 ticks) expired, in reach, and line of sight holds.

Random stroll (`ai/goal/RandomStrollGoal.java` + the zombie's
water-avoiding wrapper): flags MOVE; canUse blocked while
noActionTime >= 100; fires when `rand(reducedTickDelay(120)) == 0`
(1/60 per full check, checks every 2 ticks); target = a random
walkable position within 10 blocks horizontal / 7 vertical.

Look at player (`ai/goal/LookAtPlayerGoal.java`): flags LOOK; 2%
chance per check, range 8, look time adjustedTickDelay(40 + rand(40)),
steers the look control at the player each tick.

Random look (`ai/goal/RandomLookAroundGoal.java`): flags MOVE+LOOK
(the MOVE flag lets it steal the walk lock), 2% chance, 20 + rand(20)
ticks at a random horizontal direction.

Nearest attackable target (`ai/goal/target/NearestAttackableTargetGoal.java`
+ `TargetGoal.java`): flags TARGET; canUse fires 1/reducedTickDelay(10)
= 1/5 of full checks; picks the nearest player within FOLLOW_RANGE
(35) that passes line of sight (mustSee); canContinueToUse drops the
target when it leaves the follow range or stays unseen longer than
reducedTickDelay(unseenMemoryTicks=60) = 30 unseen ticks.

Look control (`ai/control/LookControl.java`): after a setLookAt the
head yaw chases the wanted yaw at the given speed for 2 ticks, pitch
likewise; otherwise head yaw chases the body yaw at 10/tick; the head
is clamped within 75 of the body. The body yaw itself lerps toward
the entity yaw (step 0.3, clamp 50) in `tickHeadTurn`.

Move control (`ai/control/MoveControl.java`): MOVE_TO turns the body
yaw toward the wanted position at up to 90/tick and sets
`speed = speedModifier * MOVEMENT_SPEED` (which for mobs also sets the
forward input `zza`, `Mob.setSpeed`); it requests a jump when the
wanted position is above the step height or the current cell has a
collision shape above the feet.

## 5. Ground movement physics

`LivingEntity.travelInAir` + `handleRelativeFrictionAndCalculateMovement`
+ `Mob.setSpeed` + `ai/control/MoveControl.java`:

Per tick on ground (flat grass, slipperiness 0.6):

1. input accel: `zza = speedModifier * 0.23`; the friction-influenced
   scale on grass is exactly `getSpeed()` (the > 0.6 slipperiness
   branch with 0.216/f^3 does not apply to 0.6), so the added accel is
   `zza * getSpeed() = (0.23 * modifier)^2` = 0.0529 at modifier 1.0,
   aimed along the body yaw.
2. move by delta.
3. `vy -= 0.08` (gravity attribute).
4. horizontal delta *= 0.6 * 0.91 = 0.546; vertical *= 0.98.
5. components below 0.003 clamp to 0.

Jump: `jumpFromGround` sets `vy = max(jumpPower, vy)` (0.42 base,
jump_strength attribute), and the jump control cooldown holds for 10
ticks. Ground pathing step-up covers 0.6; full blocks need the jump.

Steady-state walk speed = a * f / (1 - f) = 0.0529 * 0.546 / 0.454
~ 0.064 blocks/tick (1.3 blocks/s) at modifier 1.0 - the speed the
chase check sees, order-of-magnitude, not a gate assertion.

## 6. Despawn (`Mob.checkDespawn`)

- Peaceful difficulty and the type is not allowed there -> instant
  discard (zombies are `notInPeaceful`).
- Nearest player beyond 128 (despawn distance) -> instant discard.
- noActionTime: +1 per server AI tick (`Mob.serverAiStep`), +2 more
  when the local brightness ratio `max(blockLight, skyLight -
  skyDarken)/15 > 0.5` (`Monster.updateNoActionTime`), reset to 0 by
  any hit or by a player within 32 (no-despawn distance). When it
  exceeds 600, a 1/800 per tick roll discards the mob while the
  nearest player is beyond 32.

## 7. Daylight burning (`Mob.burnUndead` + `isSunBurnTick`)

Every tick for burn-in-daylight types (zombie is in the tag): when
the MONSTERS_BURN timeline says burn, the brightness ratio br > 0.5,
`rand * 30 < (br - 0.4) * 2` (4% per tick at full daylight), and the
eyes can see the sky, ignite for 8 seconds (320 fire ticks; fire
damage 1.0 every 20 ticks; entity flags bit 0x01 carries the flame to
clients). Approximation: no light engine, so brightness comes from
the heightmap-style sky-exposure scan + the timeline skyDarken; rain
and water do not exist yet on this build's flat world.

## 8. Doppel architecture (wave 1)

New modules under `crates/doppel/src/` (boundaries from the plan):

- `living.rs` - packet id pins, entity type id, metadata
  accessor/serializer table, attribute ids, the LivingEntity base
  (health, hurt timing, knockback, death event + 20-tick removal,
  despawn checks, ground physics step), the tracker-side pairing and
  per-tick sync encoder selection (0x23/0x36/0x37/0x39 + rotate_head),
  and `trait MobBehavior` (per-mob module, no mob enum).
- `zombie.rs` - `MobBehavior` for the zombie: goals at their reference
  priorities on a flag-preempting goal selector, the player target
  rule, daylight burning, the attribute set (speed 0.23, attack 3.0,
  follow 35).
- `pathing.rs` - straight-line movement to a wanted position with a
  1-block step-up/jump when blocked; A* lands in wave 2.
- `spawning.rs` - the monster-category cycle from section 1: global
  cap, per-chunk random column + pack jitter, distance/ground/darkness
  tests, `spawn_mobs` + difficulty gating, deterministic RNG streams.

Integration: `game_tick` gains a natural-spawn phase immediately
before the random-tick pass (the reference runs both inside the chunk
tick, before the broadcast point); mobs tick in the existing entity
pass; the tracker grows a mob half (per-mob track range 128, delta
movement encoding at interval 3, dirty-data sends).

Wire: track-in sends add_entity -> set_entity_data (health only) ->
update_attributes (max_health 20.0); per-tick delta movement +
rotate_head; hurt sends damage_event + hurt_animation; death sends
entity_event 3, entity_event 60 + remove_entities 20 ticks later.

Deviations from the reference, documented in code:

- No light engine: sky exposure via the existing heightmap-style
  column scan (exact on the flat world); block light reads as 0
  except where a torch column would matter (none in the gate area).
- No player health model: the zombie's attack sends damage_event +
  hurt_animation but no set_health/knockback for the player yet.
- Pathing is straight-line + step-up, not A*; MoveThroughVillage and
  the water-avoiding stroll variants collapse into the plain stroll.
- Only the zombie spawns; the other plains monster weights wait for
  entity implementations (wave 2).
- Mob AI ticks regardless of the entity-ticking chunk range (doppel
  spawns only within 128 blocks of a player anyway).

Gate (`crates/doppel-oracle/src/parity_mobs.rs`): boot vanilla with
default difficulty (easy) on the flat config, `time set midnight`,
`gamerule spawn_mobs true` (the default; explicit for clarity), a
stationary opped bot waits for add_entity of the zombie type at
distance 24..128, then structural assertions: the type id, the
metadata accessor/serializer set on the wire, the living packet ids
present, and a chase-distance decrease across N position packets.
Doppel boots identically (blobs + pristine world from the clean
capture) minus the vanilla jar. Structural, never exact-random.

Consult-3 refinements folded into the gate design:

- Chase needs the player inside follow range 35: a zombie that
  spawned 40+ blocks out never targets the bot. After the first
  zombie add_entity lands, the bot tp's to a standing position
  24..30 blocks from it, and the distance-trend assertion samples
  from the movement sync packets (which arrive every 3 ticks), so
  the trend check compares across sync events, not raw ticks.
- Structural spawn invariants asserted for every observed monster
  add_entity: x/z at block center +0.5, integer feet y, 3D distance
  from the bot within 24..128, zero initial velocity.
- The vanilla side spawns every plains monster, not just zombies
  (~18% of packs are zombies); the gate reports any-monster and
  zombie sightings separately so a no-zombie failure names its
  cause.
- Noon and peaceful negative controls stay unit tests on the
  doppel side (the darkness math is deterministic); the live gate
  does not pay their wall-clock cost.
- The spawner consumes its RNG at the reference draw points (the
  17/32 check-1 draw included); drawing once per tick or per pack
  would keep the mean but break the distribution.

## 9. Consult adjudications

Three escalation consults (keypool purpose mobs-research-1/2/3).

1. Goal system: choose (C), the reference selector architecture at
   wave-1 scope. Priority-ordered goals with flag locks (MOVE, LOOK,
   JUMP, TARGET), two selectors per mob (behavior + target), full
   re-evaluation every 2nd tick offset by entity id, replaceable =
   lower priority number, equal priority never preempts, stop ->
   release flags. A hardcoded if/else behavior function would pass
   the structural gate and then force a rewrite when wave 2 adds
   mob types: goals with disjoint flags must run concurrently
   (look-at-player while strolling), the preemption rules would be
   re-derived per mob, and A* path ownership needs goal lifetime.
   Seams to keep wave 2 additive: a navigation trait (move_to,
   is_done, stop) with the straight-line walker as the first impl;
   goals receive a context (world view + mob state + RNG), never
   own the mob; selector unit tests for preemption, equal-priority
   holds, flag release, off-tick cadence.
2. Sync cadence: per-tick physics + interval-3 delta sync (the
   reference behavior), not the plan's literal per-tick sends.
   Physics constants are per-tick (gravity, friction, jump); a
   3-tick physics step would distort trajectories nonlinearly.
   The sync base must be the quantized value the client
   reconstructs (delta = round(new*4096) - round(base*4096), base
   advances by the quantized amount) so rounding never drifts;
   full position_sync on out-of-i16 deltas, on-ground flips, and
   the 400-tick resync; idle mobs send nothing; metadata and
   attributes resend on dirty, not on the interval.
3. Spawn determinism: per-subsystem private RNG is sound (vanilla's
   own sequence is irreproducible across runs; a shared seed buys
   nothing). Draw granularity beats seed choice. The consult
   corrected the darkness math (check 1 reads raw sky light, 17/32
   acceptance; combined ~27% per attempt at midnight - section 1)
   and identified follow-range as the chase-assertion hazard the
   gate now handles by tp'ing the bot inside 35 blocks (section 8).

Open wire TODOs until the gate observes vanilla: zombie type id 154,
attribute registry ids (max_health 23), the packet ids 0x68/0x2b/
0x19/0x22/0x55/0x86/0x39, and the exact update_attributes entry set
at pairing (predicted: only max_health 20.0). (Wave 3 update: all of
these are now wire-verified; see sections 22-25. The pairing rule
that held: the mob snapshots carry movement_speed alone - max_health
never rides them, even the spider's off-default 16.0.)

# Wave 2 research: skeleton, creeper, spider, arrows, explosions

Same rules as wave 1: facts below come from the decompiled 26.3
reference under `scratch/vanilla-decomp/game/net/minecraft/`; engine
code carries none of these identifiers. Constants land in engine code
as plain numbers with the value in a short present-tense comment.

## 10. Wire ids and registry ids

### Packet ids (verified method)

The clientbound play ids follow the registration chain in
`network/protocol/game/GameProtocols.java`: the bundle delimiter
holds id 0, then each `addPacket` in chain order takes 1, 2, 3, ...
Counting that chain reproduces every pinned id in this repo
(add_entity 0x01, block_update 0x08, damage_event 0x19,
entity_position_sync 0x23, set_entity_data 0x65, remove_entities
0x4e, rotate_head 0x55, section_blocks_update 0x56,
set_entity_motion 0x67, set_equipment 0x68, update_attributes 0x86,
player_position 0x49, take_item_entity 0x7f, set_time 0x73,
level_chunk_with_light 0x2e - 20+ anchors, zero misses), so the
method is sound. New for wave 2:

- explode: 0x24 (registration 36)
- set_equipment: 0x68 (registration 104; pinned in wave 1)

### Entity-type registry ids

Counted the same way as ZOMBIE_TYPE = 154: 0-based index into the
`world/entity/EntityTypeIds.java` creation order (item = 72 and
zombie = 154 cross-check).

| type | id |
|------|----|
| arrow | 6 |
| creeper | 32 |
| skeleton | 118 |
| spider | 127 |

### Item registry ids

The repo's curated item table (`crates/doppel/src/inventory.rs`)
already carries `minecraft:bow` = 1008 and `minecraft:arrow` = 1009;
both were pinned there from the captured registries and survive the
survival gate, so they are the source of truth for wave 2.

## 11. Metadata indices and serializers

The accessor index continues the parent class count: base entity 8
accessors (0..7), living 7 more (8..14, health at 9), mob 1 more
(15, the mob flags byte). Counting the `defineId` declarations in
each subclass:

| mob | accessor | serializer | default | meaning |
|-----|----------|-----------|---------|---------|
| creeper | 16 | 1 (INT) | -1 | swell direction: -1 shrinking, 1 swelling |
| creeper | 17 | 10 (BOOLEAN) | false | powered (charged by lightning) |
| creeper | 18 | 10 (BOOLEAN) | false | ignited (flint and steel) |
| spider | 16 | 0 (BYTE) | 0 | bit 0x01 = climbing |
| skeleton | 16 | 10 (BOOLEAN) | false | freezing conversion |

Defaults mean the pairing data carries health alone for all three
mobs (the creeper's swell starts at -1, its default); the swell and
climb entries go out later as dirty data when they flip.

Arrow metadata (accessors after the base 8): flags byte at 8 (bit 1
critical, bit 2 no-physics), pierce level byte at 9, in-ground
boolean at 10, tipped-arrow color int at 11.

## 12. The set_equipment packet

Layout (`ClientboundSetEquipmentPacket`): entity id varint, then a
list of (slot byte, item stack) pairs. The slot byte carries the
slot ordinal in bits 0..6 and 0x80 set on every entry except the
last. Slot ordinals (`world/entity/EquipmentSlot.java` declaration
order): main hand 0, offhand 1, feet 2, legs 3, chest 4, head 5,
body 6, saddle 7. The item stack is the same optional-stack encoding
the item entity uses (count varint; when nonzero: item id varint,
added-components count varint, removed-components count varint).

Pairing order (`server/level/ServerEntity.sendPairingData`):
add_entity, set_entity_data (non-default values), update_attributes
(syncable attributes with live instances), set_equipment (non-empty
slots in slot-ordinal order), passengers, leash. A skeleton pairs
health 20.0 (accessor 9 FLOAT), movement_speed 0.25 (attribute id
26), then set_equipment with exactly one entry: byte 0x00 (main
hand, no continuation bit) then the bow stack (count 1, item 1008,
0, 0).

## 13. Skeleton

### Type and attributes

0.6 wide, 1.99 tall, eye 1.74, tracking range 8 chunks, sync
interval 3, monster category, not peaceful. Attributes:
movement_speed 0.25, attack_damage 2.0 (the monster default), max
health 20, follow_range 32 (the default; the zombie's 35 is a
zombie override). Undead: burns in daylight exactly like the zombie
(same timeline window, same brightness ratio and roll).

### Goal table

| priority | goal | flags | notes |
|----------|------|-------|-------|
| 2 | restrict sun | MOVE | no light engine; omitted |
| 3 | flee sun | MOVE | no light engine; omitted |
| 3 | avoid wolves | MOVE | no wolves; omitted |
| 4 | ranged bow attack | MOVE+LOOK | interval 20 hard / 40 otherwise, radius 15 |
| 5 | water-avoiding stroll | MOVE | speed 1.0 |
| 6 | look at player | LOOK | range 8 |
| 6 | random look | MOVE+LOOK | |
| target 1 | hurt-by | TARGET | omitted (no player attacks on mobs yet) |
| target 2 | nearest player, must see | TARGET | range 32 |

There is no float goal on the skeleton (the zombie has none either);
swimming is the navigation's canFloat flag. The bow goal is added by
the weapon reassessment at construction: holding a bow registers the
ranged goal at 4.

### Ranged bow attack

Every tick (requires-every-tick):

- If squared distance > 225 (15^2) or seen ticks < 20: pathfind to
  the target at speed 1.0; strafe counter resets to -1.
- Else stop the navigation and count strafe ticks. Every 20 strafe
  ticks each of clockwise and backwards flips with 30% chance.
- Strafe magnitudes 0.5 forward, 0.5 sideways at speed factor 0.25
  (the strafe move-control path). Backwards (forward -0.5) whenever
  squared distance < 225 * 0.25 = 56.25 (inside 7.5 blocks);
  forwards again beyond 225 * 0.75 (13.0 blocks).
- Draw: after the attack cooldown (20 ticks hard difficulty, 40
  otherwise) expires with the target seen (seen ticks >= -60), start
  drawing; release after 20 drawn ticks. Power at release = 1.0
  (the charge curve saturates: t/20 -> (t^2+2t)/3 = 1 at t=20).
  Release fires the arrow and resets the cooldown.
- Line of sight bookkeeping: seen ticks +1 with sight, -1 without;
  stop drawing at seen ticks < -60.

### The shot

Arrow spawns at the skeleton's eye Y minus 0.1 (feet + 1.74 - 0.1).
Aim: horizontal delta to the target, vertical delta to the target's
one-third height (feet + 0.333) plus horizontal distance * 0.2 (the
gravity lead). Launch speed 1.6, uncertainty 14 - 4 * difficulty id
(easy 10): each velocity component gains a triangle(0, 0.0172275 *
uncertainty) jitter before the speed scaling. Arrow base damage =
power * 2.0 + triangle(0.11 * difficulty id, 0.57425) (mean 2.11 on
easy); a hit deals ceil(speed * base damage).

## 14. Arrow flight

Per tick (`world/entity/projectile/arrow/AbstractArrow.java`):

1. Move by the current velocity, clipped against block collision
   along the segment.
2. Air drag: velocity *= 0.99 (water: 0.6 horizontal).
3. Gravity: velocity.y -= 0.05, after the drag.
4. Rotation follows the velocity vector (atan2).

Block hit: the arrow stops dead (velocity zero), pulls back 0.05
along each movement sign, marks in-ground, then despawns after 1200
ticks. Entity hit: damage ceil(length * base damage) via the arrow
damage type (alphabetical registry id 0; mob_attack's 28 is the
same counting), knockback 0.4 along the arrow's horizontal flight
direction, then the arrow discards. On the wire: add_entity carries
the owner entity id in the trailing data varint (the skeleton's id;
0 when ownerless); movement syncs at interval 20 with a 4-chunk
tracking range; arrows track deltas (set_entity_motion when
velocity changes).

Type facts: 0.5 wide and tall, MISC category, tracking range 4
chunks, update interval 20, no loot.

## 15. Creeper

### Type and attributes

0.6 wide, 1.7 tall (eye 1.445, the 0.85 default), tracking range 8,
interval 3, monster, not peaceful. movement_speed 0.25, attack
damage 2.0 (never used: the melee goal exists only to close
distance), max health 20, follow range 32.

### Goal table

| priority | goal | flags | notes |
|----------|------|-------|-------|
| 1 | float | JUMP | inert without fluids |
| 2 | swell | MOVE | stops navigation; drives the fuse |
| 3 | avoid ocelots/cats | MOVE | no cats; omitted |
| 4 | melee approach | MOVE+LOOK | closes distance, deals no damage |
| 5 | water-avoiding stroll | MOVE | speed 0.8 |
| 6 | look at player | LOOK | range 8 |
| 6 | random look | MOVE+LOOK | |
| target 1 | nearest player, must see | TARGET | range 32 |
| target 2 | hurt-by | TARGET | omitted |

### Swell and fuse

The swell goal takes MOVE and stops the navigation when active.

- Starts (canUse) when the swell direction is already positive or
  the target is alive within squared distance 9.0 (3 blocks).
- Each tick (every tick): target gone or dead -> direction -1;
  squared distance > 49.0 (7 blocks, the cancel range) -> -1; no
  line of sight -> -1; else direction +1.
- The fuse counter (swell) moves by the direction each tick, floor
  0, and the explosion fires when it reaches 30 (the max swell; the
  reference "fuse" is 30 ticks). A fall adds up to
  (fall distance * 1.5) capped at 25.
- Metadata: accessor 16 (INT) carries the direction value itself
  (-1 / 0 / 1), not the counter. It flips to 1 when the swell
  starts, back to -1 when the target escapes.

### Explosion

At fuse end the creeper explodes with radius 3 (powered: 6) as a mob
explosion and discards immediately (no death animation, no corpse:
the entity is dead=true and removed in the same tick; clients see
remove_entities). With mob_griefing true (default) and
mob_explosion_drop_decay false (default) the block interaction is
plain destroy.

## 16. Explosion mechanics

### Block destruction

Ray grid (`world/level/ServerExplosion.calculateExplodedPositions`):
a 16x16x16 grid where only the surface cells cast rays (any
coordinate 0 or 15): 16^3 - 14^3 = 1352 rays.

Per ray:

- Direction: cell / 15 * 2 - 1 per axis, normalized.
- Starting power: radius * (0.7 + random * 0.6).
- Step length 0.3; power decays 0.22500001 per step.
- Per cell entered: power -= (resistance + 0.3) * 0.3 where
  resistance is the block's explosion resistance (fluids count when
  present; air and empty fluid skip the resistance read entirely).
  If power > 0 after the read, the cell joins the destroy set (the
  creeper path has no per-block veto).
- The ray ends when power <= 0.

Resistance values come from the block's `strength`: the builder
default is 0.0, the one-argument form sets destroy time and
explosion resistance to the same number, and the two-argument form
splits them (grass_block 0.6, dirt 0.5, stone 1.5/6.0, bedrock
-1/3.6e6). On the flat world the floor is grass at 0.6, so a floor
cell costs (0.6 + 0.3) * 0.3 = 0.27 power; air cells cost only the
0.225 per-step decay. With no block-resistance pin in this repo,
wave 2 approximates: air 0.0, every other block 0.6 (the flat
world's floor value); the approximation is documented here and the
gate asserts structure, not exact crater shape.

Destroy set -> block updates: the reference shuffles the set, drops
items (this build drops nothing for explosions), sets air. The wire
carries block_update / section_blocks_update exactly like setblock
(0x08 single, 0x56 per section per tick).

### Entity damage and knockback

Entities within radius * 2 of the center (AABB test):

- dist = sqrt(distanceToSqr(center)) / (radius * 2); skip when > 1.
- exposure = seen percent: sample points on a 2x-per-block grid
  over the entity's box ((2w+1) x (2h+1) x (2d+1) points, offset
  half a step on x and z); a point counts when the clip to the
  center hits no collider. exposure = hits / count. On the flat
  world an unobstructed player reads 1.0.
- damage = (p^2 + p) / 2 * 7 * (radius * 2) + 1 where
  p = (1 - dist) * exposure. Radius 3 point blank: about 24.
- knockback power = (1 - dist) * exposure * 1.0 (the living
  multiplier; explosion knockback resistance is a separate 0.0
  attribute), direction = normalize(eye position - center), applied
  as a velocity push. Survivors take the damage event. The damage
  type: the mob-explosion source carries a direct and an indirect
  cause entity; when both exist the registry entry is the
  entity-attribution one. Alphabetical damage-type ids (the same
  ordering that gives mob_attack 28): arrow 0, explosion 9,
  mob_projectile 30, player_attack 34, entity-attributed explosion
  35. TODO wire-verify at the gate.

### The explode packet (0x24)

Fields in order: center (3 doubles), radius (float), destroyed block
count (varint), optional player knockback (boolean, then 3 doubles
when present; only the receiving player's own knockback), explosion
particle (the large-explosion particle when radius >= 2), explosion
sound (a registry holder), the weighted block-particle list (two
entries: poof 0.5, smoke 1.0), play-sound boolean. Sent to players
within 64 blocks of the center. The particle and sound ids are not
pinned in this repo; the gate observes vanilla's bytes and doppel
matches the prefix it can verify (center, radius, count) while the
tail is pinned from the capture during phase 2.

## 17. Spider

### Type and attributes

1.4 wide, 0.9 tall, eye 0.65, tracking range 8, interval 3,
monster, not peaceful. movement_speed 0.3, max health 16, attack
damage 2.0, follow range 32. The spider is the only wave 2 mob
whose max health sits off the 20.0 default, so its pairing
update_attributes carries two entries: max_health (id 23) 16.0 and
movement_speed (id 26) 0.3. Skeleton and creeper carry
movement_speed alone (0.25).

### Goal table

| priority | goal | flags | notes |
|----------|------|-------|-------|
| 1 | float | JUMP | inert without fluids |
| 2 | avoid armadillos | MOVE | no armadillos; omitted |
| 3 | leap at target | JUMP+MOVE | 2..4 blocks, 1/3 roll, vy 0.4 |
| 4 | melee attack (long memory) | MOVE+LOOK | speed 1.0 |
| 5 | water-avoiding stroll | MOVE | speed 0.8 |
| 6 | look at player | LOOK | range 8 |
| 6 | random look | MOVE+LOOK | |
| target 1 | hurt-by | TARGET | omitted |
| target 2 | nearest player, must see | TARGET | only when local brightness < 0.5 |
| target 3 | nearest iron golem | TARGET | no golems; omitted |

Light gating: the target goal refuses while the local
light-dependent brightness >= 0.5; the attack goal, once running,
drops the target on a 1/100 roll per tick while bright. Wave 2
approximates brightness with the sky-exposure + sky-darken model
the zombie's burn uses (dark when the darkened brightness < 0.5);
at midnight under open sky both read hostile.

The leap: fires from the ground at 2..4 blocks (squared 4..16) on a
1-in-3 roll per full check, velocity = horizontal-normalized *
0.4 + 0.2 * current velocity, vy 0.4.

### Climbing

The climbing bit (accessor 16, bit 0x01) sets whenever the spider
horizontally collides, every tick. Climbing spiders climb at
0.2/tick vertical while pressing into a wall (the ladder rule); the
navigation is the wall-climber: when a path ends without reaching
the target column, the move control walks straight at the target
position, and wall contact plus the climb flag carries the spider
up. Wave 2 models this as: the climb capability on the navigator
enables a straight-up neighbor in the A* grid, and the movement
rule gives the body 0.2/tick vertical rise while horizontally
colliding with the climb bit set.

### Spawn footprint

Spawn placement is the same ON_GROUND type as the zombie, but the
no-collision AABB test uses the spider's 1.4-wide box, so a spider
needs the wider clearance (this is the footprint check; the
reference applies it through the generic spawn AABB test).

## 18. Spawn weights (plains, 26.3)

The plains monster list (`data/worldgen/BiomeDefaultFeatures.java`,
plains -> commonSpawnWithZombieHorse -> monsters(90, 5, 5, 100)),
registration order:

| entry | weight | pack size |
|-------|--------|-----------|
| spider | 100 | 4..4 |
| zombie | 90 | 4..4 |
| zombie villager | 5 | 1..1 |
| zombie horse | 5 | 1..1 |
| skeleton | 100 | 4..4 |
| creeper | 100 | 4..4 |
| slime | 100 | 4..4 |
| enderman | 10 | 1..4 |
| witch | 5 | 1..1 |

Total weight 515 (bats and glow squid ride other categories). The
weighted pick draws once in [0, total) and walks the list in
registration order. Wave 2 implements spider 100, zombie 90,
skeleton 100, creeper 100 over a 390 total (the unimplemented
entries keep their reference weights in this table; the deviation -
their weight drops from the draw - is the same shape as wave 1's
zombie-only 90). Light and ground rules are exactly the zombie's
(monster darkness test, ON_GROUND placement); nothing new arrives
with these three types beyond the spider footprint.

## 19. A* navigation seam design (wave 2)

### What stays

The `Nav` seam in `crates/doppel/src/pathing.rs` keeps its public
surface unchanged: `move_to(x, z, modifier)`, `retarget(x, z,
modifier)`, `stop()`, `wanted() -> Option<(x, z, modifier)>`,
`arrived(x, z)`, `in_progress()`, `tick_age(limit) -> bool`. The
move control in `living.rs` still consumes `wanted()` each tick:
face the returned position, walk forward at the modifier, stop when
`arrived`. Nothing outside pathing.rs changes its call shape.

### What moves inside

`Nav` grows an optional path state:

- `path: Option<Vec<(i32, i32, i32)>>` - the waypoint list, integer
  block positions, start exclusive (the first entry is the next
  waypoint).
- `idx: usize` - the index of the current waypoint.
- `climb: bool` - the per-mob capability flag (spider true).
- `blocker: Option<BlockQuery>` - the world read.

`wanted()` returns the CURRENT waypoint center: (x + 0.5, y, z +
0.5) of `path[idx]`, falling back to the raw want when no path
exists (straight-line walking, the wave 1 behavior, when the grid
says no path or the target is one block out).

### The block query

```rust
type BlockQuery<'a> = &'a dyn Fn(i32, i32, i32) -> bool;
```

`solid(x, y, z) -> bool`: true when the cell blocks a walker.
`move_to` and `retarget` take the query as a parameter (the goals
have `ctx.world` in hand), or `Nav` stores it per call: the goals
call `nav.move_to(x, z, m, &|x,y,z| block_solid(ctx.world,x,y,z))`.
Unit tests pass a closure over a `HashMap<(i32,i32,i32),()>` or a
2D grid, no `Game` needed. The search itself lives in a free
function `find_path(query, from: (i32,i32,i32), to: (i32,i32,i32),
climb: bool, budget) -> Option<Vec<...>>` so it is testable in
isolation.

### The grid

A node (x, y, z) is walkable when solid(x, y-1, z) (floor) and
!solid(x, y, z) and !solid(x, y+1, z) (feet and head clear). A
2-tall mob fits a 2-air-gap; this build's mobs are all <= 1.99
tall.

Neighbors, in cost terms:

- 4-directional flat: dx or dz = +/-1, same y, cost 1.0.
- Jump-up-one: neighbor y+1 with (x, y+2, z) and (x, y+3, z) clear
  (headroom over the jump) and the target cell walkable, cost 1.5.
- Drop: y-n for n in 1..=4, the landing cell walkable, every
  intermediate cell non-solid, cost 1.0 + 0.5 * n (the bounded
  fall height 4 covers the flat world's shelves; drops deeper than
  4 are not taken).
- Climb-up (climb flag only): y+1 directly above the current node
  with (x, y+1..y+2, z) passable at the wall face, cost 2.0 -
  used only when the flat/jump neighbors are blocked, modeling the
  spider walking up a wall. A climb node needs a solid wall: the
  cell in the horizontal direction of approach must be solid at the
  node's head height.

Heuristic: manhattan distance + |dy| (admissible against the
neighbor costs above); the search pops by f = g + h. Node store: a
`HashMap<(i32,i32,i32), Node>` with a `BinaryHeap` frontier keyed
by (f, tie-break insertion counter) - deterministic order.

Budgets: a node cap (default 400 expansions, near the reference's
own visited-node budget for a 35-block follow range at ~16 nodes
per block) and a range cap: chebyshev distance from start > 48
prunes. Budget exhaustion returns the best partial path toward the
target (the reference keeps its best node's path when it runs out),
or none when nothing better than the start was reached.

### Waypoint consumption

`wanted()` advances `idx` when the mob's current block column
equals the waypoint's column (the same 0.5-block radius `arrived`
uses, at the waypoint's y +/- 1.5 for step-ups and drops). When
`idx` passes the end, `wanted()` returns none and `stop()` fires:
`arrived()` already handles the "reached the final column" case
and goals fall back on their give-up logic through `tick_age`.

### Recompute triggers

`retarget(x, z, modifier)` recomputes when the new target column
sits more than 1 block from the current path's end (the reference
repaths when the target moved >= 1 block); otherwise it just
replaces the raw want. A periodic recheck every 20 ticks of an
active path (the reference's recompute cadence) re-runs the search
when the path's next node became unwalkable (a block was placed on
the route). Stuck detection: when a path is active and horizontal
progress stays under 0.06 blocks (one walking step) for 40 ticks,
recompute; after 3 consecutive stuck recomputes without progress,
stop the navigation (`in_progress()` false) so goals give up.

### Spider climb flag

`MobKind` gains nothing: the spider's constructor passes
`climb = true` into its `Nav` at spawn (a one-line hook in the
mob's own module, or a `fn can_climb(&self) -> bool` default-false
method on `MobKind` - the fn method, so `spawn_mob` can wire it in
`Mob::new` without the goals seeing it).

### Unit tests (all on hand-built grids, no Game)

1. straight path: flat floor, assert the waypoint list and that
   `wanted()` walks it in order.
2. one-block obstacle: a single raised block between start and
   goal, assert the path hops it (a y+1 node appears).
3. drop: a two-block ledge between start and goal on the far side,
   assert the drop node and its extra cost (the path prefers the
   drop over a long way around).
4. no path: the goal fully enclosed, assert none within budget.
5. recompute after target moves: path to A, retarget to B 3 blocks
   from A, assert a fresh search ran (waypoints changed).
6. recompute after a block placement: path exists, a wall lands on
   the next waypoint, the periodic recheck finds a new route
   around.
7. wall between start and goal: a 5-tall wall with a gap at one
   end, assert the route goes around (no jump nodes).
8. climb path: a 3-tall wall between start and goal; with the
   climb flag the route goes over the top; without it, none within
   budget (or the way around when one exists).

### Wiring (plan steps 3, 8)

The zombie's chase and stroll goals already call `move_to` /
`retarget`; they gain the block-query argument. `living.rs`'s move
control is unchanged. The shared helpers (`visible`, `eye_at`,
`look_angles`) move from `zombie.rs` to `living.rs` as
`pub(crate)` free functions with the eye height passed in; the
zombie, skeleton, creeper and spider modules import them.

## 20. Wave 2 deviations, documented

- No light engine: the spider's hostility gate uses the same
  sky-exposure + sky-darken approximation as the burn check. The
  skeleton's restrict-sun and flee-sun goals are omitted (they need
  block light and a sun direction model); the daylight burn stays.
- No player attacks on mobs: hurt-by target goals stay omitted
  (wave 1 already omits them for the zombie).
- No fluid physics for mobs: float goals are inert; drowning and
  water walking do not exist.
- Explosion drops: mob explosions destroy blocks without item drops
  (the break system's drop path is break-action driven; wiring
  explosion drops is out of wave 2 scope per the plan's file
  boundaries).
- Block resistance: approximated (section 16) until a resistance
  pin exists.
- Arrow pickup: skeletons' arrows are pickup-disallowed (the
  reference default for mob arrows); no ground pickup modeling.
- The skeleton's freeze conversion (stray) accessor stays at its
  default; no powder snow exists.

# Wave 3 research: melee swing, facing, metadata audit, blast values

Facts below come from the decompiled 26.3 reference under
`scratch/vanilla-decomp/game/net/minecraft/` plus the
follow-and-attack capture (section 22).

## 21. Serializer ids: registration order, not declaration order

The static block in `network/syncher/EntityDataSerializers.java`
registers in an order that DIFFERS from the field declaration order
(the wave 1/2 tables above followed the declarations and are wrong
from slot 8 on). The wire ids:

| id | serializer |
|----|-----------|
| 0 | BYTE |
| 1 | INT |
| 2 | LONG |
| 3 | FLOAT |
| 4 | STRING |
| 5 | COMPONENT |
| 6 | OPTIONAL_COMPONENT |
| 7 | ITEM_STACK |
| 8 | BOOLEAN |
| 9 | ROTATIONS |
| 10 | BLOCK_POS |
| 11 | OPTIONAL_BLOCK_POS |
| 12 | DIRECTION |
| 13 | OPTIONAL_LIVING_ENTITY_REFERENCE |
| 14 | BLOCK_STATE |
| 15 | OPTIONAL_BLOCK_STATE |
| 16 | PARTICLE |
| 17 | PARTICLES |
| 18 | VILLAGER_DATA |
| 19 | OPTIONAL_UNSIGNED_INT |
| 20 | POSE |
| 21..=32 | variant holders (cat, cow, wolf, frog, pig, chicken, ...) |
| 33 | OPTIONAL_GLOBAL_POS |
| 34 | PAINTING_VARIANT |
| 35..=38 | sniffer, armadillo, copper golem, weathering states |
| 39 | VECTOR3 |
| 40 | QUATERNION |
| 41 | RESOLVABLE_PROFILE |
| 42 | HUMANOID_ARM |
| 43 | DYE_COLOR |

So BOOLEAN = 8 (engine `SER_BOOLEAN` was already right), BLOCK_POS
10, ROTATIONS 9, POSE 20. Corrections to the tables in sections 2
and 11: living effect-ambient (accessor 11), zombie baby (16) and
drowned conversion (18), creeper powered (17) and ignited (18),
skeleton stray conversion (16) all serialize with id 8 (BOOLEAN).

## 22. The follow-and-attack capture (vanilla-only run)

Scenario: flat world at spawn, bot teleported to (100.5, -60, 100.5),
one zombie summoned 8 blocks east at (108.5, -60, 100.5), clock set
to midnight, frozen, and the bot strafed +2z / -4z / +2z between
80/50/50/100-tick step barriers. The capture records the attack's
wire rhythm.

Observed (first run, daylight variant - the zombie burned, which
doubled as a damage-type probe; second run at midnight, clean):

- 10 swing_animation (0x7b) frames from the zombie for 10 landed
  hits, payload hand0/anim1/dur6 every time: main hand, WHACK,
  6-tick duration, the bare-hand default. The swing lands 1-2 frames
  before each hit's damage_event, every time.
- 0 animate (0x02) frames from the zombie in the whole session; the
  single 0x02 frame in the histogram targets the player (id 1). Mobs
  never send animate.
- Every mob_attack (damage type 28) hit on the bot carried a swing
  1-2 frames earlier (same tick, swing sent before doHurtTarget's
  damage_event). Wire-verifies damage type mob_attack = 28.
- The zombie's burn (first run) sent entity flags datum (accessor 0,
  BYTE) = 1 (fire bit) and fire damage events typed 31 (on_fire)
  every 20 ticks; health datum (accessor 9, FLOAT) followed each burn
  tick. Damage type 31 = on_fire is thereby wire-verified.
- The mob-flags datum (accessor 15, BYTE) flipped 0 -> 4 when the
  attack goal started (before the first hit) and back to 0 when the
  target dropped.
- rotate_head (0x55): 19 frames across the chase - only when the
  packed head byte changed, riding interval-3 sync ticks. The head
  converges on the target bearing: errors reach 90+ deg mid-turn
  (the body drag at 75/tick while walking) and settle to 0-2 deg
  once the zombie stands at the target. Every landed hit can arrive
  while the head still lags; the swing does not wait for the head.
- Movement packets while walking: 0x36 (pos) and 0x37 (pos_rot)
  both flow, chosen per sync by whether the packed yaw byte changed;
  0x39 (rot-only) appears when a mob turns in place (one frame in
  the summon leg).
- Body yaw while turning lags the movement direction by up to 180
  deg (turn-in-then-walk); while settled it faces the approach.
- Idle sync heartbeat: none observed inside the short window (the
  60-tick zero-delta move fires once per second per entity; the
  session's still stretches were under that).
- The summon leg: the spider swung 7 times for its melee hits; the
  skeleton and creeper never swung (the bow goal does not swing; the
  creeper's swell preempts the approach before reach).

## 23. The melee swing: swing_animation 0x7b, not animate 0x02

`MeleeAttackGoal.checkAndPerformAttack` (the zombie's attack goal
wraps it): resetAttackCooldown, `mob.swingForAttack(MAIN_HAND)`, then
`mob.doHurtTarget`. The swing path:

- `LivingEntity.swing` builds `ClientboundSwingAnimationPacket`
  (packet `swing_animation`, id 0x7b in the registration chain) and
  sends it via `sendToTrackingPlayers` (NOT to self; the swinging
  entity is the mob).
- Payload: entityId VAR_INT, hand VAR_INT (MAIN_HAND 0), SwingAnimation
  VAR_INT (NONE 0, WHACK 1, STAB 2), duration VAR_INT. A bare hand
  (empty item) uses SwingAnimation.DEFAULT = (WHACK, 6): hand 0,
  type 1, duration 6.
- `ClientboundAnimatePacket` (id 0x02 `animate`, entity varint +
  action byte 0/1/2 wake-up/crit/magic-crit) is ONLY sent by
  ServerPlayer (player attack crit particles). Mobs never send it.
- Order per landed hit: swing_animation (from the attacker) arrives
  BEFORE the target's damage_event (0x19) and hurt_animation (0x2b),
  same tick.

The swing state re-arms only after the 6-tick swing duration, and the
attack cooldown (adjustedTickDelay(20) = 10 ticks for a mob ticking
goals every 2nd tick) is longer, so hits swing 1:1.

## 24. Facing: look control, body rotation, tracker thresholds

- `MoveControl` MOVE_TO turns the ENTITY yaw (the yaw move packets
  carry) toward the wanted position at up to 90 deg/tick.
- `LookControl`: `setLookAt(target, 30, 30)` from the melee goal sets
  a 2-tick cooldown; while it is > 0 the head yaw chases the wanted
  yaw at the given speed (30/tick melee; default look speed 10,
  pitch cap 40) and the pitch chases likewise, resetting to 0 first
  each tick (`resetXRotOnTick`). With no look set the head chases the
  BODY yaw at 10/tick. While the navigation is not done, the head is
  then pulled toward the body yaw at up to 75/tick
  (`rotateIfNecessary`, a step cap, not a window clamp).
- `BodyRotationControl`: while the entity moves (delta^2 >
  2.5e-7), body yaw snaps to the entity yaw and the head is pulled
  toward the body at 75/tick; idle, the body rotates toward the head
  at 75/tick after the head stabilizes 10 ticks (>15 deg moves reset
  the stable timer).
- `ServerEntity.sendChanges` runs the sync block on interval-3 ticks
  (or forced syncs / dirty entity data): rotation counts as changed
  when the packed degree byte differs by >= 1; the move packet is a
  full `entity_position_sync` on precise-position need /
  teleportDelay > 400 / riding / ground flip, else a delta packet
  when the position changed (>= 7.63e-6 squared) or tickCount % 60
  == 0 (a 60-tick zero-delta heartbeat for idle entities);
  `move_entity_pos_rot` when the yaw/pitch bytes changed,
  `move_entity_pos` otherwise, `move_entity_rot` when only rotation
  changed. `rotate_head` follows the move packet whenever the packed
  head byte differs by >= 1, inside the same sync block.

## 25. Metadata audit: the four mobs

Serializer ids per section 21 (registration order: BOOLEAN 8). Every
"sent" moment observed in the follow-and-attack and summon captures
or pinned by the decompile's define/set paths.

### Zombie (entity type 154)

| accessor | ser | default | vanilla sends it when | doppel |
|----------|-----|---------|----------------------|--------|
| 0 flags BYTE | 0 | 0 | bit 0x01 while burning | yes, fire bit |
| 6 pose | 20 | STANDING | death (DYING) | no (entity_event 3 carries the visual; listed unmodeled) |
| 8 living flags BYTE | 0 | 0 | using an item (zombies never do) | no (never set; correct) |
| 9 health FLOAT | 3 | 1.0 | pairing (20.0) and every change | yes |
| 15 mob flags BYTE | 0 | 0 | bit 0x04 while the attack goal runs (capture: flips 0 -> 4 before the first hit, back after) | yes |
| 16 baby BOOLEAN | 8 | false | baby zombies | no (adults only) |
| 17 special type INT | 1 | 0 | villager conversion | no |
| 18 drowned conversion BOOLEAN | 8 | false | converting | no |

### Creeper (32)

| accessor | ser | default | vanilla sends it when | doppel |
|----------|-----|---------|----------------------|--------|
| 9 health FLOAT | 3 | 1.0 | pairing | yes |
| 15 mob flags BYTE | 0 | 0 | bit 0x04 while the approach goal runs | yes |
| 16 swell dir INT | 1 | -1 | flips +1 inside 3 blocks / -1 past 7 or unseen (capture: one flip per approach) | yes |
| 17 powered BOOLEAN | 8 | false | lightning charge | no (no lightning) |
| 18 ignited BOOLEAN | 8 | false | flint and steel | no (no item use on mobs) |

### Skeleton (118)

| accessor | ser | default | vanilla sends it when | doppel |
|----------|-----|---------|----------------------|--------|
| 8 living flags BYTE | 0 | 0 | bit 0x01 while drawing the bow (the aim pose) | yes |
| 9 health FLOAT | 3 | 1.0 | pairing | yes |
| 15 mob flags BYTE | 0 | 0 | bit 0x04 while the bow goal runs | yes |
| 16 stray conversion BOOLEAN | 8 | false | powder snow | no |

The skeleton also pairs set_equipment main hand (bow) and its
movement_speed attribute.

### Spider (127)

| accessor | ser | default | vanilla sends it when | doppel |
|----------|-----|---------|----------------------|--------|
| 9 health FLOAT | 3 | 1.0 | pairing | yes |
| 15 mob flags BYTE | 0 | 0 | bit 0x04 while the attack goal runs | yes |
| 16 climbing BYTE | 0 | 0 | bit 0x01 while pressed against a wall, every tick | yes |

All four mobs pair movement speed ALONE in update_attributes - the
spider's off-default max health (16.0) stays off the snapshot too
(wire-verified in the summon capture; the wave-2 prediction of a
second max-health entry was wrong).

### States vanilla sets that doppel never models

- Zombie baby (16), special type (17), drowned conversion (18).
- Creeper powered (17), ignited (18).
- Skeleton stray conversion (16).
- The pose datum (6) on death (DYING); doppel's death path sends
  entity_event 3 and 60 only.
- Arrow pickup/disallow and in-ground item frames ride item
  entities, out of this audit's scope.

### Player hurt-rhythm divergence (out of scope, unchanged)

The reference's melee on a PLAYER target sends the damage_event
(0x19) to the victim's trackers AND the victim
(sendToTrackingPlayersAndSelf), while the hurt_animation (0x2b) goes
ONLY to the victim itself (ServerPlayer.indicateDamage sends to its
own connection; mobs' indicateDamage is empty, so a hurt MOB sends
no hurt_animation at all). Doppel broadcasts both to every player in
tracking range of the attacker. The victim's own view is equivalent;
other players see one extra hurt_animation per hit. Left unchanged
per the wave plan; knockback for players (the victim's
set_entity_motion from markHurt's velocity sync) is likewise out of
scope.

## 26. The detonation by value (three reference captures)

The summon scenario's creeper detonation, measured across three
vanilla runs:

- The packet: center (98.64, -60.00, 98.64) every run (the walk is
  deterministic from the summon point); radius 3.0 exactly; the
  destroyed-block count 321 / 332 / 338 - the count is the FULL ray
  sphere including air cells (the reference adds every cell a ray
  leaves with power, air costs no resistance and joins like any
  other); the struck bot's own knockback (0.337..0.338, 0.288..0.294,
  0.337..0.338) - the 3D unit vector from the CENTER to the victim's
  EYE (the y share is real), scaled by (1 - dist) * exposure with
  dist measured from the FEET position over the doubled radius.
- The tail: particle 29 (explosion_emitter), sound holder varint 704,
  two weighted block-particle entries (poof 0.5/1.0/weight 1, smoke
  1.0/1.0/weight 1), play-sound byte 1.
- The crater (cells that turned to air, from the section_blocks_update
  stream): 38 / 46 / 46 cells across runs, layers -63..-61, x/z
  96..100; the 38-cell run was a strict subset of the 46-cell run
  (Jaccard 0.826) - the fringe is the per-ray power roll noise.
- Gate tolerances (parity_mobs.rs): center 2.0 per axis, radius exact,
  count 15% relative, crater Jaccard 0.6 on center-relative cells -
  each below the observed variance with margin. The knockback is a
  per-side physics check (the vector must equal the unit ray from
  the side's own center to the victim's eye, scaled by
  (1 - feet distance / doubled radius); magnitude within 0.05,
  direction within 5 deg - the reference's own frames land within
  0.001 and 1 deg of the formula), because the two walks stop at
  different points inside the swell window (observed 1.8 blocks
  apart across servers) and any cross-comparison of absolute vectors
  would read the walk, not the blast.
- Wire-verified damage types along the way: arrow 0 (the skeleton's
  hit), mob_attack 28 (melee), on_fire 31 (the burn run),
  entity-attributed explosion 35 (the blast).

The engine changes this pinned: the destroyed set now counts air
cells (the packet's count), the knockback gained its y share and the
feet-distance basis, the sound holder moved 672 -> 704, and the
block-particle weights both read 1.

A cross-server run of the final gate: vanilla center
(98.64, -60.00, 98.64) count 324 crater 47; doppel center
(100.41, -60.00, 97.67) count 325 crater 44 - the counts 0.3% apart,
both radii exactly 3.0, both damage-type histograms identical
(arrow 0, melee 28, explosion 35), and the center-relative crater
plus the per-side knockback physics both inside their bounds.

## 27. Wire-verify burn-down (wave 3)

Every `TODO wire-verify` that stood in living.rs and explosion.rs at
the wave's start, with its source:

| constant | value | source |
|----------|-------|--------|
| PACKET_DAMAGE_EVENT | 0x19 | capture: the per-hit frames |
| PACKET_ENTITY_EVENT | 0x22 | capture: burn/death event frames |
| PACKET_HURT_ANIMATION | 0x2b | capture: one per landed hit |
| PACKET_MOVE_ENTITY_ROT | 0x39 | capture: the summon leg's rot-only frame |
| PACKET_ROTATE_HEAD | 0x55 | capture: the head-tracking frames |
| PACKET_SET_EQUIPMENT | 0x68 | capture: the skeleton's bow pairing |
| PACKET_UPDATE_ATTRIBUTES | 0x86 | gate: the pairing attribute frames |
| ENTITY_TYPE_ZOMBIE/SKELETON/CREEPER/SPIDER | 154/118/32/127 | capture: the add frames |
| ATTR_MAX_HEALTH | 23 | decompile registry count; never on the wire (the reference's mob snapshots carry movement_speed alone) |
| ATTR_MOVEMENT_SPEED | 26 | prior gate + capture |
| DAMAGE_TYPE_MOB_ATTACK | 28 | capture: melee hits typed 28 |
| DAMAGE_TYPE_EXPLOSION | 35 | capture: the blast hit typed 35 |
| explode packet id | 0x24 | capture: the detonation frame |
| swing_animation | 0x7b | capture: 10 frames per 10 hits |

Also verified, outside this wave's file set (noted for the next
wave): the arrow damage type id 0 (the skeleton's landed arrow read
type 0 in the summon capture; projectile.rs's marker can flip with
this citation). Nothing remains unconfirmable.
