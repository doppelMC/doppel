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
at pairing (predicted: only max_health 20.0).
