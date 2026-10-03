import json, struct

f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
q = lambda v: int(v * f32(10000.0))  # truncation toward zero like Java

d = json.load(open('pins/worldgen/multi_noise_biome_source_parameter_list/overworld.json'))
rows = d['biomes']
AXES = ['temperature', 'humidity', 'continentalness', 'erosion', 'depth', 'weirdness']

def spans(r):
    p = r['parameters']
    out = []
    for a in AXES:
        lo, hi = p[a]
        out.append((q(f32(lo)), q(f32(hi))))
    return out, q(f32(p['offset']))

target = [1126, 5255, -1100, 1114, 1643, -851]
scored = []
for r in rows:
    sp, off = spans(r)
    fit = off * off
    for (lo, hi), t in zip(sp, target):
        above = t - hi
        dist = above if above > 0 else max(lo - t, 0)
        fit += dist * dist
    scored.append((fit, r['biome']))
scored.sort(key=lambda x: x[0])
for fit, name in scored[:8]:
    print(fit, name)
print('--- best beach and best dark_forest')
best = {}
for fit, name in scored:
    best.setdefault(name, fit)
for n in ['minecraft:beach', 'minecraft:dark_forest', 'minecraft:forest', 'minecraft:birch_forest']:
    print(n, best.get(n))
