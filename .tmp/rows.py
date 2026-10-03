import json
d = json.load(open('pins/worldgen/multi_noise_biome_source_parameter_list/overworld.json'))
rows = d['biomes']
def show(name, limit=6):
    n = 0
    for r in rows:
        if r['biome'] == name:
            p = r['parameters']
            axes = {k: (round(v[0], 3), round(v[1], 3)) for k, v in p.items() if k != 'offset'}
            print(name, axes, 'offset', p['offset'])
            n += 1
            if n >= limit:
                break
for b in ['minecraft:plains', 'minecraft:dark_forest', 'minecraft:beach', 'minecraft:river', 'minecraft:lush_caves']:
    show(b)
