import struct, zlib, sys, collections

def chunks(path):
    data = open(path, 'rb').read()
    out = []
    for lz in range(32):
        for lx in range(32):
            idx = (lz * 32 + lx) * 4
            off = (data[idx] << 16) | (data[idx+1] << 8) | data[idx+2]
            cnt = data[idx+3]
            if off == 0 or cnt == 0:
                continue
            start = off * 4096
            length = struct.unpack('>I', data[start:start+4])[0]
            comp = data[start+4]
            raw = data[start+5:start+4+length]
            if comp == 2:
                raw = zlib.decompress(raw)
            elif comp == 1:
                import gzip
                raw = gzip.decompress(raw)
            out.append((lx, lz, raw))
    return out

def payload(body, tag):
    pos = 3
    def u1():
        nonlocal pos
        v = body[pos]; pos += 1; return v
    def s2():
        nonlocal pos
        v = struct.unpack_from('>h', body, pos)[0]; pos += 2; return v
    def i4():
        nonlocal pos
        v = struct.unpack_from('>i', body, pos)[0]; pos += 4; return v
    def i8():
        nonlocal pos
        v = struct.unpack_from('>q', body, pos)[0]; pos += 8; return v
    def p(t):
        nonlocal pos
        if t == 1: v = body[pos]; pos += 1; return v
        if t == 2: return s2()
        if t == 3: return i4()
        if t == 4: return i8()
        if t == 5: pos += 4; return None
        if t == 6: pos += 8; return None
        if t == 7:
            n = i4(); pos += n; return None
        if t == 8:
            n = struct.unpack_from('>H', body, pos)[0]; pos += 2
            v = body[pos:pos+n].decode('utf-8', 'replace'); pos += n; return v
        if t == 9:
            e = u1(); n = i4()
            return [p(e) for _ in range(n)]
        if t == 10:
            out = {}
            while True:
                tt = u1()
                if tt == 0: return out
                l = struct.unpack_from('>H', body, pos)[0]; pos += 2
                name = body[pos:pos+l].decode('utf-8', 'replace'); pos += l
                out[name] = p(tt)
        if t == 11:
            n = i4(); pos += 4 * n; return None
        if t == 12:
            n = i4(); pos += 8 * n; return None
        raise ValueError(f'tag {t} at {pos}')
    return p(tag)

def level_of(raw):
    # root tag 10, empty name
    return payload(raw, 10)

root_dir = sys.argv[1]
total_chunks = 0
names = collections.Counter()
positions = []
for rn in ['r.-1.-1.mca', 'r.-1.0.mca', 'r.0.-1.mca', 'r.0.0.mca']:
    for (lx, lz, raw) in chunks(f'{root_dir}/{rn}'):
        level = level_of(raw)
        total_chunks += 1
        cx, cz = level['xPos'], level['zPos']
        surface = None
        mixed = 0
        for s in level['sections']:
            b = s.get('biomes')
            if not b:
                continue
            y = s['Y']
            if y >= 250:
                y -= 256
            if y == 4:  # surface section around y=64
                surface = b['palette']
                if len(b['palette']) > 1:
                    mixed += 1
        positions.append((cx, cz, surface, level.get('Status')))
        for n in surface or []:
            names[n] += 1
print('chunks:', total_chunks)
print('biome names at surface section:', names)
print('positions (first 40):')
for p in positions[:40]:
    print('  ', p)
