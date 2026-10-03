import struct, zlib, sys

def read_region(path, lx, lz):
    data = open(path, 'rb').read()
    idx = (lz * 32 + lx) * 4
    off = (data[idx] << 16) | (data[idx+1] << 8) | data[idx+2]
    cnt = data[idx+3]
    if off == 0 or cnt == 0:
        return None
    start = off * 4096
    length = struct.unpack('>I', data[start:start+4])[0]
    comp = data[start+4]
    body = data[start+5:start+4+length]
    if comp == 2:
        return zlib.decompress(body)
    raise SystemExit(f'compression {comp}')

# Minimal NBT walker returning (name, python value).
def parse_nbt(buf):
    pos = [0]
    def u1():
        v = buf[pos[0]]; pos[0] += 1; return v
    def u2():
        v = struct.unpack_from('>H', buf, pos[0])[0]; pos[0] += 2; return v
    def u4():
        v = struct.unpack_from('>i', buf, pos[0])[0]; pos[0] += 4; return v
    def u8():
        v = struct.unpack_from('>q', buf, pos[0])[0]; pos[0] += 8; return v
    def payload(tag):
        if tag == 1: return u1()
        if tag == 2:
            v = struct.unpack_from('>h', buf, pos[0])[0]; pos[0]+=2; return v
        if tag == 3: return u4()
        if tag == 4: return u8()
        if tag == 5:
            v = struct.unpack_from('>f', buf, pos[0])[0]; pos[0]+=4; return v
        if tag == 6:
            v = struct.unpack_from('>d', buf, pos[0])[0]; pos[0]+=8; return v
        if tag == 7:
            n = u4(); v = buf[pos[0]:pos[0]+n]; pos[0]+=n; return v
        if tag == 8:
            n = u2(); v = buf[pos[0]:pos[0]+n].decode('utf-8', 'replace'); pos[0]+=n; return v
        if tag == 9:
            elem = u1(); n = u4()
            return [payload(elem) for _ in range(n)]
        if tag == 10:
            out = {}
            while True:
                t = u1()
                if t == 0: return out
                name_len = u2()
                name = buf[pos[0]:pos[0]+name_len].decode('utf-8', 'replace'); pos[0]+=name_len
                out[name] = payload(t)
        if tag == 11:
            n = u4(); v = list(struct.unpack_from(f'>{n}i', buf, pos[0])); pos[0]+=4*n; return v
        if tag == 12:
            n = u4(); v = list(struct.unpack_from(f'>{n}q', buf, pos[0])); pos[0]+=8*n; return v
        raise SystemExit(f'tag {tag}')
    t = u1()
    assert t == 10, f'root {t}'
    name_len = u2()
    name = buf[pos[0]:pos[0]+name_len].decode(); pos[0]+=name_len
    return payload(10)

raw = read_region(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
root = parse_nbt(raw)
level = root['']
print('chunk keys:', sorted(level.keys()))
for s in level['sections']:
    b = s.get('biomes')
    if b is None:
        print('y', s['Y'], 'no biomes')
        continue
    print('y', s['Y'], 'palette', b['palette'], 'data longs', None if 'data' not in b else len(b['data']))
