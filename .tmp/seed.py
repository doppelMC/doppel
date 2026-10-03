import gzip, struct

body = gzip.open('C:/Users/nour/doppel/target/vanilla/pristine-world/level.dat', 'rb').read()
pos = [0]

def payload(tag):
    p = pos[0]
    if tag == 1:
        pos[0] += 1; return body[p]
    if tag == 2:
        pos[0] += 2; return struct.unpack_from('>h', body, p)[0]
    if tag == 3:
        pos[0] += 4; return struct.unpack_from('>i', body, p)[0]
    if tag == 4:
        pos[0] += 8; return struct.unpack_from('>q', body, p)[0]
    if tag == 5:
        pos[0] += 4; return struct.unpack_from('>f', body, p)[0]
    if tag == 6:
        pos[0] += 8; return struct.unpack_from('>d', body, p)[0]
    if tag == 7:
        n = struct.unpack_from('>i', body, p)[0]; pos[0] += 4 + n; return body[p+4:p+4+n]
    if tag == 8:
        n = struct.unpack_from('>H', body, p)[0]; pos[0] += 2
        v = body[pos[0]:pos[0]+n].decode('utf-8', 'replace'); pos[0] += n; return v
    if tag == 9:
        e = body[p]; pos[0] += 1
        n = struct.unpack_from('>i', body, pos[0])[0]; pos[0] += 4
        return [payload(e) for _ in range(n)]
    if tag == 10:
        out = {}
        while True:
            t = body[pos[0]]; pos[0] += 1
            if t == 0:
                return out
            l = struct.unpack_from('>H', body, pos[0])[0]; pos[0] += 2
            name = body[pos[0]:pos[0]+l].decode('utf-8', 'replace'); pos[0] += l
            out[name] = payload(t)
    if tag == 11:
        n = struct.unpack_from('>i', body, p)[0]; pos[0] += 4 + 4 * n; return None
    if tag == 12:
        n = struct.unpack_from('>i', body, p)[0]; pos[0] += 4 + 8 * n; return None
    raise ValueError(f'tag {tag}')

root_tag = body[0]
name_len = struct.unpack_from('>H', body, 1)[0]
pos[0] = 3 + name_len
data = payload(10)
wgs = data.get('Data', {}).get('WorldGenSettings', {})
print('seed', wgs.get('seed'))
print('dimensions', list(wgs.get('dimensions', {}).keys()))
ow = wgs.get('dimensions', {}).get('minecraft:overworld', {})
print('generator', ow.get('type'), str(ow)[:300])
print('Data keys', sorted(data.get('Data', {}).keys())[:40])
