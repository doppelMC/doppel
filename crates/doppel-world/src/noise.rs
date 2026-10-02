//! Seeded noise and seed derivation for world generation.
//!
//! Two random sources cover the two seed derivations the generator chain
//! needs: a 48-bit LCG (region placement and other position-keyed decisions)
//! and a 128-bit rotate-xor stream (noise construction). Seeds flow through
//! fixed arithmetic (64-bit mixing, name hashing) so every stage is a pure
//! function of the world seed.
//!
//! Float sampling rounds to f32 after every operation; the layered sampler
//! accumulates in f32 as well. Keeping the operation granularity identical
//! matters because the layered amplitudes are narrowed to f32 first.

/// The 48-bit LCG: 25214903917 * seed + 11 mod 2^48, top bits returned.
pub struct Lcg48 {
    seed: u64,
}

impl Lcg48 {
    pub fn new(seed: i64) -> Lcg48 {
        let mut rng = Lcg48 { seed: 0 };
        rng.set_seed(seed);
        rng
    }

    /// Seeds scramble with the multiplier before the first draw.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = ((seed as u64) ^ 0x5DEECE66D) & 0xFFFF_FFFF_FFFF;
    }

    /// Draws `bits` (1..=32) high bits of the next state.
    pub fn next_bits(&mut self, bits: u32) -> u32 {
        self.seed = (self.seed.wrapping_mul(25_214_903_917).wrapping_add(11)) & 0xFFFF_FFFF_FFFF;
        (self.seed >> (48 - bits)) as u32
    }

    /// The modulo method with rejection for bias: draws 31 bits until the
    /// leftover bucket range fits.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        debug_assert!(bound > 0);
        let b = bound as i64;
        loop {
            let bits = self.next_bits(31) as i64;
            let value = bits % b;
            if bits - value + (b - 1) <= i32::MAX as i64 {
                return value as i32;
            }
        }
    }

    pub fn next_long(&mut self) -> i64 {
        let hi = self.next_bits(32) as i32 as i64;
        let lo = self.next_bits(32) as i32 as i64;
        (hi << 32).wrapping_add(lo)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_bits(24) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    pub fn next_f64(&mut self) -> f64 {
        let hi = self.next_bits(26) as u64;
        let lo = self.next_bits(27) as u64;
        ((hi << 27) | lo) as f64 / (1u64 << 53) as f64
    }
}

/// The 128-bit rotate-xor stream: result = rotl(lo + hi, 17) + lo, then the
/// halves recombine through two different rotations.
pub struct Xoroshiro {
    lo: u64,
    hi: u64,
}

impl Xoroshiro {
    pub fn new(lo: i64, hi: i64) -> Xoroshiro {
        let (lo, hi) = (lo as u64, hi as u64);
        if lo | hi == 0 {
            // The all-zero state is the single fixed point; nudge it.
            return Xoroshiro {
                lo: (-7046029254386353131i64) as u64,
                hi: 7640891576956012809u64,
            };
        }
        Xoroshiro { lo, hi }
    }

    pub fn next_long(&mut self) -> i64 {
        let s0 = self.lo;
        let s1 = self.hi;
        let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
        let s1 = s1 ^ s0;
        self.lo = s0.rotate_left(49) ^ s1 ^ (s1 << 21);
        self.hi = s1.rotate_left(28);
        result as i64
    }

    fn next_bits(&mut self, bits: u32) -> u64 {
        (self.next_long() as u64) >> (64 - bits)
    }

    /// The wide-multiply method with rejection for bias.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        debug_assert!(bound > 0);
        let bound = bound as u32 as u64;
        // Rejection threshold: the 32-bit two's complement of the bound,
        // reduced modulo the bound.
        let threshold = (bound as u32).wrapping_neg() as u64 % bound;
        let mut bits = self.next_long() as u32 as u64;
        let mut product = bits * bound;
        let mut low = product & 0xFFFF_FFFF;
        if low < bound {
            while low < threshold {
                bits = self.next_long() as u32 as u64;
                product = bits * bound;
                low = product & 0xFFFF_FFFF;
            }
        }
        (product >> 32) as i32
    }

    pub fn next_f64(&mut self) -> f64 {
        self.next_bits(53) as f64 / (1u64 << 53) as f64
    }

    /// Derives the position-keyed factory: two draws become its halves.
    pub fn fork_positional(&mut self) -> Positional {
        Positional {
            lo: self.next_long(),
            hi: self.next_long(),
        }
    }
}

/// A frozen seed pair that spawns independent streams per name or position.
pub struct Positional {
    lo: i64,
    hi: i64,
}

impl Positional {
    /// MD5 of the name (big-endian halves) xored with the factory halves.
    pub fn from_name(&self, name: &str) -> Xoroshiro {
        let (hash_lo, hash_hi) = name_hash_128(name);
        Xoroshiro::new(hash_lo ^ self.lo, hash_hi ^ self.hi)
    }

    pub fn at(&self, x: i32, y: i32, z: i32) -> Xoroshiro {
        Xoroshiro::new(positional_seed(x, y, z) ^ self.lo, self.hi)
    }
}

/// The 64-bit Stafford 13 mixer: logical xor-shift 30, multiply, logical
/// xor-shift 27, multiply, logical xor-shift 31.
pub fn stafford13(z: i64) -> i64 {
    let mut z = z as u64;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as i64
}

/// Widens a 64-bit seed to 128 bits: xor with the silver ratio, add the
/// golden ratio, mix both halves.
pub fn upgrade_seed(seed: i64) -> (i64, i64) {
    let lo = seed ^ 0x6A09_E667_F3BC_C909;
    let hi = lo.wrapping_add(-7046029254386353131);
    (stafford13(lo), stafford13(hi))
}

/// The position seed: a wrapping int product folded with the other axes,
/// squared, mixed, and shifted down 16.
pub fn positional_seed(x: i32, y: i32, z: i32) -> i64 {
    let xs = x.wrapping_mul(3129871) as i64;
    let mut seed = xs ^ (z as i64).wrapping_mul(116129781) ^ y as i64;
    seed = seed
        .wrapping_mul(seed)
        .wrapping_mul(42317861)
        .wrapping_add(seed.wrapping_mul(11));
    seed >> 16
}

/// The positional factory for a world seed: upgrade, then fork.
pub fn world_positional(seed: i64) -> Positional {
    let (lo, hi) = upgrade_seed(seed);
    let mut rng = Xoroshiro::new(lo, hi);
    rng.fork_positional()
}

/// MD5 digest of the text, split into two big-endian 64-bit halves.
pub fn name_hash_128(name: &str) -> (i64, i64) {
    let digest = md5(name.as_bytes());
    let lo = i64::from_be_bytes(digest[0..8].try_into().unwrap());
    let hi = i64::from_be_bytes(digest[8..16].try_into().unwrap());
    (lo, hi)
}

/// MD5 (RFC 1321).
fn md5(input: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x2441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x4881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut message = input.to_vec();
    let bit_len = (input.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let (chunks, _) = message.as_chunks::<64>();
    for chunk in chunks {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let sum = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(sum.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out
}

/// The 16 corner gradients of the gradient noise family.
const GRADIENTS: [[i32; 3]; 16] = [
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 0],
    [0, -1, 1],
    [-1, 1, 0],
    [0, -1, -1],
];

/// Shared state of both gradient noise flavors: a random offset that moves
/// samples off the integer lattice and a shuffled permutation table.
#[derive(Clone)]
struct GradientBase {
    ox: f64,
    oy: f64,
    oz: f64,
    perms: [u8; 256],
}

impl GradientBase {
    /// Draws three doubles for the offsets, then Fisher-Yates shuffles the
    /// 256-entry permutation table.
    fn new(rng: &mut Xoroshiro) -> GradientBase {
        let mut base = GradientBase {
            ox: rng.next_f64() * 256.0,
            oy: rng.next_f64() * 256.0,
            oz: rng.next_f64() * 256.0,
            perms: std::array::from_fn(|i| i as u8),
        };
        for i in 0..256usize {
            let offset = rng.next_int((256 - i) as i32) as usize;
            base.perms.swap(i, i + offset);
        }
        base
    }

    fn permute(&self, x: i32) -> i32 {
        self.perms[(x & 0xFF) as usize] as i32
    }
}

/// Wraps inputs beyond the 2^24.5 float precision cliff into the representable
/// band so far coordinates keep sampling smoothly.
fn wrap(x: f64) -> f64 {
    // Largest double below 2^24; beyond it, float fractions lose meaning.
    const HALF_ROUND_OFF: f64 = 16777215.999999998;
    if (-HALF_ROUND_OFF..HALF_ROUND_OFF).contains(&x) {
        return x;
    }
    x - (x / 33554432.0 + 0.5).floor() * 33554432.0
}

fn smoothstep(x: f32) -> f32 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

fn lerp(alpha: f32, p0: f32, p1: f32) -> f32 {
    p0 + alpha * (p1 - p0)
}

#[allow(clippy::too_many_arguments)]
fn lerp3(
    a1: f32,
    a2: f32,
    a3: f32,
    x000: f32,
    x100: f32,
    x010: f32,
    x110: f32,
    x001: f32,
    x101: f32,
    x011: f32,
    x111: f32,
) -> f32 {
    let lerp2 = |a1: f32, a2: f32, x00: f32, x10: f32, x01: f32, x11: f32| {
        lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11))
    };
    lerp(
        a3,
        lerp2(a1, a2, x000, x100, x010, x110),
        lerp2(a1, a2, x001, x101, x011, x111),
    )
}

fn grad_dot(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let g = &GRADIENTS[(hash & 0xF) as usize];
    g[0] as f32 * x + g[1] as f32 * y + g[2] as f32 * z
}

/// Classic lattice gradient noise: corner gradients selected through the
/// permutation table, interpolated along smoothed cell fractions.
#[derive(Clone)]
pub struct Perlin {
    base: GradientBase,
}

impl Perlin {
    pub fn new(rng: &mut Xoroshiro) -> Perlin {
        Perlin {
            base: GradientBase::new(rng),
        }
    }

    pub fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        let x = wrap(x) + self.base.ox;
        let y = wrap(y) + self.base.oy;
        let z = wrap(z) + self.base.oz;
        let (fx, fy, fz) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
        let rx = (x - fx as f64) as f32;
        let ry = (y - fy as f64) as f32;
        let rz = (z - fz as f64) as f32;
        self.sample_and_lerp(fx, fy, fz, rx, ry, rz, ry)
    }

    /// Samples the y=0 plane.
    pub fn sample_2d(&self, x: f64, z: f64) -> f32 {
        self.sample(wrap(x), 0.0, wrap(z))
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_and_lerp(
        &self,
        x: i32,
        y: i32,
        z: i32,
        rx: f32,
        ry: f32,
        rz: f32,
        original_ry: f32,
    ) -> f32 {
        let x0 = self.base.permute(x);
        let x1 = self.base.permute(x + 1);
        let xy00 = self.base.permute(x0 + y);
        let xy01 = self.base.permute(x0 + y + 1);
        let xy10 = self.base.permute(x1 + y);
        let xy11 = self.base.permute(x1 + y + 1);
        let d000 = grad_dot(self.base.permute(xy00 + z), rx, ry, rz);
        let d100 = grad_dot(self.base.permute(xy10 + z), rx - 1.0, ry, rz);
        let d010 = grad_dot(self.base.permute(xy01 + z), rx, ry - 1.0, rz);
        let d110 = grad_dot(self.base.permute(xy11 + z), rx - 1.0, ry - 1.0, rz);
        let d001 = grad_dot(self.base.permute(xy00 + z + 1), rx, ry, rz - 1.0);
        let d101 = grad_dot(self.base.permute(xy10 + z + 1), rx - 1.0, ry, rz - 1.0);
        let d011 = grad_dot(self.base.permute(xy01 + z + 1), rx, ry - 1.0, rz - 1.0);
        let d111 = grad_dot(
            self.base.permute(xy11 + z + 1),
            rx - 1.0,
            ry - 1.0,
            rz - 1.0,
        );
        lerp3(
            smoothstep(rx),
            smoothstep(original_ry),
            smoothstep(rz),
            d000,
            d100,
            d010,
            d110,
            d001,
            d101,
            d011,
            d111,
        )
    }
}

/// Simplex gradient noise: the cell is a triangle (2D) or tetrahedron (3D)
/// and corner contributions fall off with squared distance.
#[derive(Clone)]
pub struct Simplex {
    base: GradientBase,
}

impl Simplex {
    pub fn new(rng: &mut Xoroshiro) -> Simplex {
        Simplex {
            base: GradientBase::new(rng),
        }
    }

    fn corner(&self, index: i32, x: f64, y: f64, z: f64, base: f64) -> f64 {
        let t = base - x * x - y * y - z * z;
        if t < 0.0 {
            return 0.0;
        }
        let t = t * t;
        let g = &GRADIENTS[(index % 12) as usize];
        t * t * (g[0] as f64 * x + g[1] as f64 * y + g[2] as f64 * z)
    }

    pub fn sample_2d(&self, x: f64, z: f64) -> f32 {
        const SQRT_3: f64 = 1.7320508075688772;
        const F2: f64 = 0.5 * (SQRT_3 - 1.0);
        const G2: f64 = (3.0 - SQRT_3) / 6.0;
        let x = x + self.base.ox;
        let z = z + self.base.oy;
        let s = (x + z) * F2;
        let i = (x + s).floor();
        let j = (z + s).floor();
        let t = (i + j) * G2;
        let x0 = x - (i - t);
        let z0 = z - (j - t);
        let (i1, j1) = if x0 > z0 { (1, 0) } else { (0, 1) };
        let x1 = x0 - i1 as f64 + G2;
        let z1 = z0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let z2 = z0 - 1.0 + 2.0 * G2;
        let ii = i as i32 & 0xFF;
        let jj = j as i32 & 0xFF;
        let gi0 = self.base.permute(ii + self.base.permute(jj)) % 12;
        let gi1 = self.base.permute(ii + i1 + self.base.permute(jj + j1)) % 12;
        let gi2 = self.base.permute(ii + 1 + self.base.permute(jj + 1)) % 12;
        let n0 = self.corner(gi0, x0, z0, 0.0, 0.5);
        let n1 = self.corner(gi1, x1, z1, 0.0, 0.5);
        let n2 = self.corner(gi2, x2, z2, 0.0, 0.5);
        (70.0 * (n0 + n1 + n2)) as f32
    }

    pub fn sample_3d(&self, x: f64, y: f64, z: f64) -> f32 {
        let x = x + self.base.ox;
        let y = y + self.base.oy;
        let z = z + self.base.oz;
        let s = (x + y + z) / 3.0;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let k = (z + s).floor();
        let t = (i + j + k) / 6.0;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let z0 = z - (k - t);
        // Simplex vertex visit order: the axis order of the largest fraction
        // picks the first edge, the smallest picks the second.
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };
        let x1 = x0 - i1 as f64 + 1.0 / 6.0;
        let y1 = y0 - j1 as f64 + 1.0 / 6.0;
        let z1 = z0 - k1 as f64 + 1.0 / 6.0;
        let x2 = x0 - i2 as f64 + 1.0 / 3.0;
        let y2 = y0 - j2 as f64 + 1.0 / 3.0;
        let z2 = z0 - k2 as f64 + 1.0 / 3.0;
        let x3 = x0 - 0.5;
        let y3 = y0 - 0.5;
        let z3 = z0 - 0.5;
        let ii = i as i32 & 0xFF;
        let jj = j as i32 & 0xFF;
        let kk = k as i32 & 0xFF;
        let gi0 = self
            .base
            .permute(ii + self.base.permute(jj + self.base.permute(kk)))
            % 12;
        let gi1 = self
            .base
            .permute(ii + i1 + self.base.permute(jj + j1 + self.base.permute(kk + k1)))
            % 12;
        let gi2 = self
            .base
            .permute(ii + i2 + self.base.permute(jj + j2 + self.base.permute(kk + k2)))
            % 12;
        let gi3 = self
            .base
            .permute(ii + 1 + self.base.permute(jj + 1 + self.base.permute(kk + 1)))
            % 12;
        let n0 = self.corner(gi0, x0, y0, z0, 0.6);
        let n1 = self.corner(gi1, x1, y1, z1, 0.6);
        let n2 = self.corner(gi2, x2, y2, z2, 0.6);
        let n3 = self.corner(gi3, x3, y3, z3, 0.6);
        (32.0 * (n0 + n1 + n2 + n3)) as f32
    }
}

/// Layer parameters for the octave stack: octaves double in frequency and
/// halve in amplitude from `base_octave` up; a zero modifier drops the octave
/// without shifting anything (each octave derives its seed from its index).
#[derive(Clone)]
pub struct OctaveSpec {
    pub base_octave: i32,
    pub octave_count: usize,
    pub base_amplitude: f64,
    pub amplitude_modifiers: &'static [f64],
}

struct OctaveLayer {
    noise: Perlin,
    frequency: f64,
    amplitude: f32,
}

/// The normalized octave stack: every octave contributes two lattice noise
/// layers (the second at a slightly higher frequency), scaled so the combined
/// standard deviation reaches a third of the total amplitude.
pub struct OctaveNoise {
    layers: Vec<OctaveLayer>,
}

/// Per-layer standard deviation of one lattice noise.
const LATTICE_STD: f64 = 0.2702247831245211;
/// Frequency stretch of the second noise per octave.
const SECOND_FREQUENCY: f64 = 1.0181268882175227;

impl OctaveNoise {
    pub fn new(spec: &OctaveSpec, rng: &mut Xoroshiro) -> OctaveNoise {
        let first = rng.fork_positional();
        let second = rng.fork_positional();

        let mut frequency = 2f64.powi(spec.base_octave);
        // Normalized stacks start the halving chain at the geometric sum so
        // the total amplitude lands on base_amplitude.
        let mut amplitude = spec.base_amplitude * 2f64.powi(spec.octave_count as i32 - 1)
            / (2f64.powi(spec.octave_count as i32) - 1.0);
        let mut octaves = Vec::new();
        for i in 0..spec.octave_count {
            let modifier = spec.amplitude_modifiers.get(i).copied().unwrap_or(1.0);
            if modifier != 0.0 {
                octaves.push((spec.base_octave + i as i32, frequency, amplitude * modifier));
            }
            frequency *= 2.0;
            amplitude *= 0.5;
        }

        let target: f64 = octaves.iter().map(|(_, _, a)| a.abs()).sum();
        let variance: f64 = octaves
            .iter()
            .map(|(_, _, a)| {
                let deviation = LATTICE_STD * a.abs();
                deviation * deviation
            })
            .sum();
        let deviation = variance.sqrt();
        let normalization = if deviation == 0.0 {
            0.0
        } else {
            (target / 3.0) / (deviation * 2f64.sqrt())
        };

        let mut layers = Vec::with_capacity(octaves.len() * 2);
        for (index, frequency, amplitude) in octaves {
            let name = format!("octave_{index}");
            let value_factor = (normalization * amplitude) as f32;
            layers.push(OctaveLayer {
                noise: Perlin::new(&mut first.from_name(&name)),
                frequency,
                amplitude: value_factor,
            });
            layers.push(OctaveLayer {
                noise: Perlin::new(&mut second.from_name(&name)),
                frequency: frequency * SECOND_FREQUENCY,
                amplitude: value_factor,
            });
        }
        OctaveNoise { layers }
    }

    pub fn sample_2d(&self, x: f64, z: f64) -> f32 {
        let mut value = 0f32;
        for layer in &self.layers {
            value += layer.amplitude
                * layer
                    .noise
                    .sample_2d(x * layer.frequency, z * layer.frequency);
        }
        value
    }

    pub fn sample_3d(&self, x: f64, y: f64, z: f64) -> f32 {
        let mut value = 0f32;
        for layer in &self.layers {
            value += layer.amplitude
                * layer.noise.sample(
                    x * layer.frequency,
                    y * layer.frequency,
                    z * layer.frequency,
                );
        }
        value
    }
}

#[cfg(test)]
mod tests {
    // Spot values are exact bit patterns; grouping or trimming their digits
    // would risk parsing a neighboring float.
    #![allow(clippy::excessive_precision)]

    use super::*;

    #[test]
    fn lcg48_spot_values() {
        // The first 32-bit draw at seed 0 is the widely known -1155484576.
        assert_eq!(Lcg48::new(0).next_bits(32), 3139482720);

        let mut rng = Lcg48::new(12345);
        assert_eq!(
            (0..3).map(|_| rng.next_bits(32)).collect::<Vec<_>>(),
            [1553932502, 2204218161, 4007176482]
        );
        let mut rng = Lcg48::new(12345);
        assert_eq!(
            (0..2).map(|_| rng.next_long()).collect::<Vec<_>>(),
            [6674089274190705457, -1236052134575208584]
        );
        let mut rng = Lcg48::new(12345);
        assert_eq!(
            (0..5).map(|_| rng.next_int(16)).collect::<Vec<_>>(),
            [11, 8, 1, 12, 7]
        );
        let mut rng = Lcg48::new(12345);
        assert_eq!(
            (0..2).map(|_| rng.next_f64()).collect::<Vec<_>>(),
            [0.3618031071604718, 0.932993485288541]
        );
        let mut rng = Lcg48::new(12345);
        assert_eq!(
            (0..2).map(|_| rng.next_f32()).collect::<Vec<_>>(),
            [0.3618030548095703, 0.5132095217704773]
        );
    }

    #[test]
    fn xoroshiro_spot_values() {
        let (lo, hi) = upgrade_seed(42);
        let mut rng = Xoroshiro::new(lo, hi);
        assert_eq!(
            (0..4).map(|_| rng.next_long()).collect::<Vec<_>>(),
            [
                -4695948378737616609,
                7341713790291473579,
                -7542733514721318211,
                4888889476139319686
            ]
        );
        let (lo, hi) = upgrade_seed(42);
        let mut rng = Xoroshiro::new(lo, hi);
        let factory = rng.fork_positional();
        assert_eq!(
            (factory.lo, factory.hi),
            (-4695948378737616609, 7341713790291473579)
        );

        let mut named = factory.from_name("octave_-9");
        assert_eq!(
            (0..3).map(|_| named.next_long()).collect::<Vec<_>>(),
            [
                6532832884475921422,
                3985943916790613243,
                7209119041371909024
            ]
        );
        assert_eq!(named.next_f64(), 0.9070324153473356);
    }

    #[test]
    fn seed_mixing_spot_values() {
        assert_eq!(stafford13(0x1234567890ABCDEF), -2893567902102913444);
        assert_eq!(positional_seed(10, -64, 20), -95522250699922);
        assert_eq!(positional_seed(-5, 100, -7), 69148817394877);
        assert_eq!(
            name_hash_128("octave_-9"),
            (589938935082149425, 5662732952352513153)
        );
    }

    #[test]
    fn perlin_spot_values() {
        let factory = Positional {
            lo: 0x1111111111111111,
            hi: 0x2222222222222222,
        };
        let perlin = Perlin::new(&mut factory.from_name("test"));
        assert_eq!(perlin.sample(0.5, 0.25, -0.75), -0.03374312445521355f32);
        assert_eq!(perlin.sample_2d(0.3, 0.7), -0.012237067334353924f32);

        let simplex = Simplex::new(&mut factory.from_name("test"));
        assert_eq!(simplex.sample_2d(0.5, 0.25), -0.16663165341994815f32);
        assert_eq!(simplex.sample_3d(0.5, 0.25, -0.75), 0.2689221793283955f32);
    }

    fn continental_spec() -> OctaveSpec {
        OctaveSpec {
            base_octave: -9,
            octave_count: 9,
            base_amplitude: 0.8880832896205223,
            amplitude_modifiers: &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn octave_noise_spot_values() {
        let world = world_positional(42);
        let noise = OctaveNoise::new(&continental_spec(), &mut world.from_name("continental"));
        assert_eq!(noise.sample_2d(0.0, 0.0), -0.12332764267921448f32);
        assert_eq!(noise.sample_2d(100.0, -35.0), -0.18135394155979156f32);
        assert_eq!(noise.sample_2d(1234.0, 567.0), -0.10254176706075668f32);
    }

    #[test]
    fn octave_noise_determinism_and_separation() {
        let spec = continental_spec();
        let world = world_positional(7);
        let a = OctaveNoise::new(&spec, &mut world.from_name("continental"));
        let world = world_positional(7);
        let b = OctaveNoise::new(&spec, &mut world.from_name("continental"));
        let world = world_positional(8);
        let c = OctaveNoise::new(&spec, &mut world.from_name("continental"));

        let points = (0..64).map(|i| (i as f64 * 13.7, i as f64 * -7.3));
        let mut all_equal = true;
        let mut any_differs = false;
        for (x, z) in points {
            let (va, vb, vc) = (a.sample_2d(x, z), b.sample_2d(x, z), c.sample_2d(x, z));
            all_equal &= va == vb;
            any_differs |= va != vc;
        }
        assert!(all_equal, "same seed must reproduce every sample");
        assert!(any_differs, "different seed must change the field");
    }

    #[test]
    fn octave_noise_range() {
        let world = world_positional(42);
        let noise = OctaveNoise::new(&continental_spec(), &mut world.from_name("continental"));
        // The stack output stays within the summed layer amplitudes.
        let bound = 2.0 + 1.0;
        for i in 0..400 {
            let x = i as f64 * 37.0;
            let z = i as f64 * -19.0;
            let v = noise.sample_2d(x, z).abs();
            assert!(v <= bound, "sample {v} beyond {bound} at ({x},{z})");
        }
    }
}
