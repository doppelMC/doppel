import io
p = 'crates/doppel-oracle/src/parity_break.rs'
s = io.open(p, encoding='utf-8', newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')

# --- cut 1: session setup + both capture phases ---
c1_start = '''    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
'''
c1_end = '''    let (d_digger, d_witness) = worker
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??;
'''
i = s.index(c1_start)
j = s.index(c1_end) + len(c1_end)
c1 = s[i:j]

def dedent(block):
    return '\n'.join(ln[4:] if ln.startswith('    ') else ln for ln in block.split('\n'))

runner = '''/// Boots the reference, captures its break sessions, then does the same
/// against a fresh doppel over the clean blobs and pristine world.
fn run_break_sessions() -> Result<(
    Vec<bot::CapturedPacket>,
    Vec<bot::CapturedPacket>,
    Vec<bot::CapturedPacket>,
    Vec<bot::CapturedPacket>,
)> {
''' + dedent(c1) + '''    Ok((v_digger, v_witness, d_digger, d_witness))
}

'''
s = s[:i] + '    let (v_digger, v_witness, d_digger, d_witness) = run_break_sessions()?;\n\n' + s[j:]

# --- cut 2: the diagnostic summary printer ---
c2_start = '''    let histogram = |pkts: &[bot::CapturedPacket]| {
'''
c2_end_marker = '''        println!(
            "[oracle] {who} witness writes: {}",
            show_writes(&side.updates)
        );
'''
# extend to the end of the for loop over sides: find the loop close right after
i2 = s.index(c2_start)
# the block ends with the marker line; find the following '    }\n' closing the for
k = s.index(c2_end_marker, i2) + len(c2_end_marker)
# consume trailing newline + the closing brace of the for loop
close = '    }\n'
assert s[k:k+len(close)] == close, repr(s[k:k+20])
c2 = s[i2:k+len(close)]

summary = '''/// Prints per-side frame histograms and observed writes for both bots.
fn print_break_summary(sides: &[(&str, &[bot::CapturedPacket], &[bot::CapturedPacket], &Side)]) {
''' + dedent(c2) + '''}
'''
s = s[:i2] + '    print_break_summary(&[\n        ("vanilla", &v_digger, &v_witness, &v),\n        ("doppel", &d_digger, &d_witness, &d),\n    ]);\n' + s[k+len(close):]

anchor = '/// The differential breaking test.\npub fn parity_break()'
a = s.index(anchor)
s = s[:a] + runner + '\n' + summary + '\n' + s[a:]

if crlf:
    s = s.replace('\n', '\r\n')
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('extracted both')
