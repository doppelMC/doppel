import io
p = 'crates/doppel-oracle/src/parity_survival.rs'
s = io.open(p, encoding='utf-8', newline='').read()
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')

marker = '''    // six observed tails show). The resting cadences - the same periodic'''
i = s.index(marker)

# the section begins at the comment block start a few lines earlier
start = s.rfind('    // ', 0, i - 200)
# find the true section comment start: search backwards for the blank line
bl = s.rfind('\n\n', 0, i)
section = s[bl+2:i] if s[bl+2:i].lstrip().startswith('//') else None
cut = bl + 2 if section else i

end_marker = '''    if failures.is_empty() {
        println!("PASS: survival parity");
'''
j = s.index(end_marker)
body = s[cut:j]

def dedent(t):
    return '\n'.join(ln[4:] if ln.startswith('    ') else ln for ln in t.split('\n'))

tails_fn = '''/// Compares the drop fall/slide and rest motion tails, the grass
/// cycles, and the inventory syncs.
fn compare_drop_tails(
    v: &Obs,
    d: &Obs,
    v_walk: &Obs,
    d_walk: &Obs,
    v_wit: &Obs,
    d_wit: &Obs,
    v_all: &Obs,
    d_all: &Obs,
    mut failures: &mut Vec<String>,
) {
''' + dedent(body) + '''}
'''
call = '''    compare_drop_tails(
        &v,
        &d,
        &v_walk,
        &d_walk,
        &v_wit,
        &d_wit,
        &v_all,
        &d_all,
        &mut failures,
    );
'''
s = s[:cut] + call + s[j:]

anchor = '/// The differential survival test.\npub fn parity_survival()'
a = s.index(anchor)
s = s[:a] + tails_fn + '\n' + s[a:]

if crlf:
    s = s.replace('\n', '\r\n')
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('tails split done')
