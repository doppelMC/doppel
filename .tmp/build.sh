#!/bin/sh
# Local build helper: mingw toolchain via zig wrappers, shared target dir.
export PATH="/c/Users/nour/tools/bin:$PATH"
export CC="C:\\Users\\nour\\tools\\bin\\zigcc.exe"
export AR="C:\\Users\\nour\\tools\\bin\\ar.exe"
export CARGO_TARGET_DIR=target-gnu
exec cargo +1.98.0-x86_64-pc-windows-gnu "$@"
