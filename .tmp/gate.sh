#!/bin/sh
# Local parity-worldgen gate runner: pinned JDK, toolchain on PATH.
export JAVA_BIN="C:/Program Files/Eclipse Adoptium/jdk-25.0.2.10-hotspot/bin/java.exe"
export PATH="/c/Users/nour/tools/bin:$PATH"
export DOPPEL_BIN="C:/Users/nour/doppel/.claude/worktrees/agent-a890de0190b75ff50/target-gnu/release/doppel.exe"
cd "C:/Users/nour/doppel/.claude/worktrees/agent-a890de0190b75ff50"
exec ./target-gnu/release/doppel-oracle.exe "$@"
