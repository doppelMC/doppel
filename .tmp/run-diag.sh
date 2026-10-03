#!/bin/sh
# One-off diagnostics driver: seed probe then chunk divergence dump.
cd "$(dirname "$0")/.." || exit 1
DIAG_FEATURE=dark_forest_vegetation \
DIAG_BLOCK=minecraft:dark_oak_log \
sh .tmp/build.sh test -p doppel-world probe_feature_seed_index -- \
    --ignored --nocapture > .tmp/probe-vegetation.log 2>&1
DIAG_ORDERS=capture \
sh .tmp/build.sh test -p doppel-world diagnose_chunk_divergence -- \
    --ignored --nocapture > .tmp/diag-chunk.log 2>&1
echo DONE-BOTH
