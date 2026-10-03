#!/bin/sh
# One-off probe driver: the seed-signature sweep with redecoration.
cd "$(dirname "$0")/.." || exit 1
DIAG_FEATURE=dark_forest_vegetation \
DIAG_BLOCK=minecraft:dark_oak_log \
sh .tmp/build.sh test -p doppel-world probe_feature_seed_index -- \
    --ignored --nocapture > .tmp/probe2.log 2>&1
echo DONE-PROBE2
