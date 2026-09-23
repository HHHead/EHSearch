#!/usr/bin/env bash
#
# example_run.sh - minimal end-to-end walkthrough of the EHSearch driver.
#
# It expects an fvecs/ivecs dataset laid out as
#
#   $DATA/<name>_base.fvecs
#   $DATA/<name>_query.fvecs
#   $DATA/<name>_groundtruth.ivecs
#
# and reproduces the three stages used for the paper's measurements:
#   1. build and cache the graph topology
#   2. (out-of-core backend) materialise the encrypted payload file
#   3. sweep one parameter family and write a CSV
#
# Override any of the variables below, e.g.
#   DATA=./data/gist NAME=gist NQ=1000 ./scripts/example_run.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./build/ehsearch}
DATA=${DATA:-./data/sift}
NAME=${NAME:-sift}
NQ=${NQ:-2000}

BASE="$DATA/${NAME}_base.fvecs"
QUERY="$DATA/${NAME}_query.fvecs"
GT="$DATA/${NAME}_groundtruth.ivecs"

CACHE="build/graph_${NAME}.bin"
PAYLOAD="build/payload_${NAME}.bin"
OUT="results/${NAME}.csv"

for f in "$BASE" "$QUERY" "$GT"; do
    [ -s "$f" ] || { echo "missing input: $f"; exit 1; }
done
mkdir -p build results

echo "=== 1/3  build the graph topology (cached) ==="
"$BIN" hnsw --base "$BASE" --cache "$CACHE" --M 32 --efC 64

echo "=== 2/3  prepare the encrypted payload file (out-of-core backend) ==="
"$BIN" prepare --base-raw "$BASE" --nbase 1000000 --payload-file "$PAYLOAD"

echo "=== 3/3  sweep one parameter family ==="
"$BIN" bench \
    --base "$BASE" --query "$QUERY" --gt "$GT" \
    --cache "$CACHE" \
    --m-list 16 --bps 8 \
    --bs-list 1500,3000 \
    --r-list 0,1 \
    --tau-list 1,5,10,20,30,40,50 \
    --anchor-level 1 \
    --k 100 --k-eval 10 \
    --nq "$NQ" --nq-1t 200 \
    --tag "$NAME" \
    --out "$OUT"

echo "=== done: $OUT ==="
echo
echo "To repeat with the out-of-core backend, append:"
echo "    --backend ssd --payload-file $PAYLOAD"
echo "To inspect index structure only (no queries), use:"
echo "    $BIN bench --hgi-only --cache $CACHE --bs-list 500,1000,2000,4000"
