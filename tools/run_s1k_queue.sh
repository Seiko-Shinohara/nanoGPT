#!/usr/bin/env bash
# Sequential s1k arms on the pinned GPU (device 1 only -- one training job at a
# time, by request), with a VRAM sampler so the table-memory claim of the
# learned n-gram baseline can be measured rather than asserted.
#
#   1. s1k_standard_v2   : the faithful standard arm (input embeddings now
#                          exempt from weight decay, as Table 2 specifies for
#                          every arm)
#   2. s1k_ngram         : learned n-gram baseline (Eq. 3), tables on GPU
#   3. s1k_ngram_host    : same, tables pinned host-side (the paper's point)
#
# usage: bash tools/run_s1k_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-$HOME/.cache/torch/kernels}
S1K=config/count_s1k.py

sample_vram() {
  while :; do
    venv/bin/amd-smi metric --mem-usage 2>/dev/null | grep -oE '[0-9]+ MiB' | head -1
    sleep 15
  done > "logs/vram_$1.csv" &
  echo $!
}

run_arm() {
  local tag=$1
  shift
  echo "=== $tag ==="
  local pid
  pid=$(sample_vram "$tag")
  "$PY" train.py "$@" 2>&1 | grep -v OpenBLAS | tee "logs/${tag}.log" \
    | grep -E "parameters|learned n-grams|n-gram tables|COUNT|val loss|mem:|done in|mfu" | tail -24
  kill "$pid" 2>/dev/null
  sort -rn -k1 logs/vram_"$tag".csv | head -1 | sed "s/^/$tag peak VRAM sample: /"
}

# the standard arm is the COUNT arm with no tables: --count_table= is what
# switches COUNT off (count_enabled = bool(count_table)), everything else shared
run_arm s1k_standard_v2 "$S1K" --compile=False \
    --eval_interval=256 --out_dir=out_standard_s1k_v2 \
    --count_table= --count_domain_table= --count_lambda=0.0 --count_amplify=False

run_arm s1k_ngram config/count_ngram_s1k.py --compile=False \
    --eval_interval=256 --out_dir=out_ngram_s1k

run_arm s1k_ngram_host config/count_ngram_s1k.py --compile=False \
    --eval_interval=256 --out_dir=out_ngram_host_s1k --ngram_host_tables=True

echo "(queue finished)"
