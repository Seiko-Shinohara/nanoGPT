#!/usr/bin/env bash
# Second GPU queue on the pinned card (device 1, one job at a time).
#
# Part 1 is memory, not loss: 12-iteration probes whose only purpose is the
# [mem] block train.py now prints, i.e. the peak-device and host-resident bytes
# of each arm. Peak device memory is reached at the first optimizer step, so a
# short probe reports the same peak as the 1024-iteration run of that config.
#
# Part 2 is the architectural gap the earlier arms could not close: with nanoGPT's
# default tying, Eq. 1's basis B (the unembedding) *is* the input embedding, which
# is the token-embedding basis Appendix B.7 / Figure 18 measures against the paper's
# choice and finds worse. --tie_embeddings=False builds the arm the paper trains.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
S1K=config/count_s1k.py
NGRAM=config/count_ngram_s1k.py

probe() { # tag, config, out_dir, extra args...
  local tag=$1 conf=$2 dir=$3; shift 3
  echo "=== probe $tag ==="
  "$PY" train.py "$conf" --compile=False --max_iters=12 --eval_interval=100 \
      --out_dir="$dir" "$@" 2>&1 | grep -v OpenBLAS \
      | grep -E "\[mem\]|number of parameters|learned n-grams|Traceback|Error"
}

probe standard      "$S1K"  out_mem_standard --count_table= --count_domain_table= --count_amplify=False
probe count_amp     "$S1K"  out_mem_count
probe count_noamp   "$S1K"  out_mem_count_noamp --count_amplify=False
probe ngram_device  "$NGRAM" out_mem_ngram
probe ngram_host    "$NGRAM" out_mem_ngram_host --ngram_host_tables=True

echo "=== untied COUNT arm (the paper's unembedding basis) ==="
"$PY" train.py "$S1K" --compile=False --eval_interval=256 \
    --out_dir=out_count_untied_s1k --tie_embeddings=False 2>&1 \
  | grep -v OpenBLAS | tee logs/s1k_count_untied.log \
  | grep -E "\[mem\]|number of parameters|^step |done in|Traceback|Error"

echo "(queue 2 finished)"
