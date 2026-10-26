#!/usr/bin/env bash
# Model-side table-size ablation: one checkpoint, four tables built from the same 150M tokens.
#
# eval_count.py --shrink measures the ladder table-side (coverage, candidate quality, S(w_true|h)).
# This is the loss-side view of the same question, on the arm where COUNT's gain is real: how much of
# untied `count:lam0`'s 0.371 BPB survives when the table gets 5x, 18x and 79x smaller? The rungs
# differ only in history_min_count -- the floor on how often a history must have been seen before the
# table stores a row for it -- so shrinking removes exactly the sparse tail the paper's Table 3 ladder
# is about.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
CK=out_count_untied_s1k/ckpt.pt

for table in general150m general150m_h20 general150m_h65 general150m_h265; do
  echo "=== $CK  <-  count_tables/$table ==="
  "$PY" eval_lm.py --ckpt "$CK" --device cuda --dtype bfloat16 \
      --eval-tokens 65536 --batch-windows 8 \
      --corpus general=data/fineweb-edu/val.bin \
      --corpus math=data/fineweb-edu/domain_math_val.bin \
      --count-table "count_tables/$table" \
      --lambdas 0 --k-sweep 8 --j-sweep 6 --fusion additive:0.1 \
      2>&1 | grep -v OpenBLAS | tee "logs/eval_lm_shrink_${table}.log" \
      | sed -n '/== Task loss/,/^$/p'
done

echo "(shrink ladder finished)"
