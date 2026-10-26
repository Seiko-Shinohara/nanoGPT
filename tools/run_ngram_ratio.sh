#!/usr/bin/env bash
# Eq. 3's sizing rule, taken as a knob instead of a constant.
#
# Sec. 3.3 sizes the learned n-gram tables by a ratio, not by a row count: "all input-token
# embeddings plus all n-gram table parameters equal all remaining learned parameters, including the
# output head and the projections" — i.e. table params : backbone = 1 : 1. The paper never sweeps
# that ratio, so the 1 : 1 rule is the only point on the curve we are allowed to quote. Sweeping it
# here answers the local question: at this budget, is the baseline starved for table parameters, or
# is 1 : 1 already more than a 4L/4H/256d network can use?
#
# tools/run_timing_synced.sh already timed ratios 1 / 4 / 8 at 100 iterations; this is the loss side
# of the same ladder at the arm's real length (1024 iterations), sparse retrieved-row updates, tables
# on the device, everything else identical to config/count_ngram_s1k.py.
#
# One GPU job at a time on device 1. Ratios are floats: the configurator asserts on type, so the
# overrides must be 4.0 and 8.0, not 4 and 8.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
COMMON="--device cuda --dtype bfloat16 --eval-tokens 131072 --batch-windows 8 \
  --corpus general=data/fineweb-edu/val.bin \
  --corpus math=data/fineweb-edu/domain_math_val.bin \
  --corpus law=data/fineweb-edu/domain_law_val.bin \
  --corpus biomedical=data/fineweb-edu/domain_biomedical_val.bin \
  --count-table count_tables/general150m --count-domain-table count_tables/math_s47 \
  --fusion additive:0.1,multiplicative:0.1"

# Ratio 1.0 is already trained and evaluated as out_ngram_s1k; retraining it here would only
# reproduce that directory under a new name.
for r in 4.0 8.0; do
  tag=${r%.0}
  out=out_ngram_r${tag}_s1k
  echo "=== train ratio $r -> $out ==="
  "$PY" train.py config/count_ngram_s1k.py \
      --ngram_ratio=$r --out_dir=$out --sync_timing=True 2>&1 \
    | grep -v OpenBLAS | tee "logs/ngram_ratio_${tag}.log" \
    | grep -E "learned n-grams|^step [0-9]+: train loss|\[mem\]|step 1024"
  echo "=== eval $out ==="
  "$PY" eval_lm.py --ckpt "$out/ckpt.pt" $COMMON \
      --lambdas 0,0.25 --k-sweep 8 --j-sweep 6 2>&1 \
    | grep -v OpenBLAS | tee "logs/eval_lm_${out}.log" | sed -n '/== Task loss/,/^$/p'
done

echo "(n-gram ratio ladder finished)"
