#!/usr/bin/env bash
# Arm sweeps on the three checkpoints queue 1 trained, one GPU job at a time on device 1.
#
# out_standard_s1k_v2 is the faithful standard arm (input embeddings exempt from weight decay,
# Table 2), so it gets the full lambda / J / K sweep: it is the reference every other arm's
# "bolted onto a model that never saw a table" claim is measured against.
#
# The two learned n-gram arms get a short sweep. Their interesting rows are the checkpoint's own
# trained path, the ablation that withdraws q_t (the counterpart of withdrawing f_t), and what
# COUNT's tables do when bolted onto a baseline that was trained without them.
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

full() { # out_dir
  local d=$1
  echo "=== full sweep: $d ==="
  "$PY" eval_lm.py --ckpt "$d/ckpt.pt" $COMMON \
      --lambdas 0,0.25,0.5,1 --k-sweep 8,32 --j-sweep 1,6 2>&1 \
    | grep -v OpenBLAS | tee "logs/eval_lm_${d}.log" | sed -n '/== Task loss/,/^$/p'
}

short() { # out_dir
  local d=$1
  echo "=== baseline sweep: $d ==="
  "$PY" eval_lm.py --ckpt "$d/ckpt.pt" $COMMON \
      --lambdas 0,0.25 --k-sweep 8 --j-sweep 6 2>&1 \
    | grep -v OpenBLAS | tee "logs/eval_lm_${d}.log" | sed -n '/== Task loss/,/^$/p'
}

full  out_standard_s1k_v2
short out_ngram_s1k
short out_ngram_host_s1k

echo "(eval queue finished)"
