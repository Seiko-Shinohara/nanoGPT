#!/usr/bin/env bash
# Third GPU queue (device 1, one job at a time).
#
# Part 1 re-measures iteration time for three arms *in one session*: this box's
# median iteration time drifts by up to 50% between sessions (the pinned card also
# drives the desktop), so cross-session medians are not comparable. 64 iterations,
# no evaluation in the middle, same config otherwise.
#
# Part 2 runs the full arm sweep on the untied COUNT checkpoint -- the arm whose
# f_t is built from an unembedding the input embedding is free to disagree with.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
S1K=config/count_s1k.py
NGRAM=config/count_ngram_s1k.py
COMMON="--device cuda --dtype bfloat16 --eval-tokens 131072 --batch-windows 8 \
  --corpus general=data/fineweb-edu/val.bin \
  --corpus math=data/fineweb-edu/domain_math_val.bin \
  --corpus law=data/fineweb-edu/domain_law_val.bin \
  --corpus biomedical=data/fineweb-edu/domain_biomedical_val.bin \
  --count-table count_tables/general150m --count-domain-table count_tables/math_s47 \
  --lambdas 0,0.25,0.5,1 --k-sweep 8,32 --j-sweep 1,6 \
  --fusion additive:0.1,multiplicative:0.1"

timing() { # tag, config, extra...
  local tag=$1 conf=$2; shift 2
  echo "=== timing $tag ==="
  "$PY" train.py "$conf" --compile=False --max_iters=64 --eval_interval=1000 \
      --eval_only=False --log_interval=8 --out_dir="out_timing_$tag" "$@" 2>&1 \
    | grep -v OpenBLAS | tee "logs/timing_$tag.log" \
    | grep -E "^iter 6[0-9]:|number of parameters" | tail -3
}

timing standard     "$S1K"  --count_table= --count_domain_table= --count_amplify=False
timing count       "$S1K"
timing count_noamp "$S1K"  --count_amplify=False
timing ngram       "$NGRAM"
timing ngram_host  "$NGRAM" --ngram_host_tables=True
timing untied      "$S1K"  --tie_embeddings=False

echo "=== arm sweep on the untied checkpoint ==="
"$PY" eval_lm.py --ckpt out_count_untied_s1k/ckpt.pt $COMMON 2>&1 \
  | grep -v OpenBLAS | tee logs/eval_lm_out_count_untied_s1k.log \
  | sed -n '/== Task loss/,/^$/p'

echo "(queue 3 finished)"
