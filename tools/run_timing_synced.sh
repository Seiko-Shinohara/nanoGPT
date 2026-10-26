#!/usr/bin/env bash
# Same-session throughput comparison of every arm, with --sync_timing=True so dt is
# GPU wall time rather than CPU enqueue time (see the knob's comment in train.py).
#
# Protocol follows the paper's Sec. 6: median over optimization steps 50-100. Each arm
# runs 100 iterations with no evaluation in the middle; tools/median_iters.py then takes
# the median of iterations 52-100 and reports each arm's share of the standard arm.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
S1K=config/count_s1k.py
NGRAM=config/count_ngram_s1k.py

timing() { # tag, config, extra...
  local tag=$1 conf=$2; shift 2
  echo "=== timing $tag ==="
  "$PY" train.py "$conf" --compile=False --max_iters=100 --eval_interval=1000 \
      --eval_only=False --log_interval=4 --sync_timing=True \
      --out_dir="out_timing_$tag" "$@" 2>&1 | grep -v OpenBLAS \
      | tee "logs/timing_$tag.log" | grep -E "^iter 100:|number of parameters" | tail -2
}

timing standard     "$S1K"   --count_table= --count_domain_table= --count_amplify=False
timing count       "$S1K"
timing count_noamp "$S1K"   --count_amplify=False
timing count_untied "$S1K"  --tie_embeddings=False
timing ngram       "$NGRAM"
timing ngram_host  "$NGRAM" --ngram_host_tables=True

"$PY" tools/median_iters.py logs/timing_standard.log logs/timing_count.log \
    logs/timing_count_noamp.log logs/timing_count_untied.log \
    logs/timing_ngram.log logs/timing_ngram_host.log

echo "(timing queue finished)"
