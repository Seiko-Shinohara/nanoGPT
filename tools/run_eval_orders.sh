#!/usr/bin/env bash
# Sec. 5.2's shorter-history ablation, on the loss side.
#
# A COUNT lookup gathers J histories per position (order n_1 < ... < n_J) and averages their
# candidate sets. Appendix B.4 sweeps J and K at inference; Sec. 5.2 instead cuts the *longest*
# history off — max_order — which is the axis a deployed table actually trades: order 6 rows are
# the ones a 150M-token corpus has fewest of, and they are the ones the shrink ladder mostly deletes.
#
# Run on out_count_untied_s1k, the only checkpoint here whose count arm really works (README:
# count:lam0 1.7500 against standard 2.1209), at lambda=0, J=6, K=8, so each rung differs only in
# how deep the lookup is allowed to go. max_order=0 means "whatever the table was built with".
set -uo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
"$PY" eval_lm.py --ckpt out_count_untied_s1k/ckpt.pt \
    --device cuda --dtype bfloat16 --eval-tokens 131072 --batch-windows 8 \
    --corpus general=data/fineweb-edu/val.bin \
    --corpus math=data/fineweb-edu/domain_math_val.bin \
    --corpus law=data/fineweb-edu/domain_law_val.bin \
    --corpus biomedical=data/fineweb-edu/domain_biomedical_val.bin \
    --count-table count_tables/general150m --count-domain-table count_tables/math_s47 \
    --lambdas 0 --k-sweep 8 --j-sweep 6 --max-orders 0,2,3,4,5 \
    --out eval_results/lm_out_count_untied_s1k_orders.json 2>&1 \
  | grep -v OpenBLAS | tee logs/eval_lm_orders.log
