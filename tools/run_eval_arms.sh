#!/usr/bin/env bash
# The model-side eval sweep over the arms trained at 1024 iterations (config/count_s1k.py).
# One process at a time: these runs share the GPU with nothing else, and the table reader already
# uses 8 CPU threads.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p eval_results logs

COMMON=(--device cuda --dtype bfloat16 --eval-tokens 131072 --batch-windows 8
        --corpus general=data/fineweb-edu/val.bin
        --corpus math=data/fineweb-edu/domain_math_val.bin
        --corpus law=data/fineweb-edu/domain_law_val.bin
        --corpus biomedical=data/fineweb-edu/domain_biomedical_val.bin
        --count-table count_tables/general150m
        --count-domain-table count_tables/math_s47
        --lambdas 0,0.25,0.5,1 --k-sweep 8,32 --j-sweep 1,6
        --fusion additive:0.1,multiplicative:0.1)

for ck in out_count_s1k out_count_noamp_s1k out_standard_s1k; do
    echo "=== $ck ==="
    venv/bin/python eval_lm.py --ckpt "$ck/ckpt.pt" "${COMMON[@]}" \
        --out "eval_results/lm_${ck}.json" 2>&1 | grep -v OpenBLAS | tee "logs/eval_lm_${ck}.log"
done
