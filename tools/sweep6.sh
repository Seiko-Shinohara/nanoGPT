#!/usr/bin/env bash
# Two questions that the micro-batch sweep left open:
#   1. How much of an iteration is the lm_head GEMM + cross_entropy over a 50304-wide
#      vocab? (that is what OOMs 16 GB at micro-batch 12, and what a chunked loss
#      would remove)
#   2. Is fp16 actually faster than bf16 at micro-batch 8, the measured sweet spot?
# Ablation: identical model, vocab 1024 instead of 50304. Anything the two runs
# differ by is head + loss.
cd "$(dirname "$0")/.."
set -u
CAP=NANOAMD_MAX_VRAM_GIB=8
W="--warmup 3 --iters 12"

run() {
    echo "--- $* ---"
    env "$@" 2>&1 | grep -E "^\[|sanity|OOM|Error|Traceback|no kernel"
}

echo "=== GPU state before the runs ==="
venv/bin/amd-smi static 2>/dev/null | grep -iE "GPU\[1\]|Percent|Busy|VRAM|Usage" | head -8
free -m | sed -n '2p'

# head + loss cost, compiled, bf16
run $CAP venv/bin/python tools/bench_train.py --tag comp_bf16_v1k --compile default \
    --sdpa flash --tunable --batch_size 8 --dtype bf16 --vocab_size 1024 $W
# head + loss cost, compiled, fp16
run $CAP venv/bin/python tools/bench_train.py --tag comp_fp16_v1k --compile default \
    --sdpa flash --tunable --batch_size 8 --dtype fp16 --vocab_size 1024 $W
# full vocab, fp16, apples to apples with the 662 ms bf16 run
run $CAP venv/bin/python tools/bench_train.py --tag comp_fp16_bs8 --compile default \
    --sdpa flash --tunable --batch_size 8 --dtype fp16 $W
# eager mode head cost (cheap, and where unfused cost shows up most)
run $CAP venv/bin/python tools/bench_train.py --tag eager_bf16_v1k --compile off \
    --sdpa flash --batch_size 8 --dtype bf16 --vocab_size 1024 $W
