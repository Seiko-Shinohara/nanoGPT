#!/usr/bin/env bash
# Final sweep: honest stock baseline, and how far micro-batch size carries in
# compiled mode (per-micro-batch time was flat from bs=4 to bs=8).
# Cap 12 GiB of 16 = the ceiling agreed for this GPU, leaving ~4 GiB for GNOME.
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
run () { tag=$1; shift; echo "--- $tag ---"; env "$@" 2>&1 | grep -E "^\[|sanity|OOM|Error|Traceback|no kernel" ; echo; }

CAP=NANOAMD_MAX_VRAM_GIB=8
E1=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
E0=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
W="--warmup 3 --iters 12"

echo "######## what stock nanoGPT (compile=True, no flash unlock) costs at bs=8 ########"
run "12 compile MATH bs8  (stock)"     $CAP $E0 $PY tools/bench_train.py --tag comp_math_bs8     --sdpa math  --dtype bf16 --compile default --batch_size 8 --accum 4 $W
run "13 compile MATH bs8 +Tunable"     $CAP $E0 $PY tools/bench_train.py --tag comp_math_bs8_tun --sdpa math  --dtype bf16 --compile default --batch_size 8 --accum 4 $W --tunable

echo "######## push micro-batch: compiled FLASH + TunableOp ########"
run "14 compile FLASH bs12 +Tunable"   $CAP $E1 $PY tools/bench_train.py --tag comp_flash_bs12   --sdpa flash --dtype bf16 --compile default --batch_size 12 --accum 4 $W --tunable
run "15 compile FLASH bs16 +Tunable"   $CAP $E1 $PY tools/bench_train.py --tag comp_flash_bs16   --sdpa flash --dtype bf16 --compile default --batch_size 16 --accum 4 $W --tunable
run "16 compile fp16  bs12 +Tunable"   $CAP $E1 $PY tools/bench_train.py --tag comp_fp16_bs12    --sdpa flash --dtype fp16 --compile default --batch_size 12 --accum 4 $W --tunable
