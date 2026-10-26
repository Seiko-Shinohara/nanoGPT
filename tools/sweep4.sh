#!/usr/bin/env bash
# Is eager+AOTriton-flash really better than torch.compile on this build?
# Compare at bs=8 (the batch where compiled flash hit 35.89 TFLOP/s), plus try
# max-autotune, which may pick the AOTriton kernel as a template instead of
# decomposing SDPA itself.
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
run () { tag=$1; shift; echo "--- $tag ---"; env "$@" 2>&1 | grep -E "^\[|sanity|amd_opt|OOM|Error|Traceback|no kernel" ; echo; }

CAP=NANOAMD_MAX_VRAM_GIB=8           # display GPU ceiling; do not raise it
W="--warmup 3 --iters 12"
E1=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

run "8 eager FLASH bs8"                $CAP $E1 $PY tools/bench_train.py --tag eager_flash_bs8   --sdpa flash --dtype bf16 --compile off     --batch_size 8 --accum 4 $W
run "9 eager FLASH bs8 + TunableOp"    $CAP $E1 $PY tools/bench_train.py --tag eager_flash_tun    --sdpa flash --dtype bf16 --compile off     --batch_size 8 --accum 4 $W --tunable
run "10 compile max-autotune FLASH"    $CAP $E1 $PY tools/bench_train.py --tag comp_maxauto       --sdpa flash --dtype bf16 --compile max-autotune --batch_size 8 --accum 4 $W
run "11 eager FLASH bs12 (headroom?)"  $CAP $E1 $PY tools/bench_train.py --tag eager_flash_bs12   --sdpa flash --dtype bf16 --compile off     --batch_size 12 --accum 4 $W
