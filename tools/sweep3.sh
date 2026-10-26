#!/usr/bin/env bash
# Does AOTriton flash pay off end-to-end, with and without torch.compile?
# Conservative: NANOAMD_MAX_VRAM_GIB=9 caps this process at 9 of 16 GiB, leaving
# ~7 GiB for the desktop compositor that shares this GPU (card1 DP-3 is connected).
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
run () { tag=$1; shift; echo "--- $tag ---"; env "$@" 2>&1 | grep -E "^\[|sanity|amd_opt|OOM|Error|Traceback|no kernel" ; echo; }

CAP=NANOAMD_MAX_VRAM_GIB=9
B=4; AC=4; W="--warmup 3 --iters 12"
echo "######## attention backend, all at bs=4 so peak stays under the cap ########"
run "1 eager  MATH  (stock nanoGPT on this card)" $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 $PY tools/bench_train.py --tag eager_math  --sdpa math  --dtype bf16 --compile off     --batch_size $B --accum $AC $W
run "2 eager  FLASH (AOTriton unlocked)"          $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY tools/bench_train.py --tag eager_flash --sdpa flash --dtype bf16 --compile off     --batch_size $B --accum $AC $W
run "3 compile MATH"                              $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 $PY tools/bench_train.py --tag comp_math  --sdpa math  --dtype bf16 --compile default  --batch_size $B --accum $AC $W
run "4 compile FLASH"                             $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY tools/bench_train.py --tag comp_flash --sdpa flash --dtype bf16 --compile default  --batch_size $B --accum $AC $W

echo "######## the fp32 trap: what a dtype=float32 config costs on this build ########"
run "5 compile fp32"                              $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY tools/bench_train.py --tag comp_fp32  --sdpa efficient --dtype fp32 --compile default --batch_size $B --accum $AC $W

echo "######## best path at bs=8, with and without TunableOp ########"
run "6 compile FLASH bs8"                        $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY tools/bench_train.py --tag comp_flash_bs8 --sdpa flash --dtype bf16 --compile default --batch_size 8 --accum $AC $W
run "7 compile FLASH bs8 + TunableOp"            $CAP TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 $PY tools/bench_train.py --tag comp_flash_tunable --sdpa flash --dtype bf16 --compile default --batch_size 8 --accum $AC $W --tunable
