#!/usr/bin/env bash
# Attention-backend ablation on the pinned HIP device (device 1 only).
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export HIP_VISIBLE_DEVICES=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

run () { echo "--- $1 ---"; shift; $PY tools/bench_train.py "$@" 2>&1 | grep -E "^\[|sanity|Error|error:|Traceback" ; echo; }

echo "############ ABLATION 1: attention backend (identical everything else) ############"
run "A  MATH  (nanoGPT's silent fallback today)"  --tag A_math        --sdpa math      --dtype bf16 --compile default
run "B  FLASH  (AOTriton)"                        --tag B_flash       --sdpa flash     --dtype bf16 --compile default
run "C  EFFICIENT (AOTriton mem-efficient)"       --tag C_efficient   --sdpa efficient --dtype bf16 --compile default

echo "############ ABLATION 2: dtype under FLASH ############"
run "D  FLASH fp16"   --tag D_flash_fp16   --sdpa flash --dtype fp16 --compile default
run "E  FLASH fp32"   --tag E_flash_fp32   --sdpa flash --dtype fp32 --compile default

echo "############ ABLATION 3: does torch.compile help on gfx1101? ############"
run "F  FLASH no-compile"  --tag F_flash_nocompile --sdpa flash --dtype bf16 --compile off

echo "############ ABLATION 4: micro-batch size (fixed total batch 96 seq) ############"
run "G  bs=8  accum=12"   --tag G_bs8        --sdpa flash --dtype bf16 --compile default --batch_size 8  --accum 12
run "H  bs=12 accum=8"    --tag H_bs12       --sdpa flash --dtype bf16 --compile default --batch_size 12 --accum 8
run "I  bs=24 accum=4"    --tag I_bs24       --sdpa flash --dtype bf16 --compile default --batch_size 24 --accum 4
run "J  bs=32 accum=3"    --tag J_bs32       --sdpa flash --dtype bf16 --compile default --batch_size 32 --accum 3

echo "############ ABLATION 5: TunableOp GEMM autotuning ############"
run "K  FLASH + TunableOp"  --tag K_tunable --sdpa flash --dtype bf16 --compile default --tunable
