#!/usr/bin/env bash
# Throughput + VRAM sweep on the pinned HIP device (RX 7800 XT only).
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export HIP_VISIBLE_DEVICES=1
E1=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
run () { tag=$1; shift; echo "--- $tag ---"; env "$@" 2>&1 | grep -E "^\[|sanity|Error|Traceback|AttributeError" ; echo; }

echo "########## honest 'before' baseline: nanoGPT silently on MATH attention ##########"
run "M0 math bs8"  $E1=0 $PY tools/bench_train.py --tag M0_math_bs8 --sdpa math --dtype bf16 --compile default --batch_size 8 --accum 6

echo "########## FLASH bf16: how big a micro-batch fits in 16 GiB? ##########"
for bs in 8 12 16 24 32; do
  case $bs in 8) ac=6;; 12) ac=4;; 16) ac=3;; 24) ac=2;; 32) ac=2;; esac
  run "F1 bf16 flash bs$bs" $E1=1 $PY tools/bench_train.py --tag F1_bf16_bs$bs --sdpa flash --dtype bf16 --compile default --batch_size $bs --accum $ac
done

echo "########## FLASH fp16 ##########"
for bs in 8 12 16 24 32; do
  case $bs in 8) ac=6;; 12) ac=4;; 16) ac=3;; 24) ac=2;; 32) ac=2;; esac
  run "F2 fp16 flash bs$bs" $E1=1 $PY tools/bench_train.py --tag F2_fp16_bs$bs --sdpa flash --dtype fp16 --compile default --batch_size $bs --accum $ac
done

echo "########## expandable_segments to fight fragmentation at the big batches ##########"
for bs in 24 32 40; do
  case $bs in 24) ac=2;; 32) ac=2;; 40) ac=2;; esac
  run "F3 bf16 flash bs$bs expandable" $E1=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True $PY tools/bench_train.py --tag F3_bf16_bs$bs_exp --sdpa flash --dtype bf16 --compile default --batch_size $bs --accum $ac
done
