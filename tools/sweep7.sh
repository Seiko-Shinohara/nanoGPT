#!/usr/bin/env bash
# Device-0 sweep: the RX 7900 XTX (gfx1100, 24 GB, headless).
#
# READ THIS FIRST: this script runs on GPU 0. GPU 0 is where the LLM lives. It prints
# the card's free VRAM and then keeps going -- it does not stop you, because you asked
# for commands, not guardrails. Free VRAM under ~14 GiB means micro-batch 16 will fail
# and under ~20 GiB means micro-batch 32 will fail. Kill the LLM first.
#
# Every run below uses 256 sequences per iteration, so tokens/iter (262144) and the
# 7600-iteration schedule in config/train_fineweb_edu_amd.py stay identical no matter
# which micro-batch you are timing. That makes the ms/iter numbers directly comparable.
#
# Run one line at a time if you want to watch the card: each bench_train.py line is
# self-contained, and this script is just the list.

set -u
cd "$(dirname "$0")/.."

export NANOAMD_DEVICE=0
PY="venv/bin/python"

free=$($PY -c "pass" >/dev/null 2>&1; venv/bin/amd-smi metric -g 0 2>/dev/null | awk '/FREE_VRAM/{print $2; exit}')
echo "device 0 free VRAM: ${free:-unknown} MB  (needs ~14000 at micro-batch 16, ~20000 at 32)"
echo

B="$PY tools/bench_train.py --block_size 1024 --compile default --tunable --warmup 12 --iters 8"

echo "== 1. micro-batch ladder, same 256 sequences/iter =="
$B --batch_size 8  --accum 32 --tag "mb8  x32 bf16"
$B --batch_size 16 --accum 16 --tag "mb16 x16 bf16"
$B --batch_size 32 --accum 8  --tag "mb32 x8  bf16"

echo
echo "== 2. eager vs compiled at micro-batch 8 (7800 XT: eager won by 1.78x at bs=4) =="
$B --batch_size 8 --accum 32 --compile off --sdpa flash --tag "mb8  x32 eager"

echo
echo "== 3. fp16 vs bf16 at micro-batch 16 (fp16 was +4% at micro-batch 12) =="
$B --batch_size 16 --accum 16 --dtype fp16 --tag "mb16 x16 fp16"

echo
echo "== 4. how much of an iteration is lm_head + cross_entropy? =="
# --synthetic generates token ids on the GPU, so a 1024-wide vocab cannot index-fault on
# real 50256-valued ids. Same model, same shapes, only the vocab (and therefore the head
# GEMM + the softmax over its logits) changes. head share = (t50304 - t1024) / t50304.
$B --batch_size 16 --accum 16 --synthetic --vocab_size 50304 --tag "head v50304"
$B --batch_size 16 --accum 16 --synthetic --vocab_size 1024  --tag "head v1024"

echo
echo "== 5. is the memmap random read costing anything? (same shape, synthetic data) =="
$B --batch_size 16 --accum 16 --synthetic --vocab_size 50304 --tag "mb16 synthetic"

echo
echo "== 6. the two VRAM levers, measured instead of estimated =="
# Identical shapes; only the head/chunking settings change. Compare ms/iter AND the peak
# allocated/reserved lines, because the two levers buy different things: chunking shrinks the
# transient around cross_entropy, checkpointing is what stops storing block activations.
# At bs=16/T=1024 there are 16384 rows, so --chunked-head-rows 4096 is 4 chunks.
$B --batch_size 16 --accum 16                               --tag "mb16 stock head"
$B --batch_size 16 --accum 16 --chunked-head-rows 4096      --tag "mb16 head/4096"
$B --batch_size 16 --accum 16 --activation-checkpoint       --tag "mb16 block checkpoint"
$B --batch_size 16 --accum 16 --chunked-head-rows 4096 --activation-checkpoint --tag "mb16 both"

