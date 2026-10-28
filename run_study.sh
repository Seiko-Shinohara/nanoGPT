#!/bin/bash
# =============================================================================
# End-to-end mini-study of "Breaking the Token Ceiling: Distilling Smaller,
# Stronger Byte Models" (arXiv:2609.12303) on the fineweb-small corpus
# (4M BPE tokens, vocab 4096 with byte fallback).
#
# Mirrors the paper's Table 1 with a 40M-parameter Llama-style teacher and
# six ~12M-parameter layer-parameter-matched students:
#
#   Stage 1: train token teacher (CE) on the BPE stream
#   Stage 2: dump teacher logits -> byte dists (Marginalize-It + End-Of-Token)
#            and top-k token logits (k=600, paper A.6)
#   Stage 3: train the six student configurations (3000 steps each):
#            token_ce, token_distill, byte_ce, byte_mi_distill,
#            byte_eot_ce, byte_eot_distill
#   Stage 4: fit BPB-vs-FLOPs power laws, write plots + comparison table
#
# Resumable: each stage is skipped if its .done marker exists.
# GPU: uses HIP device 1 (set below).
# =============================================================================
set -u
set -o pipefail
cd "$(dirname "$0")/.."
export HIP_VISIBLE_DEVICES=1
# ROCm: the default segmented caching allocator over-reserves (~15 GB reserved
# for ~11.5 GB allocated on the 40M teacher) and stutters. expandable_segments
# cuts reserved to ~12.8 GB at the same speed.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=./venv/bin/python
STUDY=out/byte_distill_study
LOGS=$STUDY/logs
mkdir -p "$LOGS" "$STUDY/byte_streams"

TEACHER_MAX_ITERS=2500
STUDENT_MAX_ITERS=3000

stage() { # $1 name, $2 marker, rest: command
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then
    echo "[skip] $name (marker $marker exists)"
    return 0
  fi
  echo "[start] $name"
  if "$@" 2>&1 | tee "$LOGS/$name.log"; then
    touch "$marker"
    echo "[done]  $name"
  else
    echo "[FAIL]  $name -- see $LOGS/$name.log"
    exit 1
  fi
}

# -----------------------------------------------------------------------------
# Stage 0: byte streams (Marginalize-It + End-Of-Token)
# -----------------------------------------------------------------------------
stage byte_streams "$STUDY/byte_streams/.done" \
  $PY byte_distill/build_byte_streams.py \
    --data_dir data/fineweb-small --out_dir $STUDY/byte_streams

# -----------------------------------------------------------------------------
# Stage 1: token teacher (40M, Llama-style, CE)
# -----------------------------------------------------------------------------
# NOTE: batch 128x2048 OOMs on 16 GB (peak ~19 GB: activations + bf16 logits
# + fp32 cross-entropy upcast). 64x2048 with grad_accum=2 keeps the same
# global batch (262,144 units/iter) at ~10 GB peak.
stage teacher "$STUDY/teacher/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=token --objective=ce --alpha=0.0 \
    --data_dir=data/fineweb-small \
    --n_layer=10 --n_head=10 --n_embd=640 --block_size=2048 --batch_size=64 \
    --grad_accum=2 \
    --learning_rate=2e-3 --max_iters=$TEACHER_MAX_ITERS \
    --eval_interval=250 --eval_iters=200 --eval_batch_size=16 --log_interval=50 \
    --compile=False \
    --out_dir=$STUDY/teacher

# -----------------------------------------------------------------------------
# Stage 2: teacher logit dump (byte dists MI+EOT, top-k=600)
# -----------------------------------------------------------------------------
stage dump "$STUDY/teacher_dumps/.done" \
  $PY byte_distill/dump_teacher.py \
    --ckpt=$STUDY/teacher/ckpt.pt --data_dir=data/fineweb-small \
    --out_dir=$STUDY/teacher_dumps --splits train --topk=600 \
    --block=1024 --device=cuda:0

# -----------------------------------------------------------------------------
# Stage 3: six student configurations (~12M each, 3000 steps)
# global batch ~= 0.5M units (paper A.4): token 256x2048, byte 56x9216,
# EOT 46x11264; qk-norm on byte models (paper A.4)
# -----------------------------------------------------------------------------
# NOTE: --compile=False: torch.compile (inductor) crashes on this ROCm build
# (Triton codegen fp32/fp64 loop-carried type error in the attention kernel).
COMMON="--n_layer=6 --n_head=6 --n_embd=384 --learning_rate=2e-3 \
  --max_iters=$STUDENT_MAX_ITERS --eval_interval=250 --eval_iters=200 \
  --log_interval=50 --data_dir=data/fineweb-small --compile=False"

# eval batch sizes (eval runs without activation checkpointing)
TOK_EVAL_BS=32
BYTE_EVAL_BS=16

stage token_ce "$STUDY/token_ce/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=token --objective=ce --alpha=0.0 \
    $COMMON --block_size=2048 --batch_size=128 --grad_accum=2 --eval_batch_size=$TOK_EVAL_BS \
    --out_dir=$STUDY/token_ce

stage token_distill "$STUDY/token_distill/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=token --objective=distill --alpha=1.0 --topk=600 \
    --teacher_dir=$STUDY/teacher_dumps \
    $COMMON --block_size=2048 --batch_size=128 --grad_accum=2 --eval_batch_size=$TOK_EVAL_BS \
    --out_dir=$STUDY/token_distill

stage byte_ce "$STUDY/byte_ce/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=byte --objective=ce --alpha=0.0 \
    --byte_dir=$STUDY/byte_streams \
    $COMMON --block_size=9216 --batch_size=56 --qk_norm=True --eval_batch_size=$BYTE_EVAL_BS \
    --out_dir=$STUDY/byte_ce

stage byte_mi_distill "$STUDY/byte_mi_distill/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=byte --objective=distill --alpha=1.0 \
    --byte_dir=$STUDY/byte_streams --teacher_dir=$STUDY/teacher_dumps \
    $COMMON --block_size=9216 --batch_size=56 --qk_norm=True --eval_batch_size=$BYTE_EVAL_BS \
    --out_dir=$STUDY/byte_mi_distill

stage byte_eot_ce "$STUDY/byte_eot_ce/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=byte_eot --objective=ce --alpha=0.0 \
    --byte_dir=$STUDY/byte_streams \
    $COMMON --block_size=11264 --batch_size=46 --qk_norm=True --eval_batch_size=$BYTE_EVAL_BS \
    --out_dir=$STUDY/byte_eot_ce

stage byte_eot_distill "$STUDY/byte_eot_distill/.done" \
  $PY byte_distill/train_byte.py \
    --tokenization=byte_eot --objective=distill --alpha=1.0 \
    --byte_dir=$STUDY/byte_streams --teacher_dir=$STUDY/teacher_dumps \
    $COMMON --block_size=11264 --batch_size=46 --qk_norm=True --eval_batch_size=$BYTE_EVAL_BS \
    --out_dir=$STUDY/byte_eot_distill

# -----------------------------------------------------------------------------
# Stage 4: scaling-law analysis + plots
# -----------------------------------------------------------------------------
stage analyze "$STUDY/analyze/.done" \
  $PY byte_distill/analyze.py --study_dir $STUDY --ref_flops 1e17

echo ""
echo "==================================================================="
echo "Study complete. Results:"
echo "  table + plots: $STUDY/{bpb_vs_flops.png,feather.png}"
echo "  per-run logs:  $STUDY/logs/"
echo "==================================================================="