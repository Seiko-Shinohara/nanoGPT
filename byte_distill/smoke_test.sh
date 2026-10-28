#!/bin/bash
# Tiny end-to-end smoke test of the byte-distillation pipeline on GPU (HIP device 1).
# Trains a 2-layer teacher, dumps its logits, then trains one byte-distilled
# student (EOT) and one byte CE student for a few steps each.
set -e
cd "$(dirname "$0")/.."
export HIP_VISIBLE_DEVICES=1
PY=./venv/bin/python
OUT=out/smoke
rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== [1/4] train tiny token teacher (CE) ==="
$PY byte_distill/train_byte.py \
  --tokenization=token --objective=ce --alpha=0.0 \
  --data_dir=data/fineweb-small \
  --n_layer=2 --n_head=4 --n_embd=128 --block_size=512 --batch_size=8 \
  --learning_rate=2e-3 --max_iters=60 --eval_interval=30 --eval_iters=8 \
  --log_interval=10 --out_dir=$OUT/teacher --compile=False 2>&1 | tail -6

echo "=== [2/4] dump teacher logits (topk=64, first 100k tokens) ==="
$PY byte_distill/dump_teacher.py \
  --ckpt=$OUT/teacher/ckpt.pt --data_dir=data/fineweb-small \
  --out_dir=$OUT/dumps --splits train val --max_tokens=100000 \
  --topk=64 --block=256 --device=cuda:0 2>&1 | tail -8

echo "=== [3/4] train tiny EOT-distilled byte student ==="
$PY byte_distill/train_byte.py \
  --tokenization=byte_eot --objective=distill --alpha=1.0 \
  --data_dir=data/fineweb-small --byte_dir=$OUT/dumps --teacher_dir=$OUT/dumps \
  --n_layer=2 --n_head=4 --n_embd=128 --block_size=1024 --batch_size=8 \
  --qk_norm=True --learning_rate=2e-3 --max_iters=40 --eval_interval=20 --eval_iters=8 \
  --log_interval=10 --out_dir=$OUT/eot_distill --compile=False 2>&1 | tail -6

echo "=== [4/4] train tiny byte CE student ==="
$PY byte_distill/train_byte.py \
  --tokenization=byte --objective=ce --alpha=0.0 \
  --data_dir=data/fineweb-small --byte_dir=$OUT/dumps \
  --n_layer=2 --n_head=4 --n_embd=128 --block_size=1024 --batch_size=8 \
  --qk_norm=True --learning_rate=2e-3 --max_iters=40 --eval_interval=20 --eval_iters=8 \
  --log_interval=10 --out_dir=$OUT/byte_ce --compile=False 2>&1 | tail -6

echo "=== scaling logs ==="
for d in teacher eot_distill byte_ce; do
  echo "--- $d ---"
  cat $OUT/$d/scaling.jsonl 2>/dev/null | tail -3
done
echo "SMOKE TEST COMPLETE"