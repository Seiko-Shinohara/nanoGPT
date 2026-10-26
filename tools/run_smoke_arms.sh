#!/usr/bin/env bash
# Two smoke trainings, identical except for COUNT, on the allowed card (the RX 7800 XT).
# Sequential on purpose: one GPU job at a time on this box, and both runs share the CPU
# threads the COUNT prefetcher wants.
#
# Device note: amd_opt pins HIP_VISIBLE_DEVICES to the 7800 XT (index 1 on the box, index 0
# *in-process*), so inside these processes the allowed card is plain 'cuda'. Asking for cuda:1
# here is an "invalid device ordinal" crash, not a second GPU. The 7900 XTX stays untouched.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DEV=cuda

echo "=== COUNT smoke arm -> out_count_smoke ==="
venv/bin/python train.py config/count_smoke.py --device=$DEV --out_dir=out_count_smoke \
  2>&1 | tee logs/smoke_count.log

echo "=== standard smoke arm -> out_smoke_standard ==="
venv/bin/python train.py config/count_smoke.py --device=$DEV \
  --count_table= --count_domain_table= --count_lambda=0.0 --count_amplify=False \
  --out_dir=out_smoke_standard 2>&1 | tee logs/smoke_standard.log

echo "=== smoke training done ==="
