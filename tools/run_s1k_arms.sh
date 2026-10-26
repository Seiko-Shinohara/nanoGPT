#!/usr/bin/env bash
# Three arms at 1024 iterations, identical except for the COUNT switches. Sequential: one GPU job at
# a time on this box, and the COUNT arms want the CPU threads for their prefetchers.
#
# amd_opt pins HIP_VISIBLE_DEVICES to the RX 7800 XT (box index 1, in-process index 0), so 'cuda'
# here *is* the allowed card and 'cuda:1' would be an invalid-ordinal crash. The 7900 XTX is left
# alone.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CFG=config/count_s1k.py
DEV=cuda

run () { # run <out_dir> [extra args...]
  local out=$1; shift
  echo "=== $out ==="
  venv/bin/python train.py "$CFG" --device=$DEV --out_dir="$out" "$@" 2>&1 | tee "logs/${out}.log"
}

run out_count_s1k
run out_count_noamp_s1k --count_amplify=False
run out_standard_s1k --count_table= --count_domain_table= --count_lambda=0.0 --count_amplify=False

echo "=== arms done ==="
