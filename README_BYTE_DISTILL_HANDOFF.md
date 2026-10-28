# Byte-Model Distillation Study — Handoff / Status README

Implementation of **"Breaking the Token Ceiling: Distilling Smaller, Stronger
Byte Models"** (arXiv:2609.12303) on top of this nanoGPT repo.

> **This file is a handoff document.** The previous session implemented and
> verified everything listed under "Done" below. Pick up at "Remaining work".
> The user is available for questions if a decision is genuinely ambiguous.

---

## 1. Environment (important)

| Item | Value |
|---|---|
| GPU to use | **HIP device 1** (AMD Radeon RX 7800 XT, 16 GB). Device 0 (7900 XTX) is occupied — always prefix commands with `export HIP_VISIBLE_DEVICES=1` (or it is set inside the shell scripts). |
| Python | `./venv/bin/python` (Python 3.14.7) |
| Key packages | torch 2.14.0+rocm, triton 3.8.0, numpy 2.5.3, tiktoken, matplotlib 3.11.2 (installed into the venv during this work) |
| Disk | ~54 GB free on /home/picode. The full teacher dump needs ~22 GB (see §6). |
| Data | `data/fineweb-small/` — 4,194,304 train / 1,048,576 val BPE tokens, vocab 4096 (256 raw bytes + 3839 word pieces + marker `<|docsplit|>` id 4095), byte fallback. Codec: `data/fineweb-small/codec.py`, vocab: `vocab.json`. |

### ROCm quirk that shaped the design
This torch+ROCm build has **no SDPA flash or mem-efficient attention kernels**
(`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` → "No available kernel"; the math
backend materializes the full (B, nh, T, T) matrix and OOMs). Solution:
`byte_distill/triton_attn.py` — hand-written **Triton flash attention
(fwd + bwd)**, verified against the math backend. `llama.py` uses it
automatically and falls back to a torch chunked attention if Triton is
unavailable.

---

## 2. File map (all new files)

```
byte_distill/
  __init__.py
  llama.py            Llama-3-style model: RMSNorm, RoPE, SwiGLU, no bias,
                      untied wte/lm_head, optional qk-norm, activation
                      checkpointing, Triton flash attention (or chunked fallback)
  tokenizer.py        ByteCodec: BPE token stream -> byte streams
                      (MI: 260-vocab, EOT: 261-vocab), special-token layout,
                      per-token byte lengths for BPB
  conversion.py       Marginalize-It + End-Of-Token logit conversion
                      (vectorized, precomputed live sets) + paper-pseudocode
                      reference implementation for tests
  test_conversion.py  8 unit tests (all passing)
  dump_teacher.py     teacher logit dump: byte dists (MI+EOT) + top-k tokens
  train_byte.py       training script for all six study configurations
  triton_attn.py      Triton flash attention (fwd+bwd), autograd.Function
  build_byte_streams.py  CLI to build byte_{mi,eot}_{train,val}.bin
  analyze.py          BPB-vs-FLOPs power-law fits + plots (bpb_vs_flops.png,
                      feather.png)
  smoke_test.sh       tiny end-to-end pipeline test (passing)
run_study.sh          full study orchestration (resumable, .done markers)
```

Existing repo files were **not modified** (vanilla nanoGPT `model.py`,
`train.py` etc. are untouched; `configurator.py` is reused as-is).

---

## 3. What the paper says (the parts we implement)

- **Two logit-conversion methods** (paper §2, App. F.1) turn a token-level
  teacher's next-token distribution into byte-level distributions:
  - **Marginalize-It** (approximate): at byte position k of the gold token,
    marginalize over tokens matching the gold prefix with length > k; mass of
    shorter prefix tokens is dropped and renormalized.
  - **End-Of-Token** (exact): same, but every token's symbol sequence is
    `bytes + [<eot>]`; tokens ending at position k put their mass on `<eot>`.
    No probability is lost: `prod_k dist_k[gold symbol] == P(gold)` (verified
    numerically, §5).
- **Six configurations** (paper Table 1): {token, byte, byte_eot} ×
  {supervised CE, distillation}. Distilled loss (paper eq. 3–5):
  `L = alpha * L_KL + (1-alpha) * L_CE`, forward KL over the full byte vocab
  (260/261) or top-k=600 token entries; distilled runs use alpha=1.
- **Architecture** (paper §A.1, Table 9): Llama-3 style (RMSNorm, RoPE,
  SwiGLU with d_ffn = ceil(4/3·d/256)·256, no bias, untied embeddings),
  qk-norm for byte-model stability (§A.4).
- **BPB** (paper eq. 6): tokenizer-independent bits-per-byte; special tokens
  (DOC/EOT/PAD/EOS/BOS) excluded from both the loss sum and the byte count.
- **FLOPs axis** (paper Table 9): `F_total = 3·(F_attn + F_qkvo + F_ffn +
  F_deemb)` with `F_attn = 2·n·d·(l+0.5)`, `F_qkvo = 8·n·d²`,
  `F_ffn = 4·n·d·d_ffn`, `F_deemb = 2·d·V` (per unit = per token/byte).
- **Optimization** (paper §A.3): AdamW β=(0.9, 0.95), weight decay 1e-4,
  grad clip 1.0, 10% linear warmup, cosine decay to lr/10. (Paper uses
  lr=4e-3 for 1.28B models; we use 2e-3 for the ~12–40M demo models.)

### Byte-stream layout (our convention, matches paper §2.2)
Special tokens: `PAD=256, EOS=257, BOS=258, DOC=259, EOT=260`.
- MI stream: each BPE token → its UTF-8 bytes; marker → `DOC`.
- EOT stream: each BPE token → bytes + `EOT`; marker → `DOC, EOT`.
Teacher-dump alignment: `Tdist[p]` = teacher distribution for the target at
byte position p; `TopkProbs[t]` = teacher distribution at token position t
(i.e. **predicting token t**, from teacher position t−1).

---

## 4. Done & verified (previous session)

1. **Conversion** — `test_conversion.py`: 8/8 pass, including the paper's
   "Tiramisu" worked example (MI B3: P(u)=0.8/P(k)=0.2; EOT B3:
   P(u)=2/3, P(k)=1/6, P(<eot>)=1/6; EOT B4: P(<eot>)=1) and the EOT
   exactness property for every token of the test vocab.
2. **Triton flash attention** — fwd+bwd match the math backend: fp32 max err
   ≤1.03e-5 (shapes up to (4,10,4096,64)); bf16 within bf16 precision
   (≤0.7% rel vs fp32 reference). ~11× faster than the torch chunked
   fallback. Kernel config (RDNA3 has 64 KB LDS): fwd BLOCK_M=128,
   BLOCK_N=64, num_stages=1; bwd BLOCK_M=64, BLOCK_N=64, num_stages=1.
3. **Llama model** — fwd/bwd/generate OK with and without activation
   checkpointing, with and without qk-norm, under bf16 autocast.
4. **Teacher dump** — EOT exactness verified on real dump data: product of
   teacher byte-probs over a token's byte sequence vs the teacher's token
   probability, max rel err 1.06e-3 (= fp16 precision) over 200 tokens.
5. **Smoke test** — `bash byte_distill/smoke_test.sh` passes end-to-end:
   tiny token teacher (CE) → dump (topk=64, 100k tokens) → EOT-distilled
   byte student → byte CE student; BPB + FLOPs logged to scaling.jsonl.
   Artifacts in `out/smoke/`.
6. **Byte streams built** — `out/byte_distill_study/byte_streams/`:
   MI train 10,235,683 / val 2,512,359 symbols; EOT train 14,429,987 /
   val 3,560,935 symbols.

### Bugs found & fixed (do not reintroduce)
- **RMSNorm under autocast**: `pow`/`rsqrt` are fp32 ops under autocast and
  fp32 weight × bf16 input promotes to fp32 → attention got mixed-dtype
  q/k/v. Fix: RMSNorm computes in fp32 and returns the input dtype.
- **`model(x)` without targets returns last-position-only logits** — anywhere
  full-position logits are needed (eval, dump) pass `targets` (we pass
  `x` itself).
- **2-D memmap reads need explicit `shape=`** (numpy infers 1-D otherwise).
- **`configurator.py` type-checks overrides** — defaults that get overridden
  by strings must be `''`, not `None` (byte_dir, teacher_dir).
- **Off-by-one in teacher alignment** — `TopkProbs[t]` predicts token t (not
  t+1); `get_teacher` uses rows `s+1 .. s+L` for a window starting at s.
- **Triton 3.8 API**: no `.dtype.element_ty` on tensor dtypes (use
  `.dtype`); backward kernels need smaller blocks than forward (64 KB LDS).

---

## 5. Remaining work (pick up here)

### Step 1 — Probe the teacher config (5 min)
The 40M teacher (batch 128×2048) OOM'd *before* Triton/checkpointing/
eval_batch_size existed; it has **not yet been re-verified** to fit in 16 GB.
Run:

```bash
export HIP_VISIBLE_DEVICES=1
./venv/bin/python byte_distill/train_byte.py \
  --tokenization=token --objective=ce --alpha=0.0 \
  --data_dir=data/fineweb-small \
  --n_layer=10 --n_head=10 --n_embd=640 --block_size=2048 --batch_size=128 \
  --learning_rate=2e-3 --max_iters=12 --eval_interval=100 --eval_iters=4 \
  --eval_batch_size=16 --log_interval=1 --out_dir=out/probe_teacher
```

Expect ~10–12 GB peak. If OOM: reduce `--batch_size` to 64 and set
`--grad_accum=2` in `run_study.sh` (teacher stage only).

### Step 2 — Launch the full study (background, ~3–5 h)
```bash
nohup bash run_study.sh > out/byte_distill_study/study.log 2>&1 &
```
`run_study.sh` is resumable (`.done` markers per stage; logs in
`out/byte_distill_study/logs/`). Stages:
1. byte streams (already built → skipped)
2. teacher: 40M Llama, token CE, 2500 steps, 128×2048, lr 2e-3
3. dump: full 4M-token train stream, topk=600 (~22 GB in
   `out/byte_distill_study/teacher_dumps/`)
4. six students (~12M each, 3000 steps, global batch ≈0.5M units):
   `token_ce`, `token_distill` (128×2048×2 accum), `byte_ce`,
   `byte_mi_distill` (56×9216), `byte_eot_ce`, `byte_eot_distill`
   (46×11264); qk-norm on the byte models (paper §A.4)
5. analyze: power-law fits + `bpb_vs_flops.png` + `feather.png`

### Step 3 — Monitor
- `tail -f out/byte_distill_study/study.log`
- Sanity: teacher val BPB should drop well below ~3.5 by step 2500; student
  losses should decrease monotonically-ish; no OOMs (watch logs).
- If a stage fails, fix and re-run `bash run_study.sh` (completed stages
  are skipped).

### Step 4 — Write the user-facing docs
- `README_byte_distill.md` (rename this file or split it): implementation
  notes + mapping to paper sections + how to run + **results** (table from
  `analyze.py`, the two plots, interpretation vs the paper's findings:
  token models plateau early, byte models catch up / cross over, EOT ≥ MI,
  distillation vs CE).
- Optionally: sample text from the byte models (decode the byte stream) as a
  qualitative check.

### Step 5 (optional, if time)
- α sweep (paper uses α as a free parameter; we only ran α=1).
- A second data scale (the paper's core axis is data scale; we have one).

---

## 6. Storage budget for the full dump (~22 GB)

| File | Shape | Dtype | Size |
|---|---|---|---|
| tdist_mi_train.fp16 | (10,235,683, 260) | fp16 | 5.3 GB |
| tdist_eot_train.fp16 | (14,429,987, 261) | fp16 | 7.5 GB |
| topk_probs_train.fp16 | (4,194,304, 600) | fp16 | 4.8 GB |
| topk_idx_train.int16 | (4,194,304, 600) | int16 | 4.8 GB |
| byte streams (4 files) | — | uint16 | 0.1 GB |

## 7. Quick reference: running individual pieces

```bash
export HIP_VISIBLE_DEVICES=1
# conversion unit tests
./venv/bin/python byte_distill/test_conversion.py
# tiny end-to-end smoke test (~5 min)
bash byte_distill/smoke_test.sh
# one student run (example)
./venv/bin/python byte_distill/train_byte.py \
  --tokenization=byte_eot --objective=distill --alpha=1.0 \
  --data_dir=data/fineweb-small \
  --byte_dir=out/byte_distill_study/byte_streams \
  --teacher_dir=out/byte_distill_study/teacher_dumps \
  --n_layer=6 --n_head=6 --n_embd=384 --block_size=11264 --batch_size=46 \
  --qk_norm=True --learning_rate=2e-3 --max_iters=3000 \
  --eval_interval=250 --eval_iters=200 --eval_batch_size=16 \
  --out_dir=out/byte_distill_study/byte_eot_distill
# analysis
./venv/bin/python byte_distill/analyze.py --study_dir out/byte_distill_study
```