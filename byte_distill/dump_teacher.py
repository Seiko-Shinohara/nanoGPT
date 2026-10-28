"""
Offline teacher logit dumping (paper Section 1.1, A.6).

Runs a token-level teacher model over the BPE token stream and writes:

  1. Byte distributions for the Marginalize-It student   (M_mi, 260)  fp16
  2. Byte distributions for the End-Of-Token student     (M_eot, 261) fp16
  3. Top-k token probabilities + indices for the token-distilled student
     (N, k) fp16 + (N, k) int32   (paper: k = 600)

Alignment:
  Tdist_mi[p]  = teacher distribution (over the 260-byte vocab) for the
                 target at byte-stream position p
  Tdist_eot[p] = same over the 261-byte vocab (EOT = 260)
  TopkProbs[t] = teacher distribution (top-k) at token position t, i.e. for
                 predicting token t (from teacher position t-1)

Byte positions inside the very first token of the stream have no teacher
distribution (left as zeros) and are masked out in the training loss.

The teacher is evaluated in blocks of B target tokens with B tokens of left
context (input length 2B = the teacher's block size), so every target (except
the first B-1 of the stream) is conditioned on its true preceding context.

Usage:
  HIP_VISIBLE_DEVICES=1 python byte_distill/dump_teacher.py \
      --ckpt out/byte_distill_study/teacher/ckpt.pt \
      --data_dir data/fineweb-small \
      --out_dir out/byte_distill_study/teacher_dumps \
      --topk 600 --device cuda:0
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ByteCodec, VOCAB_BYTE, VOCAB_BYTE_EOT, write_meta  # noqa: E402
from conversion import precompute_live_sets, first_symbol_matrix, convert_chunk  # noqa: E402
from llama import Llama, LlamaConfig  # noqa: E402


def load_teacher(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    config = LlamaConfig(**{k: model_args[k] for k in
                            ("block_size", "vocab_size", "n_layer", "n_head",
                             "n_embd", "d_ffn", "dropout", "rope_base",
                             "qk_norm", "tie_weights") if k in model_args})
    model = Llama(config)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def dump_split(model, codec, stream, out_dir, split, topk, B, device,
               live_sets_mi, first_mat_mi, live_sets_eot, first_mat_eot,
               max_len_mi, max_len_eot, max_tokens=None):
    """Dump one split (train or val). Returns meta dict."""
    N = len(stream)
    if max_tokens is not None:
        N = min(N, max_tokens)
        stream = stream[:N]
    V = codec.vocab_size

    # byte streams + alignment
    n_mi, starts_mi = codec.build_byte_stream(stream, os.path.join(out_dir, f"byte_mi_{split}.bin"), use_eot=False)
    n_eot, starts_eot = codec.build_byte_stream(stream, os.path.join(out_dir, f"byte_eot_{split}.bin"), use_eot=True)
    print(f"[{split}] byte streams: MI={n_mi:,} symbols, EOT={n_eot:,} symbols")

    tdist_mi = np.memmap(os.path.join(out_dir, f"tdist_mi_{split}.fp16"), dtype=np.float16, mode="w+", shape=(n_mi, VOCAB_BYTE))
    tdist_eot = np.memmap(os.path.join(out_dir, f"tdist_eot_{split}.fp16"), dtype=np.float16, mode="w+", shape=(n_eot, VOCAB_BYTE_EOT))
    topk_probs = np.memmap(os.path.join(out_dir, f"topk_probs_{split}.fp16"), dtype=np.float16, mode="w+", shape=(N, topk))
    topk_idx = np.memmap(os.path.join(out_dir, f"topk_idx_{split}.int16"), dtype=np.int16, mode="w+", shape=(N, topk))

    lens_mi = codec.lens
    lens_eot = codec.lens + 1

    t0 = time.time()
    n_blocks = (N + B - 1) // B
    with torch.no_grad():
        for b in range(n_blocks):
            a = b * B
            hi = min(N, a + B)
            n_tgt = hi - a
            lo = max(0, a - B)
            x = stream[lo:hi]
            if len(x) < 2 * B:
                x = np.pad(x, (2 * B - len(x), 0), constant_values=0)
            xt = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(xt, xt)                  # (1, 2B, V) fp32 (targets=xt forces full logits)
            probs = torch.softmax(logits[:, B - 1: B - 1 + n_tgt, :], dim=-1).squeeze(0)  # (n_tgt, V)
            gold = torch.tensor(stream[a:hi], dtype=torch.int64, device=device)

            # top-k token dump (aligned to token positions a..hi-1)
            vals, idx = torch.topk(probs, topk, dim=-1)
            topk_probs[a:hi] = vals.cpu().numpy().astype(np.float16)
            topk_idx[a:hi] = idx.cpu().numpy().astype(np.int16)

            # byte distribution dumps (aligned to byte-stream positions)
            dists_mi = convert_chunk(probs, gold, live_sets_mi, first_mat_mi, max_len_mi, VOCAB_BYTE)
            dists_eot = convert_chunk(probs, gold, live_sets_eot, first_mat_eot, max_len_eot, VOCAB_BYTE_EOT)
            rows = np.arange(n_tgt)
            for k in range(max_len_mi):
                m = lens_mi[gold.cpu().numpy()] > k
                if m.any():
                    tdist_mi[starts_mi[a:hi][m] + k] = dists_mi[rows[m], k].cpu().numpy().astype(np.float16)
            for k in range(max_len_eot):
                m = lens_eot[gold.cpu().numpy()] > k
                if m.any():
                    tdist_eot[starts_eot[a:hi][m] + k] = dists_eot[rows[m], k].cpu().numpy().astype(np.float16)

            if (b + 1) % 50 == 0 or b + 1 == n_blocks:
                dt = time.time() - t0
                print(f"[{split}] block {b + 1}/{n_blocks} ({(b + 1) / n_blocks * 100:.0f}%), "
                      f"{dt:.0f}s elapsed, {n_tgt * (b + 1) / dt:,.0f} tokens/s")
    tdist_mi.flush(); tdist_eot.flush(); topk_probs.flush(); topk_idx.flush()
    for mm in (tdist_mi, tdist_eot, topk_probs, topk_idx):
        del mm
    return dict(n_tokens=N, n_bytes_mi=n_mi, n_bytes_eot=n_eot, topk=topk,
                max_len_mi=max_len_mi, max_len_eot=max_len_eot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument("--block", type=int, default=1024, help="B: target tokens per block (input length 2B)")
    ap.add_argument("--max_tokens", type=int, default=None, help="only dump the first N tokens (debugging)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(1337)

    codec = ByteCodec(os.path.join(args.data_dir, "vocab.json"))
    model = load_teacher(args.ckpt, device)

    sym_mi, lens_mi = codec.symbol_table(use_eot=False)
    sym_eot, lens_eot = codec.symbol_table(use_eot=True)
    live_sets_mi = precompute_live_sets(sym_mi, lens_mi)
    live_sets_eot = precompute_live_sets(sym_eot, lens_eot)
    first_mat_mi = torch.tensor(first_symbol_matrix(sym_mi, VOCAB_BYTE), dtype=torch.float32)
    first_mat_eot = torch.tensor(first_symbol_matrix(sym_eot, VOCAB_BYTE_EOT), dtype=torch.float32)
    max_len_mi, max_len_eot = sym_mi.shape[1], sym_eot.shape[1]
    print(f"vocab={codec.vocab_size}, max_len MI={max_len_mi} EOT={max_len_eot}, "
          f"live sets precomputed for both variants")

    meta = dict(codec_vocab=codec.vocab_size, marker_id=codec.marker_id,
                vocab_byte=VOCAB_BYTE, vocab_byte_eot=VOCAB_BYTE_EOT)
    for split in args.splits:
        stream = np.memmap(os.path.join(args.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        meta[split] = dump_split(model, codec, stream, args.out_dir, split, args.topk,
                                 args.block, device, live_sets_mi, first_mat_mi,
                                 live_sets_eot, first_mat_eot, max_len_mi, max_len_eot,
                                 max_tokens=args.max_tokens)
        del stream
    write_meta(os.path.join(args.out_dir, "meta.pkl"), **meta)
    print(f"dump complete -> {args.out_dir}")


if __name__ == "__main__":
    main()