"""
Prepare the downloaded FinewebEDU parquet shards into nanoGPT's uint16 memmap format.

This machine's constraint: the LLM serving on GPU 0 already holds most of the 62 GB of
system RAM (~7 GB available, zram swap already full). nanoGPT's get_batch() does RANDOM
access into train.bin via np.memmap, so if train.bin is bigger than the free page cache
every batch stalls on NVMe random reads. Size the dataset to fit in RAM.

Usage:
    python data/fineweb-edu/prepare.py                     # default: ~2B tokens (4 GB)
    python data/fineweb-edu/prepare.py --max-tokens 4e9    # ~8 GB
    python data/fineweb-edu/prepare.py --shards 2          # only read the first 2 shards

Only the listed shards are read, streaming one row group at a time, so peak RSS stays
well under a single shard's decompressed size.
"""
import os
import re
import sys
import time
import argparse
import pickle

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = 50304  # GPT-2's 50257 padded up to a multiple of 64: faster matmuls
_NO_SPECIAL = frozenset()  # tiktoken rejects a plain tuple here; frozenset is accepted
BPE_VOCAB = 50257

enc = tiktoken.get_encoding("gpt2")


def natural_key(name):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-tokens', type=float, default=2e9,
                    help='total train tokens to write (uint16, so 2 bytes each)')
    ap.add_argument('--val-tokens', type=float, default=2e7, help='tokens held out for val.bin')
    ap.add_argument('--shards', type=int, default=0, help='limit to first N shards (0 = all)')
    ap.add_argument('--batch', type=int, default=20000, help='docs per tiktoken encode_batch call')
    ap.add_argument('--max-docs-per-shard', type=int, default=0, help='debug cap')
    args = ap.parse_args()

    shards = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')], key=natural_key)
    if not shards:
        sys.exit("no .parquet shards found in data/fineweb-edu/ -- download them first")
    if args.shards:
        shards = shards[:args.shards]
    print(f"reading {len(shards)} shard(s): {shards[0]} ... {shards[-1]}")

    train_path = os.path.join(DATA_DIR, 'train.bin')
    val_path = os.path.join(DATA_DIR, 'val.bin')
    for p in (train_path, val_path):
        if os.path.exists(p):
            os.remove(p)

    max_train = int(args.max_tokens)
    max_val = int(args.val_tokens)
    n_train = n_val = 0
    n_docs = 0
    t0 = time.time()

    train_f = open(train_path, 'wb')
    val_f = open(val_path, 'wb')

    for si, shard in enumerate(shards):
        if n_train >= max_train:
            break
        pf = pq.ParquetFile(os.path.join(DATA_DIR, shard))
        cols = pf.schema_arrow.names
        text_col = 'text' if 'text' in cols else cols[0]
        print(f"[{si+1}/{len(shards)}] {shard}: {pf.num_row_groups} row groups, cols={cols}")

        for rg_i in range(pf.num_row_groups):
            if n_train >= max_train:
                break
            table = pf.read_row_group(rg_i, columns=[text_col])
            texts = table[text_col].to_pylist()
            # FinewebEDU rows are already .text extracted; strip control chars like the
            # upstream prepare script does so tiktoken output stays in-range.
            texts = [t for t in texts if t]
            docs_done = 0
            for start in range(0, len(texts), args.batch):
                chunk = texts[start:start + args.batch]
                # Some FinewebEDU docs contain the end-of-text marker as literal text.
                # Encode it as ordinary BPE tokens instead of raising; never let it
                # become a real special token, and never print raw document text.
                tok_lists = enc.encode_batch(chunk, allowed_special=_NO_SPECIAL,
                                             disallowed_special=_NO_SPECIAL)
                buf_parts = []
                for toks in tok_lists:
                    buf_parts.append(np.asarray(toks, dtype=np.uint16))
                    n_docs += 1
                flat = np.concatenate(buf_parts) if len(buf_parts) > 1 else buf_parts[0]

                # split off a slice for the val set until it is filled
                room_val = max_val - n_val
                if room_val > 0:
                    take = min(room_val, flat.size // 10)  # ~10% until val is full
                    val_f.write(flat[:take].tobytes())
                    n_val += take
                    flat = flat[take:]

                room_train = max_train - n_train
                take = min(room_train, flat.size)
                train_f.write(flat[:take].tobytes())
                n_train += take
                docs_done += len(chunk)

                if n_train >= max_train:
                    break
            del table, texts
            rate = n_train / (time.time() - t0)
            print(f"    row group {rg_i+1}/{pf.num_row_groups}: {n_train/1e9:.3f}B train tokens, "
                  f"{n_val/1e6:.1f}M val, {n_docs} docs, {rate/1e3:.0f}k tokens/s")

    train_f.close()
    val_f.close()

    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': VOCAB_SIZE}, f)

    dt = time.time() - t0
    print(f"\nwrote {n_train/1e9:.3f}B train tokens ({n_train*2/1e9:.2f} GB) and "
          f"{n_val/1e6:.1f}M val tokens ({n_val*2/1e6:.2f} MB) over {n_docs} docs in {dt:.0f}s")
    print(f"tiktoken BPE vocab = {BPE_VOCAB}, writing meta.pkl vocab_size = {VOCAB_SIZE} (multiple of 64)")
    if n_train >= max_train:
        print("hit --max-tokens limit; remaining shards were not read")


if __name__ == '__main__':
    main()
