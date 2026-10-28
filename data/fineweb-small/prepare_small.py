"""
Data prep for the small-data (data-constrained) nanoGPT experiments.

Goal: take a ~4M-token slice of FineWeb-Edu and make it as information-dense as
possible, so a small model can be trained well on a *downsized* dataset.

Pipeline:
  1. stream documents from a few parquet shards (bounded memory)
  2. optional domain keyword filter (re-uses keyword lists from the existing
     domain_*_meta.json files)
  3. exact (sha1) + approximate (5-word shingle) deduplication
  4. train a small corpus-specific tokenizer: 256 raw byte ids + top-K learned
     word-aligned spans (1..5 words, ranked by tokens saved) + one reserved
     doc-split marker id
  5. doc-aware packing: every document is closed by the doc-split id, so a
     randomly sampled block sees "this document ended" instead of silently
     splicing two unrelated documents together
  6. write train.bin / val.bin (uint16) + meta.pkl (nanoGPT reads vocab_size from
     it automatically) + meta.json (tokenizer stats, chars-per-token)

The doc-split marker is a *regular learned vocabulary row*, not a decoding stop
sequence: the model must be able to emit it. If a corpus ever literally contained
the marker string, we refuse to reserve it (assertion in the scan loop).

Usage (from the repo root):
  venv/bin/python data/fineweb-small/prepare_small.py \
      --shards 013_00000.parquet --train-tokens 4194304 --val-tokens 1048576
  venv/bin/python data/fineweb-small/prepare_small.py \
      --keywords-from data/fineweb-edu/domain_math_meta.json --min-density 2.0 \
      --shards 001_00000.parquet 002_00000.parquet --out-dir data/fineweb-small-math
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from array import array

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(os.path.dirname(DATA_DIR), "fineweb-edu")

MARKER = "<|EOT|>"   # reserved doc-boundary marker (never used as a stop token)
N_BYTES = 256             # ids 0..255 are raw bytes: the lossless fallback
WORD_BASE = N_BYTES       # word pieces start here
MAX_PIECE_WORDS = 5       # a learned piece may span up to this many whole words
MAX_PIECE_CHARS = 32      # ... but no more than this many bytes; longer -> byte fallback
MIN_PIECE_SCORE = 8       # a single-word piece must save at least this many tokens to earn a slot
MIN_PHRASE_COUNT = 15     # a multi-word span must occur this often to earn a slot
WS_RUN = re.compile(r"\s+")


def collapse_ws(text):
    """Whitespace runs collapse to one space byte.

    Line breaks and indentation are redundant signal that would otherwise eat a
    large slice of a tiny token budget. Returns (normalized_text, raw_char_count).
    """
    raw = len(text)
    return WS_RUN.sub(" ", text).strip(), raw


def parse_args():
    p = argparse.ArgumentParser(description="downsize FineWeb-Edu into a dense small corpus")
    p.add_argument("--src", default=DEFAULT_SRC, help="dir holding the FineWeb-Edu parquet shards")
    p.add_argument("--shards", nargs="+", default=["013_00000.parquet"], help="shard file names to read")
    p.add_argument("--out-dir", default=DATA_DIR, help="where train.bin/val.bin/meta.* are written")
    p.add_argument("--vocab", type=int, default=4096, help="vocab size = byte ids + word pieces + marker")
    p.add_argument("--train-tokens", type=int, default=4194304, help="target training token count")
    p.add_argument("--val-tokens", type=int, default=1048576, help="target validation token count")
    p.add_argument("--char-headroom", type=float, default=4.0,
                   help="collect this many raw chars per target token before stopping the scan")
    p.add_argument("--unigram-share", type=float, default=0.85,
                   help="fraction of piece slots reserved for single words; the rest go to phrases")
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--max-docs", type=int, default=300_000, help="hard cap on docs scanned")
    p.add_argument("--keywords-from", default=None,
                   help="domain_*_meta.json whose keyword list defines the domain filter")
    p.add_argument("--min-density", type=float, default=0.0,
                   help="required keyword hits per 100 words (0 disables the filter)")
    p.add_argument("--near-dup", type=float, default=0.5,
                   help="fraction of a doc's 5-word shingles already seen that marks it a near-duplicate "
                        "(0 disables; exact dedup always runs)")
    p.add_argument("--val-bucket", type=int, default=8, help="1 in N docs goes to val (doc-level split)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def keyword_density(text, keywords):
    low = text.lower()
    hits = sum(low.count(k) for k in keywords)
    return (hits / max(1, text.count(" ") + 1)) * 100.0


def iter_docs(src, shards, max_docs):
    n = 0
    for name in shards:
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise SystemExit(f"shard not found: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text
                n += 1
                if n >= max_docs:
                    return


def main():
    args = parse_args()
    t0 = time.time()

    keywords = []
    if args.keywords_from:
        with open(args.keywords_from) as f:
            keywords = json.load(f).get("keywords", [])
        print(f"domain filter: {len(keywords)} keywords (min_density={args.min_density}/100 words)")

    split_id = args.vocab - 1
    word_slots = split_id - WORD_BASE
    if word_slots < 64:
        raise SystemExit("vocab too small to fit byte ids + word pieces + marker")

    seen_exact = set()
    shingle_seen = {} if args.near_dup > 0 else None
    stats = dict(docs_scanned=0, docs_too_short=0, docs_filtered=0, docs_exact_dup=0, docs_near_dup=0)
    train_docs, val_docs = [], []          # (normalized_doc, raw_chars)
    train_chars = val_chars = 0
    cap_train = args.train_tokens * args.char_headroom
    cap_val = args.val_tokens * args.char_headroom

    for text in iter_docs(args.src, args.shards, args.max_docs):
        stats["docs_scanned"] += 1
        if args.max_chars and len(text) > args.max_chars:
            text = text[:args.max_chars]
        doc, raw_chars = collapse_ws(text)
        if len(doc) < args.min_chars:
            stats["docs_too_short"] += 1
            continue
        if MARKER in doc:
            raise SystemExit(f"corpus literally contains the reserved marker {MARKER!r}; not using it")
        if keywords and args.min_density > 0 and keyword_density(doc, keywords) < args.min_density:
            stats["docs_filtered"] += 1
            continue

        h = hashlib.sha1(doc.encode("utf-8")).digest()
        if h in seen_exact:
            stats["docs_exact_dup"] += 1
            continue
        seen_exact.add(h)

        words = doc.split(" ")
        if shingle_seen is not None:
            shingles = {hashlib.sha1(" ".join(words[i:i + 5]).encode()).digest()
                        for i in range(max(0, len(words) - 4))}
            if shingles and sum(1 for s in shingles if s in shingle_seen) / len(shingles) >= args.near_dup:
                stats["docs_near_dup"] += 1
                continue
            shingle_seen.update(dict.fromkeys(shingles))

        # doc-level split: each document lands on exactly one side
        to_val = hashlib.sha1(h + b"|split").digest()[0] % args.val_bucket == 0
        if to_val:
            val_docs.append((doc, raw_chars))
            val_chars += len(doc)
        else:
            train_docs.append((doc, raw_chars))
            train_chars += len(doc)

        if stats["docs_scanned"] % 25_000 == 0:
            print(f"  scanned {stats['docs_scanned']:,} | train {len(train_docs):,} docs "
                  f"| exact-dup {stats['docs_exact_dup']:,} | near-dup {stats['docs_near_dup']:,}")
        if train_chars >= cap_train and val_chars >= cap_val:
            print(f"  reached char headroom after {stats['docs_scanned']:,} docs scanned")
            break

    kept = len(train_docs) + len(val_docs)
    print(f"kept {kept:,} docs = {len(train_docs):,} train / {len(val_docs):,} val "
          f"(exact-dup {stats['docs_exact_dup']:,}, near-dup {stats['docs_near_dup']:,} dropped)")

    # --- tokenizer: learned pieces are word-aligned spans (1..MAX_PIECE_WORDS words)
    # counted on TRAIN only. A piece earns its vocab slot by how many tokens it
    # saves: occurrences * (bytes - 1). This is what buys compression at a tiny vocab.
    counts = {}
    train_bytes = 0
    for doc, _raw in train_docs:
        words = doc.split(" ")
        n = len(words)
        train_bytes += len(doc)
        for i in range(n):
            acc = ""
            for j in range(i, min(i + MAX_PIECE_WORDS, n)):
                w = words[j]
                if not w:
                    break
                acc = acc + " " + w
                if len(acc) > MAX_PIECE_CHARS:
                    break
                key = acc.encode("utf-8")
                counts[key] = counts.get(key, 0) + 1

    # Slot allocation is deliberately unigram-first: coverage of common words is what
    # buys compression, and rare long spans crowd it out if they compete directly.
    uni, phrase = [], []
    for key, cnt in counts.items():
        blen = len(key)
        if blen < 2:
            continue
        score = cnt * (blen - 1)
        if key.count(0x20) == 1:                     # single word
            if score >= MIN_PIECE_SCORE:
                uni.append((score, blen, cnt, key))
        elif cnt >= MIN_PHRASE_COUNT:                 # multi-word span
            phrase.append((score, blen, cnt, key))
    uni.sort(reverse=True)
    phrase.sort(reverse=True)
    uni_slots = min(len(uni), int(word_slots * args.unigram_share))
    ranked = uni[:uni_slots] + phrase[:word_slots - uni_slots]

    vocab = {key: WORD_BASE + i for i, (_s, _b, _c, key) in enumerate(ranked)}
    inv = {v: k.decode("utf-8") for k, v in vocab.items()}
    inv[split_id] = MARKER
    max_piece_len = max((len(k) for k in vocab), default=1)
    selected_saving = sum(s for s, _b, _c, _k in ranked)
    byte_cov = min(1.0, sum(c * b for _s, b, c, _k in ranked) / max(1, train_bytes))
    print(f"tokenizer: {len(vocab):,} learned word-aligned pieces (upper-bound coverage "
          f"{100*byte_cov:.1f}% of train bytes), {N_BYTES} byte ids + 1 marker = {args.vocab} vocab; "
          f"{selected_saving/1e6:.1f}M tokens saved vs pure byte encoding")

    # --- greedy longest-match encode, byte fallback for anything unseen ---
    def encode(doc, out):
        data = (" " + doc).encode("utf-8")
        i, n = 0, len(data)
        while i < n:
            if data[i] == 0x20:                      # every word piece starts with a space
                for ln in range(min(max_piece_len, n - i), 1, -1):
                    vid = vocab.get(data[i:i + ln])
                    if vid is not None:
                        out.append(vid)
                        i += ln
                        break
                else:
                    out.append(0x20)
                    i += 1
            else:
                out.append(data[i])                  # byte fallback (rare words, non-ASCII)
                i += 1
        out.append(split_id)                          # close the document

    def encode_side(docs, budget):
        ids, used, chars_norm, chars_raw = array("H"), 0, 0, 0
        for doc, raw in docs:
            encode(doc, ids)
            used += 1
            chars_norm += len(doc)
            chars_raw += raw
            if len(ids) >= budget:
                break
        if len(ids) > budget:
            ids = ids[:budget]
        return ids, used, chars_norm, chars_raw

    train_ids, tr_docs, tr_chars, tr_raw = encode_side(train_docs, args.train_tokens)
    val_ids, va_docs, va_chars, va_raw = encode_side(val_docs, args.val_tokens)

    def mix(ids):
        n = max(1, len(ids))
        markers = sum(1 for t in ids if t == split_id)
        byte_ids = sum(1 for t in ids if t < N_BYTES)
        return n, byte_ids, markers, n - byte_ids - markers

    n_tr, b_tr, m_tr, w_tr = mix(train_ids)
    n_va, b_va, _m_va, _w_va = mix(val_ids)
    print(f"train {n_tr:,} tokens from {tr_docs:,} docs | val {n_va:,} tokens from {va_docs:,} docs")
    print(f"token mix: {100*w_tr/n_tr:.1f}% word pieces, {100*b_tr/n_tr:.1f}% byte fallback, "
          f"{100*m_tr/n_tr:.1f}% doc markers")

    os.makedirs(args.out_dir, exist_ok=True)
    tr = np.frombuffer(train_ids, dtype=np.uint16)
    va = np.frombuffer(val_ids, dtype=np.uint16)
    tr.tofile(os.path.join(args.out_dir, "train.bin"))
    va.tofile(os.path.join(args.out_dir, "val.bin"))
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": args.vocab}, f)

    meta = dict(stats)
    meta.update(dict(
        shards=args.shards, vocab_size=args.vocab, marker=MARKER, marker_id=split_id,
        byte_ids=N_BYTES, word_pieces=len(vocab), piece_byte_overlap=byte_cov,
        tokens_saved_vs_bytes=selected_saving,
        docs_train=tr_docs, docs_val=va_docs, tokens_train=n_tr, tokens_val=n_va,        chars_norm_per_token=(tr_chars + va_chars) / max(1, n_tr + n_va),
        chars_raw_per_token=(tr_raw + va_raw) / max(1, n_tr + n_va),
        tokens_per_vocab_entry=n_tr / args.vocab,
        word_token_share=w_tr / n_tr, byte_fallback_share=b_tr / n_tr, marker_share=m_tr / n_tr,
        min_density=args.min_density, near_dup_threshold=args.near_dup, max_chars=args.max_chars,
        keywords=keywords, seed=args.seed, wall_seconds=time.time() - t0,
    ))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump({"vocab_size": args.vocab, "marker": MARKER, "marker_id": split_id,
                   "id_to_piece": inv}, f)

    print(f"wrote {args.out_dir}: train {n_tr/1e6:.2f}M tokens ({tr.nbytes/1e6:.1f}MB), "
          f"val {n_va/1e6:.2f}M tokens ({va.nbytes/1e6:.1f}MB), {time.time()-t0:.0f}s total")
    print(f"chars per token: {meta['chars_norm_per_token']:.2f} normalized / "
          f"{meta['chars_raw_per_token']:.2f} raw")
    print(f"average exposure per vocab entry: {meta['tokens_per_vocab_entry']:.0f} training occurrences")


if __name__ == "__main__":
    main()
