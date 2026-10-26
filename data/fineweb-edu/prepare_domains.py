"""Build held-out *domain* slices out of the local FineWeb-Edu shards (no downloads).

Sec. 5.1 of the paper mixes a general count table with a domain count table -- Math, Biomedical,
Law -- at inference time, with no retraining and no new table format. To reproduce that here we
need a domain corpus that is (a) local and (b) disjoint from the general table's tokens, so that
"the domain table knows things the general table does not" is a real property and not leakage.

The local corpus is `data/fineweb-edu/*.parquet` (14 shards of raw text). Documents are classified
by keyword density -- the paper describes its own domain selection as keyword/topic based -- and
only documents that pass are tokenized. Accepted documents are shuffled deterministically and
split so that the validation slice contains documents the count table never saw.

    venv/bin/python data/fineweb-edu/prepare_domains.py --domain math --shards 2 --tokens 40e6
    venv/bin/python data/fineweb-edu/prepare_domains.py --domain biomedical --shards 2 --tokens 40e6

Outputs, in this directory:
    domain_<name>_train.bin   uint16 tokens used to *build* the domain count table
    domain_<name>_val.bin     uint16 tokens from held-out documents, used only for evaluation
    domain_<name>_meta.json   document counts, keyword thresholds, token counts

Peak memory stays at one row group because the shards are streamed the same way prepare.py does.
"""

import os
import re
import sys
import json
import time
import random
import argparse

import numpy as np
import pyarrow.parquet as pq
import tiktoken

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = 50304
enc = tiktoken.get_encoding("gpt2")

# Keyword sets. Deliberately narrow: a document has to talk about the domain, not mention it once.
DOMAINS = {
    "math": [
        "theorem", "lemma", "proof", "conjecture", "manifold", "topology", "algebra", "geometry",
        "polynomial", "derivative", "integral", "matrix", "vector space", "banach", "hilbert space",
        "convergence", "asymptotic", "eigenvalue", "group action", "ring", "field extension",
        "differential equation", "fourier", "combinatoric", "graph theory", "prime number",
    ],
    "biomedical": [
        "patient", "patients", "clinical", "trial", "dose", "dosage", "therapy", "treatment",
        "diagnosis", "diagnostic", "biopsy", "tumor", "carcinoma", "receptor", "protein", "gene",
        "expression", "cell line", "antibody", "cohort", "randomized", "placebo", "symptom",
        "pathway", "biomarker", "infection", "immune", "plasma", "mortality", "biography",
    ],
    "law": [
        "court", "courts", "statute", "statutory", "plaintiff", "defendant", "judgment", "verdict",
        "litigation", "attorney", "counsel", "contract", "clause", "jurisdiction", "appeal",
        "amendment", "legislation", "regulation", "regulatory", "ordinance", "tort", "liability",
        "copyright", "patent", "licensor", "arbitration", "testimony", "felony", "sentencing",
    ],
}


def natural_key(name):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def keyword_stats(text: str, patterns, words_total: int):
    """Distinct keywords hit and hits per 1000 words -- cheap, and enough to separate the domains."""
    hits = 0
    distinct = 0
    for word, pattern in patterns:
        n = text.count(word)
        if n:
            distinct += 1
            hits += n
    return distinct, (1000.0 * hits / max(1, words_total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--shards", type=int, default=2, help="how many leading parquet shards to scan")
    ap.add_argument("--shard-range", default="", help="instead of --shards, scan shards A:B (0-based, B excluded), e.g. 4:8")
    ap.add_argument("--suffix", default="", help="append to the output names, e.g. --suffix _s47 keeps a second slice of the same domain beside the first")
    ap.add_argument("--tokens", type=float, default=40e6, help="target training tokens")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--min-density", type=float, default=3.0, help="keyword hits per 1000 words")
    ap.add_argument("--min-distinct", type=int, default=4, help="distinct keywords required")
    ap.add_argument("--min-chars", type=int, default=800, help="skip stub documents")
    ap.add_argument("--max-chars", type=int, default=12000, help="cap document length")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    patterns = [(w, re.compile(r"\b" + re.escape(w), re.IGNORECASE)) for w in DOMAINS[args.domain]]
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")], key=natural_key)
    if not shards:
        sys.exit("no .parquet shards found")
    if args.shard_range:
        a, b = args.shard_range.split(":")
        shards = shards[int(a):int(b)]
        if not shards:
            sys.exit(f"--shard-range {args.shard_range} selected no shards")
    else:
        shards = shards[: args.shards]
    tag = args.suffix
    print(f"[{args.domain}] scanning {len(shards)} shard(s): {shards}")

    rng = random.Random(args.seed)
    docs = []
    tokens_seen = 0
    n_docs_all = 0
    n_docs_kept = 0
    t0 = time.time()

    for shard in shards:
        path = os.path.join(DATA_DIR, shard)
        pf = pq.ParquetFile(path)
        for rg_index in range(pf.num_row_groups):
            table = pf.read_row_group(rg_index)
            col = table.column("text").to_pylist()
            table = None  # release the row group before tokenizing
            for text in col:
                n_docs_all += 1
                if not text or len(text) < args.min_chars:
                    continue
                if len(text) > args.max_chars:
                    text = text[: args.max_chars]
                words = len(text) // 6
                distinct, density = keyword_stats(text, patterns, words)
                if distinct < args.min_distinct or density < args.min_density:
                    continue
                n_docs_kept += 1
                ids = enc.encode(text, disallowed_special=_none())
                if len(ids) < 64:
                    continue
                docs.append(ids)
                tokens_seen += len(ids)
            if tokens_seen >= args.tokens:
                break
        print(f"[{args.domain}] {shard}: docs={n_docs_all:,} kept={n_docs_kept:,} "
              f"tokens={tokens_seen:,}  {time.time() - t0:.1f}s")
        if tokens_seen >= args.tokens:
            break

    if not docs:
        sys.exit(f"[{args.domain}] no documents matched -- loosen --min-density/--min-distinct")

    rng.shuffle(docs)
    n_val_docs = max(1, int(len(docs) * args.val_fraction))
    val_docs, train_docs = docs[:n_val_docs], docs[n_val_docs:]

    def write(name, doc_list):
        if not doc_list:
            arr = np.zeros(0, dtype=np.uint16)
        else:
            arr = np.concatenate([np.asarray(d, dtype=np.uint16) for d in doc_list])
            arr = arr[: int(args.tokens)]
        out = os.path.join(DATA_DIR, name)
        arr.tofile(out)
        return arr.size

    tag = args.suffix  # --suffix keeps a second slice of the same domain beside the first,
                       # e.g. a shard-disjoint slice whose documents the general count table never
                       # saw. Without it, a second extraction silently overwrites the first.
    n_train = write(f"domain_{args.domain}{tag}_train.bin", train_docs)
    n_val = write(f"domain_{args.domain}{tag}_val.bin", val_docs)
    meta = {
        "domain": args.domain,
        "shards": shards,
        "suffix": tag,
        "keywords": DOMAINS[args.domain],
        "min_density": args.min_density,
        "min_distinct": args.min_distinct,
        "docs_scanned": n_docs_all,
        "docs_kept": n_docs_kept,
        "docs_train": len(train_docs),
        "docs_val": len(val_docs),
        "tokens_train": n_train,
        "tokens_val": n_val,
        "vocab_size": VOCAB_SIZE,
        "tokenizer": "tiktoken gpt2",
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(DATA_DIR, f"domain_{args.domain}{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.domain}] wrote domain_{args.domain}{tag}_train.bin ({n_train:,} tokens, "
          f"{len(train_docs):,} docs) and domain_{args.domain}{tag}_val.bin ({n_val:,} tokens, "
          f"{len(val_docs):,} docs)")


def _none():
    return frozenset()


if __name__ == "__main__":
    main()
