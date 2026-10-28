"""
Build the byte streams (Marginalize-It and End-Of-Token variants) for
train/val from the BPE token stream. Outputs <out_dir>/byte_{mi,eot}_{train,val}.bin
(uint16) plus meta.pkl. Run once before byte-model training.

Usage:
  python byte_distill/build_byte_streams.py \
      --data_dir data/fineweb-small --out_dir out/byte_distill_study/byte_streams
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ByteCodec, write_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    codec = ByteCodec(os.path.join(args.data_dir, "vocab.json"))
    os.makedirs(args.out_dir, exist_ok=True)
    meta = dict(codec_vocab=codec.vocab_size, marker_id=codec.marker_id)
    for use_eot, tag in ((False, "mi"), (True, "eot")):
        for split in ("train", "val"):
            out = os.path.join(args.out_dir, f"byte_{tag}_{split}.bin")
            n, _ = codec.build_byte_stream(os.path.join(args.data_dir, f"{split}.bin"), out, use_eot)
            meta[f"{tag}_{split}"] = n
            print(f"byte_{tag}_{split}: {n:,} symbols -> {out}")
    write_meta(os.path.join(args.out_dir, "meta.pkl"), **meta)
    print("done")


if __name__ == "__main__":
    main()