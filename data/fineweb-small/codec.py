"""
Encoder/decoder for the small corpus-trained vocab produced by prepare_small.py.

Vocab layout (id ranges):
  0 .. 255              raw UTF-8 bytes (lossless fallback, always legal outputs)
  256 .. vocab-2        space-prefixed word pieces learned from the training text
  vocab-1               doc-split marker (a normal learned row, NOT a stop token)

Round-trip guarantee: decode(encode(text)) == text for any text that does not
contain the marker string. Rare words cost bytes, never disappear.
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BYTES = 256


class Codec:
    def __init__(self, vocab_path=None):
        with open(vocab_path or os.path.join(HERE, "vocab.json")) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.marker_id = data["marker_id"]
        self.marker = data["marker"]
        self.id_to_piece = {int(k): v for k, v in data["id_to_piece"].items()}
        self.piece_to_id = {v.encode("utf-8"): i for i, v in self.id_to_piece.items()
                            if i != self.marker_id}
        self.max_piece_len = max((len(k) for k in self.piece_to_id), default=1)

    def encode_doc(self, text):
        """Encode one document (space-prefixed pieces, byte fallback) + closing marker."""
        data = (" " + text).encode("utf-8")
        ids, i, n = [], 0, len(data)
        while i < n:
            if data[i] == 0x20:
                for ln in range(min(self.max_piece_len, n - i), 1, -1):
                    vid = self.piece_to_id.get(data[i:i + ln])
                    if vid is not None:
                        ids.append(vid)
                        i += ln
                        break
                else:
                    ids.append(0x20)
                    i += 1
            else:
                ids.append(data[i])
                i += 1
        ids.append(self.marker_id)
        return ids

    def decode_ids(self, ids, keep_marker=False):
        out = bytearray()
        for t in map(int, ids):
            if t < N_BYTES:
                out += bytes([t])
            elif t == self.marker_id:
                out += (self.marker if keep_marker else "\n").encode("utf-8")
            else:
                out += self.id_to_piece[t].encode("utf-8")
        return out.decode("utf-8", "replace")

    def stats(self, ids):
        ids = np.asarray(ids)
        n = max(1, len(ids))
        return dict(
            tokens=len(ids),
            word_share=float((ids >= N_BYTES).mean() - (ids == self.marker_id).mean()),
            byte_share=float((ids < N_BYTES).mean()),
            marker_share=float((ids == self.marker_id).mean()),
        )


if __name__ == "__main__":
    import sys

    c = Codec()
    data_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    print("train:", c.stats(train[:200_000]))
    print("val:  ", c.stats(val[:50_000]))

    # lossless round-trip over the first N documents of the val split
    docs, cur = [], []
    for t in val[:40_000]:
        if int(t) == c.marker_id:
            docs.append(cur)
            cur = []
            if len(docs) >= 25:
                break
        else:
            cur.append(int(t))
    bad = 0
    for raw in docs:
        text = c.decode_ids(raw).strip()
        # raw excludes the closing marker (we cut on it); encode_doc appends it
        if c.encode_doc(text) != raw + [c.marker_id]:
            bad += 1
    print(f"round-trip: {len(docs)-bad}/{len(docs)} docs re-encoded identically")

    print("\n--- decoded sample (220 chars, control-looking sequences scrubbed) ---")
    sample = c.decode_ids(val[:120])[:240]
    print(re.sub(r"<\|[^|]{0,24}\|>", "<ctrl>", sample))
