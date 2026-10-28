"""
Byte-stream construction for the byte-distillation study (paper Sections 2, A.2).

The paper's byte tokenizer uses a vocabulary of 256 UTF-8 bytes plus special
tokens (260 for Bytes / Marginalize-It, 261 for Bytes w/ <eot> / End-Of-Token).
We keep the same layout:

    0..255   raw UTF-8 bytes
    256      PAD
    257      EOS
    258      BOS
    259      DOC   (document separator; replaces the BPE docsplit marker)
    260      EOT   (end-of-token; only present in the End-Of-Token variant)

The source token stream is a BPE stream with a byte fallback (here: the
fineweb-small codec, vocab 4096 = 256 bytes + 3839 word pieces + 1 marker).
Every BPE token decodes to a byte sequence, which is what the logit
conversion in conversion.py operates on.

Stream layout (paper Section 2.2):
  Marginalize-It : [..., b1, b2, ..., DOC, ...]
  End-Of-Token   : [..., b1, b2, <eot>, ..., DOC, <eot>, ...]
i.e. <eot> is appended after every BPE token, including the document marker.
"""

import json
import os
import pickle

import numpy as np

# special token ids (paper: 256 bytes + 4 special = 260; + EOT = 261)
PAD, EOS, BOS, DOC, EOT = 256, 257, 258, 259, 260
VOCAB_BYTE = 260        # Bytes / Marginalize-It
VOCAB_BYTE_EOT = 261    # Bytes w/ <eot> / End-Of-Token
N_BYTES = 256


class ByteCodec:
    """Wraps a BPE-with-byte-fallback codec (vocab.json) and exposes the
    per-token byte (symbol) sequences needed for byte-stream construction and
    logit conversion."""

    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.marker_id = data["marker_id"]
        self.id_to_piece = {int(k): v for k, v in data["id_to_piece"].items()}

        # per-token raw byte sequences (no EOT), -1 padded
        V = self.vocab_size
        pieces = []
        for t in range(V):
            if t < N_BYTES:
                pieces.append([t])
            elif t == self.marker_id:
                pieces.append([DOC])  # marker becomes the DOC special token
            else:
                pieces.append(list(self.id_to_piece[t].encode("utf-8")))
        max_len = max(len(p) for p in pieces)
        table = np.full((V, max_len), -1, dtype=np.int64)
        lens = np.zeros(V, dtype=np.int64)
        for t, p in enumerate(pieces):
            table[t, : len(p)] = p
            lens[t] = len(p)
        self.sym_table = table          # (V, max_len) int64, -1 padded
        self.lens = lens                # (V,) int64, symbol length (== byte length)
        self.max_len = max_len
        # byte length used for BPB accounting on token models:
        # the marker decodes to "\n" (1 byte)
        self.token_byte_lengths = lens.copy()
        self.token_byte_lengths[self.marker_id] = 1

    # ------------------------------------------------------------------
    # symbol tables for the two conversion variants
    # ------------------------------------------------------------------
    def symbol_table(self, use_eot: bool):
        """Symbol sequence per token.
        use_eot=False: bytes only                      (lengths = self.lens)
        use_eot=True : bytes + [EOT] after every token (lengths = self.lens + 1)
        """
        if not use_eot:
            return self.sym_table, self.lens
        V, L = self.sym_table.shape
        table = np.full((V, L + 1), -1, dtype=np.int64)
        table[:, :L] = self.sym_table
        table[np.arange(V), self.lens] = EOT
        return table, self.lens + 1

    # ------------------------------------------------------------------
    # byte stream construction
    # ------------------------------------------------------------------
    def build_byte_stream(self, token_bin, out_bin_path, use_eot: bool,
                          chunk_tokens=1_000_000):
        """Map a uint16 token stream (path or array) to the byte stream and
        write it out. Returns (n_symbols, starts) where starts[t] is the
        byte-stream offset of the first symbol of token t."""
        if isinstance(token_bin, str):
            tokens = np.memmap(token_bin, dtype=np.uint16, mode="r")
        else:
            tokens = token_bin
        table, lens = self.symbol_table(use_eot)
        N = len(tokens)
        # total length (two passes: length, then fill)
        total = 0
        for i in range(0, N, chunk_tokens):
            total += int(lens[tokens[i:i + chunk_tokens]].sum())
        out = np.memmap(out_bin_path, dtype=np.uint16, mode="w+", shape=(total,))
        starts = np.empty(N, dtype=np.int64)
        off = 0
        for i in range(0, N, chunk_tokens):
            tk = tokens[i:i + chunk_tokens]
            lt = lens[tk]
            m = len(tk)
            st = np.empty(m, dtype=np.int64)
            st[0] = off
            if m > 1:
                st[1:] = off + np.cumsum(lt[:-1])
            starts[i:i + m] = st
            base = st
            for j in range(table.shape[1]):
                mask = lt > j
                if mask.any():
                    out[base[mask] + j] = table[tk[mask], j]
            off += int(lt.sum())
        out.flush()
        del out
        return total, starts

    def stream_length(self, token_bin_path, use_eot: bool, chunk_tokens=1_000_000):
        tokens = np.memmap(token_bin_path, dtype=np.uint16, mode="r")
        _, lens = self.symbol_table(use_eot)
        N = len(tokens)
        total = 0
        for i in range(0, N, chunk_tokens):
            total += int(lens[tokens[i:i + chunk_tokens]].sum())
        return total


def write_meta(path, **kwargs):
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)


def read_meta(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data", "fineweb-small")
    codec = ByteCodec(os.path.join(data_dir, "vocab.json"))
    for use_eot in (False, True):
        n = codec.stream_length(os.path.join(data_dir, "train.bin"), use_eot)
        print(f"use_eot={use_eot}: train byte stream length = {n:,} symbols")
        n = codec.stream_length(os.path.join(data_dir, "val.bin"), use_eot)
        print(f"use_eot={use_eot}: val   byte stream length = {n:,} symbols")