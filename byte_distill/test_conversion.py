"""
Unit tests for the Marginalize-It / End-Of-Token logit conversion.

The main test case reproduces the worked example from the paper (Section 2,
Figure 2, "Tiramisu"): a vocabulary whose i-prefixed tokens carry the
probabilities used in the paper's prose:

    isu  0.4     isk  0.1     is   0.1     imm  0.05     imn  0.05     in  0.1
    m    0.1     n    0.1                                  (total = 1.0)

with gold token "isu". Expected results (paper Sections 2.1 / 2.2):

  Marginalize-It:
    B1: P(i)=0.8  P(m)=0.1  P(n)=0.1
    B2: P(s)=0.75 P(m)=0.125 P(n)=0.125
    B3: P(u)=0.8  P(k)=0.2          (mass of prefix token "is" dropped)
  End-Of-Token:
    B1: P(i)=0.8  P(m)=0.1  P(n)=0.1
    B2: P(s)=0.75 P(m)=0.125 P(n)=0.125
    B3: P(u)=2/3  P(k)=1/6  P(<eot>)=1/6
    B4: P(<eot>)=1.0
  Exactness (EOT): prod_k dist_k[gold symbol] == P(gold)  (0.8*0.75*2/3*1 = 0.4)

Run:  ./venv/bin/python byte_distill/test_conversion.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conversion import (  # noqa: E402
    precompute_live_sets, first_symbol_matrix, convert_chunk,
    byte_probabilities_reference,
)

EOT = 256  # EOT id in the synthetic test alphabet (bytes 0..255, then EOT)


def make_synthetic_vocab():
    """Tokens: isu, isk, is, imm, imn, in, m, n with the paper's probs."""
    tokens = ["isu", "isk", "is", "imm", "imn", "in", "m", "n"]
    probs = np.array([0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1], dtype=np.float64)
    assert abs(probs.sum() - 1.0) < 1e-12
    return tokens, probs


def build_tables(tokens, use_eot):
    V = len(tokens)
    max_len = max(len(t) for t in tokens) + (1 if use_eot else 0)
    table = np.full((V, max_len), -1, dtype=np.int64)
    lens = np.zeros(V, dtype=np.int64)
    for i, t in enumerate(tokens):
        b = [ord(c) for c in t]
        if use_eot:
            b.append(EOT)
        table[i, : len(b)] = b
        lens[i] = len(b)
    return table, lens


def run_conversion(tokens, probs, gold_token, use_eot):
    """Run the vectorized conversion for a single boundary; return list of
    distributions (numpy arrays)."""
    table, lens = build_tables(tokens, use_eot)
    v_byte = 257 if use_eot else 256
    live_sets = precompute_live_sets(table, lens)
    first_mat = first_symbol_matrix(table, v_byte)
    gold_syms = [ord(c) for c in gold_token] + ([EOT] if use_eot else [])
    g_id = tokens.index(gold_token)
    N = 1
    p = torch.tensor(probs.reshape(1, -1), dtype=torch.float32)
    gold = torch.tensor([g_id], dtype=torch.int64)
    out = convert_chunk(p, gold, live_sets, torch.tensor(first_mat), table.shape[1], v_byte)
    return [out[0, k].numpy() for k in range(len(gold_syms))], gold_syms


def approx(a, b, tol=1e-5):
    return np.allclose(a, b, atol=tol)


def test_paper_example_mi():
    tokens, probs = make_synthetic_vocab()
    dists, _ = run_conversion(tokens, probs, "isu", use_eot=False)
    i, s, u, k, m, n = (ord(c) for c in "isukmn")
    assert approx(dists[0][i], 0.8) and approx(dists[0][m], 0.1) and approx(dists[0][n], 0.1), dists[0]
    assert approx(dists[1][s], 0.75) and approx(dists[1][m], 0.125) and approx(dists[1][n], 0.125), dists[1]
    assert approx(dists[2][u], 0.8) and approx(dists[2][k], 0.2), dists[2]
    print("PASS test_paper_example_mi")


def test_paper_example_eot():
    tokens, probs = make_synthetic_vocab()
    dists, _ = run_conversion(tokens, probs, "isu", use_eot=True)
    i, s, u, k, m, n = (ord(c) for c in "isukmn")
    assert approx(dists[0][i], 0.8) and approx(dists[0][m], 0.1) and approx(dists[0][n], 0.1), dists[0]
    assert approx(dists[1][s], 0.75) and approx(dists[1][m], 0.125) and approx(dists[1][n], 0.125), dists[1]
    assert approx(dists[2][u], 2 / 3) and approx(dists[2][k], 1 / 6) and approx(dists[2][EOT], 1 / 6), dists[2]
    assert approx(dists[3][EOT], 1.0), dists[3]
    print("PASS test_paper_example_eot")


def test_eot_exactness():
    """The EOT conversion preserves likelihood: the product of the gold
    symbol probabilities equals the gold token's probability."""
    tokens, probs = make_synthetic_vocab()
    for g in tokens:
        dists, gold_syms = run_conversion(tokens, probs, g, use_eot=True)
        prod = 1.0
        for d, sym in zip(dists, gold_syms):
            prod *= d[sym]
        expected = probs[tokens.index(g)]
        assert abs(prod - expected) < 1e-5, (g, prod, expected)
    print("PASS test_eot_exactness")


def test_mi_not_exact_in_general():
    """Marginalize-It renormalizes: for gold 'isu' the product is 0.48 != 0.4
    (the dropped prefix mass is redistributed)."""
    tokens, probs = make_synthetic_vocab()
    dists, gold_syms = run_conversion(tokens, probs, "isu", use_eot=False)
    prod = 1.0
    for d, sym in zip(dists, gold_syms):
        prod *= d[sym]
    assert abs(prod - 0.48) < 1e-5, prod
    print("PASS test_mi_not_exact_in_general")


def test_first_byte_exact():
    """k=0 must equal the direct marginalization over the full vocabulary."""
    tokens, probs = make_synthetic_vocab()
    for use_eot in (False, True):
        dists, _ = run_conversion(tokens, probs, "imm", use_eot=use_eot)
        direct = {}
        for t, p in zip(tokens, probs):
            direct[ord(t[0])] = direct.get(ord(t[0]), 0.0) + p
        for b, v in direct.items():
            assert approx(dists[0][b], v), (use_eot, b, dists[0][b], v)
    print("PASS test_first_byte_exact")


def test_vectorized_matches_reference():
    """Randomized cross-check against the paper's Appendix F.1 pseudocode."""
    rng = np.random.default_rng(0)
    for trial in range(8):
        for use_eot in (False, True):
            V = int(rng.integers(20, 80))
            max_len = int(rng.integers(1, 6))
            # random byte sequences (allow shared prefixes and short tokens)
            seqs = []
            for _ in range(V):
                L = int(rng.integers(1, max_len + 1))
                seqs.append([int(rng.integers(0, 256)) for _ in range(L)])
            lens = np.array([len(s) for s in seqs])
            if use_eot:
                seqs = [s + [EOT] for s in seqs]
                lens = lens + 1
            max_len = max(len(s) for s in seqs)
            table = np.full((V, max_len), -1, dtype=np.int64)
            for i, s in enumerate(seqs):
                table[i, : len(s)] = s
            # random teacher distribution
            z = rng.random(V)
            probs = z / z.sum()
            # random gold token
            g = int(rng.integers(0, V))
            gold_syms = seqs[g]

            # reference (paper pseudocode)
            ref = byte_probabilities_reference(probs, table, lens, np.array(gold_syms), use_eot)

            # vectorized
            v_byte = 257 if use_eot else 256
            live_sets = precompute_live_sets(table, lens)
            first_mat = first_symbol_matrix(table, v_byte)
            out = convert_chunk(
                torch.tensor(probs.reshape(1, -1), dtype=torch.float32),
                torch.tensor([g], dtype=torch.int64),
                live_sets, torch.tensor(first_mat), max_len, v_byte,
            )
            for k in range(len(gold_syms)):
                got = out[0, k].numpy()
                # compare on the 256-byte + EOT support (reference uses 256/257)
                assert np.allclose(got[: len(ref[k])], ref[k], atol=1e-5), (trial, use_eot, k, got, ref[k])
    print("PASS test_vectorized_matches_reference")


def test_batched_chunk():
    """A chunk with several boundaries (different golds) must match per-position
    single-boundary conversion."""
    tokens, probs = make_synthetic_vocab()
    for use_eot in (False, True):
        table, lens = build_tables(tokens, use_eot)
        v_byte = 257 if use_eot else 256
        live_sets = precompute_live_sets(table, lens)
        first_mat = first_symbol_matrix(table, v_byte)
        gold_ids = [0, 1, 2, 5, 6, 0, 3, 7]  # isu, isk, is, in, m, isu, imm, n
        N = len(gold_ids)
        p = np.stack([probs] * N)
        out = convert_chunk(
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(gold_ids, dtype=torch.int64),
            live_sets, torch.tensor(first_mat), table.shape[1], v_byte,
        )
        for n, g in enumerate(gold_ids):
            single, _ = run_conversion(tokens, probs, tokens[g], use_eot=use_eot)
            L = len(single)
            for k in range(L):
                assert np.allclose(out[n, k].numpy(), single[k], atol=1e-6), (use_eot, n, g, k)
    print("PASS test_batched_chunk")


def test_real_codec_shapes():
    """Smoke test on the real fineweb-small codec: distributions must be
    valid probability vectors at every defined position."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    codec_path = os.path.join(root, "data", "fineweb-small", "vocab.json")
    if not os.path.exists(codec_path):
        print("SKIP test_real_codec_shapes (no fineweb-small vocab)")
        return
    from tokenizer import ByteCodec, VOCAB_BYTE, VOCAB_BYTE_EOT, EOT as EOT_ID
    codec = ByteCodec(codec_path)
    rng = np.random.default_rng(1)
    for use_eot, v_byte in ((False, VOCAB_BYTE), (True, VOCAB_BYTE_EOT)):
        table, lens = codec.symbol_table(use_eot)
        live_sets = precompute_live_sets(table, lens)
        first_mat = first_symbol_matrix(table, v_byte)
        N = 512
        gold = rng.integers(0, codec.vocab_size, size=N)
        z = rng.random((N, codec.vocab_size))
        p = torch.tensor(z / z.sum(axis=1, keepdims=True), dtype=torch.float32)
        out = convert_chunk(p, torch.tensor(gold, dtype=torch.int64),
                            live_sets, torch.tensor(first_mat), table.shape[1], v_byte)
        for n in range(0, N, 37):
            L = int(lens[gold[n]])
            for k in range(L):
                row = out[n, k].numpy()
                assert abs(row.sum() - 1.0) < 1e-4, (use_eot, n, k, row.sum())
                assert (row >= 0).all()
    print("PASS test_real_codec_shapes")


if __name__ == "__main__":
    test_paper_example_mi()
    test_paper_example_eot()
    test_eot_exactness()
    test_mi_not_exact_in_general()
    test_first_byte_exact()
    test_vectorized_matches_reference()
    test_batched_chunk()
    test_real_codec_shapes()
    print("\nAll conversion tests passed.")