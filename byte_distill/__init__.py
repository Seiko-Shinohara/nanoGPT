"""
Implementation of "Breaking the Token Ceiling: Distilling Smaller, Stronger
Byte Models" (arXiv:2609.12303) on top of nanoGPT.

Modules:
  llama.py           Llama-3-style transformer (RMSNorm, RoPE, SwiGLU, qk-norm)
  tokenizer.py       byte-stream construction (Bytes / Bytes w/ <eot>)
  conversion.py      Marginalize-It and End-Of-Token logit conversion
  dump_teacher.py    offline teacher logit dumping (byte dists + top-k tokens)
  train_byte.py      training script for all six study configurations
  analyze.py         BPB-vs-FLOPs scaling-law fits and plots
  test_conversion.py unit tests for the logit conversion
"""