"""COUNT: n-gram count tables on host RAM (Magnusson, Smith & Koh, "The COUNT").

* count.table      -- ctypes front end for the compiled host-resident lookup
* count.reference  -- slow numpy twin used by count.selfcheck to prove the C++ path is right
* count.prefetch   -- background batch lookups so table search overlaps GPU compute
* count.data       -- .bin windows -> GPU batches with count lookups attached (train.py, eval)
* count.generate   -- the count=... batch for one decode window (GPT.generate)
* count.selfcheck  -- correctness + throughput harness (python -m count.selfcheck)

torch is imported inside the functions that need it, so opening a table never pulls torch in.
"""

from count.data import CountDataPipeline, open_tables, table_weights
from count.generate import count_batch
from count.prefetch import CountPrefetcher
from count.table import CountResult, CountTable, load_library, mix, mix_next, set_threads

__all__ = [
    "CountTable",
    "CountResult",
    "CountPrefetcher",
    "CountDataPipeline",
    "count_batch",
    "open_tables",
    "table_weights",
    "mix",
    "mix_next",
    "set_threads",
    "load_library",
]
