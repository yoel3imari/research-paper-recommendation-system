# Index Benchmark

Run against the built `ivf-sq8` index (30000 vectors). Ground truth: exhaustive `IndexFlatIP` over the same embeddings. Latency: single-threaded, warmed.

| Metric | Value |
|---|---|
| Profile | `ivf-sq8` |
| Vectors | 30000 |
| Recall@10 | 0.9667 |
| p95 search latency | 0.871 ms |
| mean search latency | 0.534 ms |
| queries timed | 1000 |
| index size (est.) | 12.9 MB |
