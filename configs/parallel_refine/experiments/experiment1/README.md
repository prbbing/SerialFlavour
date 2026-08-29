# Experiment 1: fixed 1M training-data budget

This matrix tests whether a frozen-feature DNN changes the attainable performance ceiling under a fixed total training-data budget. A-train is used by the Parallel Transformer; B-train is reserved for the DNN. A-val and B-val are 100k each, and the locked Y-test split is 500k for every condition.

| Parallel config | Exact parameters | Target band | d_model | heads | layers | d_ffn |
|---|---:|---:|---:|---:|---:|---:|
| p023k | 23,197 | 20k-30k | 24 | 2 | 4 | 48 |
| p056k | 56,381 | 50k-60k | 32 | 2 | 6 | 64 |
| p086k | 86,093 | 80k-100k | 40 | 4 | 6 | 80 |
| p122k | 122,077 | 120k-200k | 48 | 4 | 6 | 96 |
| p160k | 159,997 | 120k-200k | 48 | 4 | 8 | 96 |

| Split config | A-train | B-train | Route |
|---|---:|---:|---|
| a100_b000 | 1,000,000 | 0 | Transformer-only baseline |
| a090_b010 | 900,000 | 100,000 | Transformer + DNN |
| a080_b020 | 800,000 | 200,000 | Transformer + DNN |
| a070_b030 | 700,000 | 300,000 | Transformer + DNN |

The 20 experiment JSON files are the Cartesian product of these five model sizes and four data allocations. The DNN architecture remains input -> 128 -> 64 -> 32 -> output. The a100_b000 baseline uses the transformer_only refiner component: only its Y-test cache is generated, evaluation should use --model parallel, and DNN training is intentionally disabled because B-train is empty.
