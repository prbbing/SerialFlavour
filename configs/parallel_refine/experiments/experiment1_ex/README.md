# Experiment 1 EX: fixed-total-data Transformer/DNN scaling

This matrix compares the 122,077-parameter Parallel Transformer with the
default DNN-refiner configuration while the total selected dataset grows.
Each condition uses an exact 70/10/20 dataset split: 70% is the training pool,
10% is one validation split shared by Parallel and every DNN refiner, and 20%
is the locked shared Y-test split. The DNN therefore never receives training
examples used by the Transformer, but both stages may early-stop on the same
validation examples.

| Total dataset | Training pool (70%) | Shared validation (10%) | Shared Y-test (20%) |
|---|---:|---:|---:|
| 1M | 700,000 | 100,000 | 200,000 |
| 2M | 1,400,000 | 200,000 | 400,000 |
| 3M | 2,100,000 | 300,000 | 600,000 |
| 4M | 2,800,000 | 400,000 | 800,000 |
| 5M | 3,500,000 | 500,000 | 1,000,000 |
| 6M | 4,200,000 | 600,000 | 1,200,000 |

For every row, the training pool is partitioned without overlap between
Parallel A-train and DNN B-train.

| A/B training allocation | Parallel A-train | DNN B-train | Route |
|---|---:|---:|---|
| 100/0 | 100% | 0% | Transformer-only baseline |
| 95/5 | 95% | 5% | Transformer + default DNN refiners |
| 90/10 | 90% | 10% | Transformer + default DNN refiners |
| 85/15 | 85% | 15% | Transformer + default DNN refiners |
| 80/20 | 80% | 20% | Transformer + default DNN refiners |
| 75/25 | 75% | 25% | Transformer + default DNN refiners |
| 70/30 | 70% | 30% | Transformer + default DNN refiners |

The 1M--3M totals use every listed allocation; 4M--6M currently add the
80/20, 75/25, and 70/30 points, for 30 experiment config files in total.
`shared_validation: true` is an explicit split protocol: it makes
`a_val` and `b_val` identical at both jet and event level, while every other
cross-split pair remains event-disjoint.
