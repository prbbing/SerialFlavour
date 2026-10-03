# MASSIVE XLM-R configurations

- `cluster_full.json`: offline public pretrained XLM-R Base; full en-US official test; ST/MT, 5 upstream seeds × 5 downstream seeds. Use `scripts/run_cluster.sh`. Prepared only, not executed.
- `smoke.json`: random reduced encoder, 560 partition records. Current experiment is `smoke_cpu_v2`, not executed after the cluster-interface update. Historical results belong to `smoke_cpu_v1` and its original code fingerprint.

Paths are relative to repository root unless absolute. Copy the config and change `experiment` when changing any run settings or source. Cluster usage and asset preparation: `cross-domain/docs/nlp_massive_xlm/cluster_agent_handoff_zh.md`.
