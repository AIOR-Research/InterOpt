# Shared Experiment Tools

This directory contains reusable utilities shared by the external-method
adapters. It is infrastructure rather than a separate method reported in the
paper.

- `baseline_runner_common.py`: common case loading, hosted-model access, user
  simulation, transcript writing, judging, and metric calculation.
- `run_external_baseline_batches.py`: runs ORPilot and T-P-P in bounded
  multi-case batches.
- `run_external_baseline_case_fanout.py`: fills missing case/run combinations
  with bounded parallelism.
