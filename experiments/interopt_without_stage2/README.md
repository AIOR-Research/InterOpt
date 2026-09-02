# InterOPT w/o Stage 2

This system-level ablation retains InterOPT's Stage 1 gap discovery and
persistent ledger while removing Stage 2 candidate construction and the
selector. Stage 1 directly emits one public MC-D question tied to a ledger gap.

The direct-question interface is the minimum adaptation needed after removing
the downstream question-writing stage. The result therefore compares two
complete systems; it is not an estimate of the isolated marginal value of one
Stage 2 model call.

## Invariants

- Stage 1 adds at most three new gaps per turn.
- Stage 2 and selector calls, usage, and cost must be zero.
- The selected gap changes from `OPEN` to `ASKED` after its public question.
- Ledger identifiers and evidence remain outside the public transcript,
  simulated-user context, detector, and judge.
- The interaction protocol remains MC-D with a 20-turn limit.

## Run

From the artifact root:

```bash
python experiments/interopt_without_stage2/run_parallel.py \
  --toml_dirs data --interaction_modes mc_d \
  --k 5 --max_turns 20 --max_concurrency 100 \
  --output_dir runs/interopt_without_stage2
```

Add `--case_ids 001 --k 1 --max_concurrency 1` for a one-case smoke test.

## Outputs

Each run includes `statistics.json`, `judge_result.json`,
`ledger_direct_question_events.json`, the public transcript, and protocol audit
events.

## Offline verification

```bash
python -m py_compile experiments/interopt_without_stage2/run_pipeline.py \
  experiments/interopt_without_stage2/run_parallel.py \
  experiments/interopt_without_stage2/frontier_control.py
python -m unittest discover -s experiments/interopt_without_stage2/tests -v
```
