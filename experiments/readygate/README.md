# ReadyGate

ReadyGate augments MC-D with an independent stopping reviewer. When the tested
agent emits `READY_TO_MODEL`, the reviewer checks whether the public
conversation still supports a material unresolved modeling interpretation.

## Method boundary

- The interaction protocol is MC-D with a 20-turn limit.
- The reviewer sees only the initial brief, public transcript, and current
  readiness summary.
- It never sees hidden slots, reference answers, or the judge rubric.
- `PASS` accepts the stop request. `BLOCK` returns private feedback to the
  tested agent and continues the interaction.
- Reviewer failures default to `PASS` and are recorded for audit, preventing an
  infrastructure error from being counted as a method gain.

## Run

From the artifact root:

```bash
python experiments/readygate/run_parallel.py \
  --toml_dirs data --interaction_modes mc_d \
  --k 5 --max_turns 20 --max_concurrency 100 \
  --output_dir runs/readygate
```

Add `--case_ids 001 --k 1 --max_concurrency 1` for a one-case smoke test.

## Outputs

Each run contains the public transcript, `counterfactual_ready_gate_events.json`,
detector and simulator audit events, `judge_result.json`, `statistics.json`, and
`run_health.json`. The ReadyGate event file records every `PASS`, `BLOCK`, and
reviewer error without exposing the reviewer feedback to the simulator or
judge.

## Offline verification

```bash
python -m unittest discover -s experiments/readygate/tests -v
```
