# Choice MC and MC-D

This directory implements the two Choice protocols reported in the paper.

- **MC** presents three generated options (A–C).
- **MC-D** adds a fixed option D for a free-form correction when A–C do not
  match the simulated user's private facts.

Both protocols use the same cases, tested model, simulator, judge, and passive
protocol monitor.

## Files

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Sequential experiment runner |
| `run_parallel.py` | Bounded parallel scheduler using the same experiment logic |
| `prompts/` | Frozen agent, simulator, detector, and judge prompts |
| `tests/` | Offline protocol and scheduler contract tests |

## Run

From the artifact root:

```bash
python experiments/choice_mc_mcd/run_pipeline.py \
  --toml_dirs data \
  --interaction_modes mc mc_d \
  --case_ids 001 \
  --k 1 \
  --max_turns 20 \
  --output_dir runs/smoke-choice
```

For bounded parallel execution:

```bash
python experiments/choice_mc_mcd/run_parallel.py \
  --toml_dirs data \
  --interaction_modes mc mc_d \
  --case_ids 001 \
  --k 1 \
  --max_turns 20 \
  --output_dir runs/smoke-choice-parallel
```

## Offline tests

```bash
python -m unittest discover \
  -s experiments/choice_mc_mcd/tests -q
```

The tested agent receives only the public brief and public transcript.
Simulator-only rationale and match-status fields are retained for auditing but
are never exposed to the tested agent or post-hoc judge.
