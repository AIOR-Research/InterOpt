# ORPilot Interview Adapter

This directory adapts ORPilot's interview stage to the OR-Clarify evaluation
protocol. It uses the same simulated user, judge, 20-turn limit, and
hosted-model configuration as the other packaged methods.

The adapter maps ORPilot's interview-completion signal to
`READY_TO_MODEL`. It does not reproduce ORPilot's downstream data processing,
model generation, solver execution, or full production system. See
`METHOD_SCOPE.md` for the exact adaptation boundary.

From the artifact root:

```bash
python experiments/orpilot/run_pipeline.py \
  --toml_dirs data \
  --case_ids 001 --k 1 --max_turns 20 \
  --output_dir runs/orpilot_smoke
```
