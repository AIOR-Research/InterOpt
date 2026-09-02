# Lets T-P-P Adapter

This directory adapts the T-P-P interaction design from *Let's Have a
Conversation* to the OR-Clarify evaluation protocol. It uses the same simulated
user, judge, 20-turn limit, and hosted-model configuration as
the other packaged methods.

The adapter covers the clarification interaction only. It does not reproduce
the upstream SFUSD application, solver toolkit, or utility-maximization
environment. See `METHOD_SCOPE.md` for the exact adaptation boundary.

From the artifact root:

```bash
python experiments/lets_tpp/run_pipeline.py \
  --toml_dirs data \
  --case_ids 001 --k 1 --max_turns 20 \
  --output_dir runs/lets_tpp_smoke
```
