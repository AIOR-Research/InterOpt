# InterOPT w/o Stage 1

This system-level ablation removes InterOPT's Stage 1 gap discovery and ledger
population. Stage 2 candidate construction, the independent selector, MC-D,
the simulated user, and the judge remain active.

Because Stage 1 normally creates the persistent ledger, the ablation begins
with an empty ledger and keeps it empty. It therefore measures the complete
system after removing upstream gap discovery; it is not an equal-compute
estimate of a single additional model call.

## Invariants

- Gap-search calls, usage, and cost must be zero.
- Stage 2 still produces candidate clarification questions.
- The selector still chooses the public question.
- The simulated user's private structured history remains isolated from the
  tested agent and judge.
- The interaction protocol remains MC-D with a 20-turn limit.

## Run

From the artifact root:

```bash
python experiments/interopt_without_stage1/run_parallel.py \
  --toml_dirs data --interaction_modes mc_d \
  --k 5 --max_turns 20 --max_concurrency 100 \
  --agent_prompt_supplements experiments/interopt_without_stage1/prompts/agent_ledger_supplement.md \
  --selector_prompt_supplements experiments/interopt_without_stage1/prompts/selector_ledger_supplement.md \
  --output_dir runs/interopt_without_stage1
```

Add `--case_ids 001 --k 1 --max_concurrency 1` for a one-case smoke test.

## Offline verification

```bash
python -m py_compile experiments/interopt_without_stage1/run_pipeline.py \
  experiments/interopt_without_stage1/run_parallel.py \
  experiments/interopt_without_stage1/frontier_control.py
python -m unittest discover -s experiments/interopt_without_stage1/tests -v
```
