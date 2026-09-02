# Shared Evaluation Protocol

This directory contains the common evaluation components used across the
packaged experiments: case loading, the simulated user, question counting,
protocol monitoring, hidden-slot judging, and run-level statistics.

## Reported configuration

- The tested agent uses temperature `0.2`.
- The simulated user, detector, selector, readiness reviewer, and judge use
  temperature `0.0`.
- Structural retries are disabled in the reported runs.
- The answer-scope monitor is passive when enabled; it never edits a simulated
  user's answer.

Credentials and model identifiers are read from environment variables shown in
the artifact-root `.env.example`.

## Minimal run

From the artifact root:

```bash
python experiments/evaluation_protocol/run_pipeline.py \
  --toml_dirs data --limit 1 --k 1 --max_turns 2 \
  --output_dir runs/evaluation_protocol_smoke
```

This command calls the configured hosted models. Offline contract tests do not.

## Offline verification

```bash
python -m unittest discover -s experiments/evaluation_protocol/tests \
  -p "test_*.py" -v
python -m py_compile experiments/evaluation_protocol/run_pipeline.py
```
