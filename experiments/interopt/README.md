# InterOPT

InterOPT is the paper's full two-stage clarification method. Stage 1 maintains
a persistent ledger of unresolved formulation gaps. Stage 2 generates candidate
clarification states and questions tied to open gaps, after which a selector
chooses the public question.

## Method

1. **Gap discovery:** Stage 1 reads the public conversation and current ledger,
   then adds at most three new formulation-critical gaps.
2. **Question construction:** Stage 2 produces three candidate clarification
   questions. Each candidate must bind to an open gap.
3. **Selection:** An independent selector chooses one candidate for the public
   MC-D interaction.
4. **Ledger update:** The selected gap changes from `OPEN` to `ASKED`.
5. **Stopping:** The method does not impose a minimum question count or
   override `READY_TO_MODEL`.

The ledger is private method state. It is written to `ledger_events.json` but
is never shown to the simulated user, protocol detector, judge, or public
transcript.

## Run

From the artifact root:

```bash
python experiments/interopt/run_parallel.py \
  --toml_dirs data --interaction_modes mc_d \
  --k 5 --max_turns 20 --max_concurrency 100 \
  --gap_search_prompt_path experiments/interopt/prompts/prompt-ledger-gap-search.md \
  --agent_prompt_supplements experiments/interopt/prompts/agent_ledger_supplement.md \
  --selector_prompt_supplements experiments/interopt/prompts/selector_ledger_supplement.md \
  --output_dir runs/interopt
```

Add `--case_ids 001 --k 1 --max_concurrency 1` for a one-case smoke test.

## Outputs

Each run contains `statistics.json`, `judge_result.json`, `ledger_events.json`,
the public transcript, and simulator/detector audit events. Ledger counters in
`statistics.json` report additions, consumed gaps, duplicate suppression,
unbound turns, and stop attempts made while open gaps remain.

## Offline verification

```bash
python -m py_compile experiments/interopt/run_pipeline.py \
  experiments/interopt/run_parallel.py experiments/interopt/frontier_control.py
python -m unittest discover -s experiments/interopt/tests -v
```
