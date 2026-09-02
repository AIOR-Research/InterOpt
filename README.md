# Ask Before You Optimize

Official code and benchmark repository for **Ask Before You Optimize: Dynamic Pre-Formulation Clarification for Interactive Optimization**.

This repository provides OR-Clarify, a benchmark for pre-formulation clarification in operations research, together with implementations of InterOPT and comparison methods under Choice and open/free-form interaction protocols. InterOPT combines **Dynamic Gap Search** and **Gap-Guided Action Search** to identify formulation-critical information gaps and guide questioning and stopping.

## Authors and Contributors

**Core contributors (equal contribution):** [Sihan Ge](https://github.com/Sleepyorangezz), Yichen Lin, and Chenyu Zhou.

**Paper authors:** Sihan Ge, Yichen Lin, Chenyu Zhou, Jianghao Lin, Tao Yao, and Dongdong Ge.

**Corresponding authors:** Jianghao Lin and Tao Yao.

This repository is hosted by [AIOR-Research](https://github.com/AIOR-Research).

## Scope

Included:

- Choice MC, MC-D and ReadyGate;
- Choice InterOPT and its two system-level stage ablations;
- open/free-form InterOPT and its two system-level stage ablations;
- the ORPilot interview-stage adapter;
- the *Let's Have a Conversation* T-P-P adapter;
- shared simulator, judge, protocol-monitor, and batch utilities.

The ORPilot and T-P-P directories are interface adapters, not claims of reproducing either external system in its entirety. Their adaptation boundaries are documented in the corresponding `METHOD_SCOPE.md` files.

## Directory map

| Path | Contents |
|---|---|
| `data/` | 100 canonical OR-Clarify TOML cases and the data card |
| `experiments/` | Runnable pipelines, prompts, method notes, and contract tests |
| `.env.example` | Environment-variable template without credentials |
| `requirements.txt` | Minimal Python dependency declaration |

## Benchmark cases

The `data/` directory contains the OR-Clarify benchmark cases. Each method
provides a runner with configurable input and output paths. See the
corresponding method documentation or run the script with `--help` for
available command-line options.

## Paper-to-artifact map

| Setting | Paper method | Implementation |
|---|---|---|
| Choice | MC / MC-D | `experiments/choice_mc_mcd/` |
| Choice | ReadyGate | `experiments/readygate/` |
| Choice | InterOPT | `experiments/interopt/` |
| Choice | InterOPT w/o Stage 1 | `experiments/interopt_without_stage1/` |
| Choice | InterOPT w/o Stage 2 | `experiments/interopt_without_stage2/` |
| Open / FreeQA | InterOPT | `experiments/open_interopt/` |
| Open / FreeQA | InterOPT w/o Stage 1 | `experiments/open_interopt_without_stage1/` |
| Open / FreeQA | InterOPT w/o Stage 2 | `experiments/open_interopt_without_stage2/` |
| Open adapter | ORPilot | `experiments/orpilot/` |
| Open adapter | Lets T-P-P | `experiments/lets_tpp/` |

Shared simulator, judge, and protocol-monitor components are in `experiments/evaluation_protocol/`. Shared external-method batch utilities are in `experiments/shared_tools/`.

## Environment

Python 3.11 or later is recommended. Python 3.10 requires the optional `tomli` dependency.

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and insert credentials locally. The Choice-side runners load the root `.env` file. The open/free-form runners read the same variables directly from the process environment, so export or otherwise load them before execution.
