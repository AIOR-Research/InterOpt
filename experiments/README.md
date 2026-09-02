# Experiment code

The directories below follow the method names used in the paper. Each method
keeps its runner and method-specific prompts together.

| Directory | Setting | Role |
|---|---|---|
| `choice_mc_mcd/` | Choice | MC and MC-D |
| `readygate/` | Choice | ReadyGate |
| `interopt/` | Choice | Full InterOPT |
| `interopt_without_stage1/` | Choice | InterOPT without Stage 1 |
| `interopt_without_stage2/` | Choice | InterOPT without Stage 2 |
| `open_interopt/` | Open / FreeQA | Full InterOPT |
| `open_interopt_without_stage1/` | Open / FreeQA | InterOPT without Stage 1 |
| `open_interopt_without_stage2/` | Open / FreeQA | InterOPT without Stage 2 |
| `orpilot/` | Open adapter | ORPilot interview-stage adapter |
| `lets_tpp/` | Open adapter | Lets T-P-P adapter |
| `evaluation_protocol/` | Shared | Simulator, judge, and passive monitor |
| `shared_tools/` | Shared | External-method batch utilities |

## Common method layout

Depending on the method, a directory may contain:

```text
method/
├── README.md or METHOD_NOTES.md
├── run_pipeline.py
├── run_parallel.py or run_pipeline_parallel.py
├── prompts/
└── tests/ or test_contracts.py
```

ORPilot and T-P-P are adapters rather than full-system reproductions. Their
`METHOD_SCOPE.md` files state exactly which parts of the upstream methods are
represented.

Install dependencies once from the artifact root using `requirements.txt`.
Per-method dependency files are intentionally omitted because all packaged
runners use the same standard-library-first environment.
