# ORPilot Interview Adapter: Method Scope

This adapter represents only the ORPilot interview stage.

It preserves the original method's role as an OR consultant interview agent: ask focused structural questions about objective, decisions, constraints, parameters, and indices before modeling. The local `READY_TO_MODEL` token is only a benchmark-protocol mapping of ORPilot's original `[INTERVIEW_COMPLETE]` completion marker.

It deliberately excludes ORPilot's downstream production pipeline:

- data collection
- raw-data validation
- parameter computation
- direct code generation
- solver execution
- IR compilation
- final report generation

Therefore, paper/report wording should say `ORPilot-style interview baseline`, not `full ORPilot system baseline`.

One implementation nuance: ORPilot's interview prompt asks the assistant to discuss parameters and indices in its summary, but the current `ProblemDefinition` schema in the local ORPilot repository does not expose parameters as a stable first-class field. This adapter therefore treats parameter/index discussion as summary text, not as a guaranteed structured schema output.

Upstream implementation files consulted when constructing the adapter
(not redistributed in this artifact):

- `README.md`
- `orpilot/prompts/interview_system.md`
- `orpilot/workflow/nodes/interview.py`
- `orpilot/workflow/graph.py`
