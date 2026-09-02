# Let's Have a Conversation T-P-P Adapter: Method Scope

This adapter represents a local transfer of the Lets/LLMforIO T-P-P interaction design, not the full original benchmark.

The original LLMforIO setup evaluates interactive optimization through:

- an optimization assistant,
- a decision/stakeholder agent with hidden utility,
- a base optimization model,
- constrained tools for model modification,
- repeated solver calls,
- final utility-based scoring.

The local benchmark is different: it evaluates pre-modeling clarification quality against hidden OR modeling slots. This adapter keeps only the transferable T-P-P design principle: a domain-prompted interactive optimization assistant that interprets stakeholder feedback, asks clarifying questions, and maintains a model-ready summary.

It deliberately excludes:

- SFUSD school-start-time data,
- Julia/Gurobi solver calls,
- MCP tool execution,
- hidden utility maximization,
- solution-utility scoring,
- the original decision-agent `check_utility` loop.

Therefore, paper/report wording should say `Lets/LLMforIO T-P-P adapter baseline`, not `full Lets reproduction`.

Upstream implementation files consulted when constructing the adapter
(not redistributed in this artifact):

- `optimization_agent/prompts/T-P-P_agent.md`
- `optimization_agent/optimization_agent_playground.py`
- `decision_agent/decision_agent_playground.py`
- `evaluation/evaluation.py`
