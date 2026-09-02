# Gap Ledger Selection Boundary

The runner also supplies the current open gap ledger.

- Select only a candidate whose `frontier_gap_id` references an OPEN ledger
  gap; when no OPEN gap remains, any candidate with `frontier_gap_id: "NONE"`
  is acceptable.
- Prefer the candidate whose answer would most change the objective, decision
  scope, constraints, time boundary, entity relationship, or hard-versus-soft
  policy of the resulting model.
- Do not reward mathematical sophistication, extra detail, or repeated
  confirmation of an already asked gap.
