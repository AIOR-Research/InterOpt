# Gap Ledger Question Binding

When internal gap-ledger guidance is supplied:

- Generate Q1/Q2/Q3 only for gap IDs listed under `open_gaps`.
- Add `frontier_gap_id` to every Q1/Q2/Q3 object.
- Do not generate a question for gaps already marked ASKED.
- Different candidate phrasings may share one gap ID when only one open gap
  remains.
- If the guidance says no OPEN gap remains, generate Q1/Q2/Q3 freely for any
  remaining clarification need and set `frontier_gap_id` to `"NONE"`.

Example candidate shape:

{
  "frontier_gap_id": "G001",
  "question": "...",
  "options": [...],
  "allow_other": true
}
