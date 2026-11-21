# Instructions for Claude in this repo

For an overview of the project from scratch, read `docs/STATUS.md` (written for readers with no prior context).
Start with `docs/HANDOFF.md`: current status, commit hashes, every decision so far and where it is recorded,
open questions, and the exact next step. Then `docs/SPEC.md` (the design; approved amendments in §17) and
`docs/HARNESS.md` (implementation notes, findings, VM re-run list).

- Effort checkpoints (SPEC §16): before any step on the max-effort list, stop and say "Effort checkpoint: please
  switch to max effort before I start [step]." and wait; afterwards say "Effort checkpoint complete: you can switch
  back to xhigh."
- Work on branch `harness-v1`. Commit and push only when asked, with test results in the commit message; never
  commit to or push `main`. Keep `docs/STATUS.md` current when the state changes.
- Spec changes need the user's approval and a dated entry in SPEC §17.
- Milestone checkpoint (SPEC §18): announce "Milestone checkpoint: the harness build is complete and Phase 0 on the
  L4 VM is next." only when all seven conditions hold, then wait for an explicit go-ahead before creating any
  cloud resources or spending money. Never announce it early or partially.
- Mac test results show code paths only; every equality and determinism test is re-run on the L4 VM before the freeze.
- Tests: `.venv/bin/python -m pytest` (unit), `.venv-sim/bin/python -m pytest tests/reference` (simulator).
  Do not run commands through `timeout` on this Mac (x86_64 binary; breaks the arm64 Python wheels).
- No hostnames, usernames or network addresses of the team's machines in the repo.
