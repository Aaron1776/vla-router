# Instructions for Claude in this repo

Start with `docs/HANDOFF.md`: current status, commit hashes, every decision so far and where it is recorded,
open questions, the effort schedule, and the exact next step. Read `docs/SPEC.md` (the design; approved amendments
in §17) and `docs/HARNESS.md` (implementation notes, findings, VM re-run list) only in the sections the task needs.
`docs/STATUS.md` is the overview for readers with no prior context. See "Usage discipline" below.

- Effort checkpoints (SPEC §16): before any step on the max-effort list, stop and say "Effort checkpoint: please
  switch to max effort before I start [step]." and wait; afterwards say "Effort checkpoint complete: you can switch
  back to xhigh."
- Work on branch `harness-v1`. Commit and push only when asked, with test results in the commit message; never
  commit to or push `main`. Update `docs/HANDOFF.md` every session and `docs/STATUS.md` when a §18 item or phase
  changes status.
- Spec changes need the user's approval and a dated entry in SPEC §17.
- Milestone checkpoint (SPEC §18): announce "Milestone checkpoint: the harness build is complete and Phase 0 on the
  L4 VM is next." only when all seven conditions hold, then wait for an explicit go-ahead before creating any
  cloud resources or spending money. Never announce it early or partially.
- Mac test results show code paths only; every equality and determinism test is re-run on the L4 VM before the freeze.
- Tests: `.venv/bin/python -m pytest` (unit), `.venv-sim/bin/python -m pytest tests/reference` (simulator).
  Do not run commands through `timeout` on this Mac (x86_64 binary; breaks the arm64 Python wheels).
- No hostnames, usernames or network addresses of the team's machines in the repo.

## Usage discipline
The user's 5-hour usage limit is the project's bottleneck. In every session:
- Startup: read docs/HANDOFF.md only. Read SPEC.md, HARNESS.md or STATUS.md sections only when the current task needs them, and only the relevant sections. Confirm sync with `git fetch` and one status check; don't re-verify further.
- Keep the start-of-session summary to five lines or fewer.
- Tests: during development, run only the tests for the code being changed, with quiet output (failures only). Run the full suites once, right before each commit.
- Mutation checks: required for anything on the max-effort list in SPEC §16. For other work, ordinary tests are enough unless a test passes suspiciously easily.
- Don't spawn subagents unless the user asks.
- Reports: brief. What changed, test results in one line, decisions needed, and the next step. No narration of routine steps.
- Ask the user only for genuine decisions, phrased so they can be answered in a line.
- HANDOFF.md: update every session. STATUS.md: update only when a §18 item or phase changes status.
- Effort: tell the user which effort level the next task needs (max, xhigh, high or medium, per the schedule in HANDOFF.md), not only when max is required.
- Stopping rule: at about 85% of the 5-hour usage or half of context, stop at the next clean point, update HANDOFF.md, commit, push harness-v1 (never main), and say it's safe to clear.
