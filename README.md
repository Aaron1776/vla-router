# vla-router

Semester project for CSCI 5942 (AI engineering).

**Question:** a simulated robot runs a small vision-language-action model (VLA) onboard. For each action chunk, a decision policy chooses between acting locally and escalating to a large VLA on a remote server, under a per-decision deadline. Can that decision beat both "always local" and "always remote" on success within the deadline, once every cost is counted (network, queueing, and the policy's own overhead)?

**Team:** Oscar Rodriguez and Miguel Ramirez-Aleman.

**Status:** building the Phase 0 harness for the X-VLA (local) / π0.5 (remote) design in [docs/SPEC.md](docs/SPEC.md), which replaces the SmolVLA plan. Done so far: versioned harness config, client API draft, job planner and resumable queue, spend guard, archive packaging and the GCP scripts. Not yet built: the simulator layer, adapters, model servers, clock and snapshots.

## Documents

- [docs/CONCEPT.md](docs/CONCEPT.md): working draft of the two-page concept document, due Oct 13.
- [docs/SPEC.md](docs/SPEC.md): the current design: budgeted escalation from X-VLA to π0.5 on LIBERO-Long, router vs. cascader.
- [docs/PLAN.md](docs/PLAN.md): the original plan (SmolVLA local model); superseded by SPEC.md where they differ.
- [ops/README.md](ops/README.md): Google Cloud setup, spend guard and pause/resume runbook.
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): every assumption and unknown, and the evidence for each item we have verified.
- [envs/README.md](envs/README.md): how each machine's environment was set up.

## Conventions

- Every claim in the docs is labeled as verified by running it, stated in a source, an assumption, or unknown.
- Model weights, third-party clones, decision records and videos stay out of git (see `.gitignore`).
- Hostnames, usernames and network addresses of our machines do not go in this repo.
