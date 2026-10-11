# vla-router

Semester project for CSCI 5942 (AI engineering).

**Question:** a simulated robot runs a small vision-language-action model (VLA) onboard. For each action chunk, a decision policy chooses between acting locally and escalating to a large VLA on a remote server, under a per-decision deadline. Can that decision beat both "always local" and "always remote" on success within the deadline, once every cost is counted (network, queueing, and the policy's own overhead)?

**Status:** Phase 0 (smoke tests). No implementation code yet.

## Documents

- [docs/PLAN.md](docs/PLAN.md): the implementation plan and design reference.
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): every assumption and unknown, and the evidence for each item we have verified.

## Conventions

- Every claim in the docs is labeled as verified by running it, stated in a source, an assumption, or unknown.
- Model weights, third-party clones, decision records and videos stay out of git (see `.gitignore`).
- Hostnames, usernames and network addresses of our machines do not go in this repo.
