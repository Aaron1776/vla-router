"""Phase 2: decider training (SPEC §6-§8), outside the frozen harness (§15). Runs on the user's Mac or the VM from a
local copy of the records: `python -m vlar_train` (needs numpy, torch and pyarrow: `.venv-sim` locally).

- `data`: training points from the runs, events and points tables, the branch planner's label map and the frozen
  calibration (labels and remaining time per profile from `vlar_harness.labels`, gripper fractions offline).
- `deciders`: each decider's inputs (§6), one MLP family differing only in input width (§8).
- `folds`: leave-one-task-out with a separate validation task (§7).
- `train`: from scratch per fold and decider, a checkpoint every epoch, early stopping on the validation loss.
- `lam`: λ on the validation task from branch labels (§8).
- `stats`: effective sample size, re-weighted label statistics and episode-bootstrap intervals (amendments 57, 68).
"""
