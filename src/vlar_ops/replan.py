"""The re-plan after the pilot's stop (SPEC §16 item 7; §17 amendments 61, 63, 64): `vlar-ops pilot replan`.

- Amendment 63 (3)'s offline analyses: (a) `time_fits`, X-VLA's base successes that still finish within the time
  limit under each profile's measurement charges; (c) `clock_comparison`, the frozen clock's per-call charges, a
  π0.5 request's expected wait and each profile's average measurement charge per decision point. (b) is the pilot's
  labels re-scored under every profile (`pilot.point_rows`' `positive_by`), projected in `report`.
- The budget at the pilot's measured prices (`Prices`, `coupon_budget`): the coupon's lines up to the end of the
  DAgger round (amendment 61), the label money and the points per task it buys with and without the DAgger round,
  and amendment 58's rule re-applied there under `cascader_nospread`'s profile (amendment 63 (2)).
- The other account's evaluation cost sheet (`evaluation_sheet`): amendment 61's two rounds with amendment 63's
  deciders, at the same prices.

Prices are VM-seconds (VM time divided among the jobs that shared it, as the pilot report measures them) at the
VM's hourly price. Every figure not measured is named an estimate where it is set (`Prices`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Optional, Sequence

from . import pilot

HOURLY_USD = 0.472
PROFILES_COMPARED = ("none", "router", "cascader_nospread", "cascader_chunkspread", "cascader_full")
RULE_PROFILE = pilot.STOP_PROFILE          # amendment 63 (2): cascader_nospread


# ---------------------------------------------------------------- (a) and (c): the clock (amendment 63 (3))

def base_logs(runs: Sequence[Mapping], events: Sequence[Mapping]) -> dict[str, dict[str, Any]]:
    """The 500 base rollouts: task, success at 520 steps (runs table) and event log, by episode id."""
    from vlar_train.data import group_events
    logs = group_events([e for e in events if str(e["run_id"]).startswith("base/")])
    out = {}
    for r in runs:
        rid = str(r["run_id"])
        if rid.startswith("base/") and not r.get("branch") and rid in logs:
            out[rid] = {"task_id": int(r["task_id"]), "log": logs[rid]}
    if len(out) != 500:
        raise pilot.PilotError(f"{len(out)} base rollouts with event logs, not 500")
    return out


def time_fits(base: Mapping[str, Mapping[str, Any]], *, charges, limits_s: Mapping[int, float],
              profiles: Sequence[str] = PROFILES_COMPARED) -> dict[str, Any]:
    """(a) X-VLA's successes at 520 steps (uncut base rollouts, no escalation) that still finish within their task's
    time limit when each profile's measurement charges are added, per task and overall: the time cost of measuring
    alone. Scored by the harness's own clock (`clock.score`), exact for uncut scripted logs (§5)."""
    from vlar_harness import clock as clk
    per_task: dict[int, dict[str, int]] = {}
    for rid, b in sorted(base.items()):
        t = b["task_id"]
        row = per_task.setdefault(t, {"episodes": 0, "successes": 0, **{p: 0 for p in profiles}})
        row["episodes"] += 1
        sc = {p: clk.score(b["log"], p, charges) for p in profiles}
        if sc["none"].success:
            row["successes"] += 1
        for p in profiles:
            row[p] += int(sc[p].within(limits_s[t]))
    total = {k: sum(r[k] for r in per_task.values()) for k in ("episodes", "successes", *profiles)}
    return {"per_task": per_task, "all": total,
            "share_of_successes": {p: total[p] / total["successes"] for p in profiles}}


def request_wait(net) -> dict[str, float]:
    """A routed π0.5 request's clock time from the network model's parameters (§5): each attempt fails with
    probability q (dropped, or slower than the timeout) and then costs the timeout, plus the backoff before the next
    attempt; the attempt that gets through costs upload + round trip + download + inference, the round trip
    conditioned on getting through; after 1 + max_retries failures the escalation is abandoned. Exact (log-normal
    round trip), no sampling."""
    from vlar_harness.network import p_attempt_fails
    q = p_attempt_fails(net)
    mu, s = math.log(net.rtt_median_s), net.rtt_sigma
    slack = net.timeout_s - net.upload_s - net.download_s - net.pi05_infer_s
    nd = NormalDist()
    if slack <= 0:
        raise pilot.PilotError("no attempt can get through: the timeout is below the fixed transfer and inference time")
    p_in = nd.cdf((math.log(slack) - mu) / s)
    rtt_ok = math.exp(mu + s * s / 2) * nd.cdf((math.log(slack) - mu - s * s) / s) / p_in
    ok_s = net.upload_s + net.download_s + net.pi05_infer_s + rtt_ok
    n = net.max_attempts_routed
    fail_then = net.timeout_s + net.backoff_s
    mean = sum(q ** j * (1 - q) * (j * fail_then + ok_s) for j in range(n))
    abandoned = q ** n
    mean += abandoned * (n * net.timeout_s + (n - 1) * net.backoff_s)
    return {"attempt_fail_prob": q, "ok_attempt_s": ok_s, "mean_request_s": mean, "abandon_prob": abandoned,
            "inference_s": net.pi05_infer_s, "mean_network_and_retries_s": mean - net.pi05_infer_s}


def clock_comparison(base: Mapping[str, Mapping[str, Any]], *, charges, cfg,
                     profiles: Sequence[str] = PROFILES_COMPARED) -> dict[str, Any]:
    """(c) The frozen per-call charges, a π0.5 request's expected wait (`request_wait`), and each profile's average
    measurement charge per decision point where escalation is offered, chunk and mid points weighted as they occur in
    the base rollouts (the decider is consulted at every one of them there: no escalation, budget never used)."""
    lat = cfg["latency"]
    kinds = {"chunk": 0, "mid": 0}
    for b in base.values():
        for e in b["log"]:
            if e.get("t") == "decision_point" and e.get("can_escalate"):
                kinds[e["kind"]] += 1
    n = kinds["chunk"] + kinds["mid"]
    per_point = {p: (kinds["chunk"] * charges.point_s[p]["chunk"] + kinds["mid"] * charges.point_s[p]["mid"]) / n
                 for p in profiles}
    req = request_wait(charges.network)
    return {"charges_s": {"step": charges.step_s, "xvla_act": float(lat["xvla_act_s"]),
                          "xvla_features": float(lat["xvla_features_s"]),
                          "xvla_sample_k": float(lat["xvla_sample_k_s"]), "pi05_inference": float(lat["pi05_infer_s"])},
            "point_charges_s": {p: dict(charges.point_s[p]) for p in profiles},
            "decision_points": kinds, "per_episode_points": {k: v / len(base) for k, v in kinds.items()},
            "pi05_request": req, "mean_charge_per_point_s": per_point,
            "per_point_over_pi05_request": {p: per_point[p] / req["mean_request_s"] for p in profiles}}


# ---------------------------------------------------------------- prices and the coupon's budget

@dataclass(frozen=True)
class Prices:
    """VM-seconds per unit at 7 workers (amendment 64). Measured on the pilot's resumed boot at 6 workers (branch
    runs) and on the base rollouts at 8 (a record-`full` episode); `scale_*` carry them to 7 workers: 1 keeps the
    6-worker branch price (the conservative case: the 7th worker adds no throughput), 6/7 if it adds a worker's
    share; 8/7 carries the 8-worker record-`full` price to 7 workers at the same per-worker speed (conservative),
    1 if the GPU was the limit."""
    b1: float                         # VM-s per b1 run (pilot, resumed boot: 8.28)
    b2: float                         # VM-s per b2 run (8.13)
    step: float                       # VM-s per X-VLA step in a branch-like run (b1: 0.0219)
    pi05_call: float = 0.5            # VM-s per extra π0.5 call (est.: b2 against b1 0.47, b4 against b1 ~0.5)
    features_call: float = 0.05       # VM-s per /features call (est.: 0.075 GPU-s per padded batch, shared)
    full_episode: float = 73.85       # record full, base rollouts at 8 workers (step 400: 36,925 s / 500)
    pi05_episode: float = 12.74       # π0.5 alone, parity at 8 workers (step 300: 1,274 s / 100, server start incl.)
    mean_steps: float = 312.0         # the base rollouts' mean steps per episode
    mid_points: float = 20.6          # per base episode (10.9 chunk points)
    sample_call: float = 2.0          # VM-s per /sample K call (est.: (73.9 - 6.8) VM-s over 31.5 calls + recording)
    scale_branch: float = 1.0
    scale_full: float = 8 / 7
    setup_vm_s: float = 3600.0        # a new disk: boot, 010-090 (images, weights, the VM re-run rows), golden check
    boot_vm_s: float = 480.0          # a later boot of the same disk: startup and golden check (est.)

    @staticmethod
    def from_pilot(costs: Mapping[str, Any], **kw) -> "Prices":
        r, st = costs["vm_s_per_run"], costs["vm_s_per_step"]
        return Prices(b1=float(r["b1_continue"]), b2=float(r["b2_one_call"]), step=float(st["b1_continue"]), **kw)

    @property
    def run_b1(self) -> float:
        return self.b1 * self.scale_branch

    @property
    def run_b2(self) -> float:
        return self.b2 * self.scale_branch

    @property
    def rerun(self) -> float:
        """A record-`full` re-run (exploration, DAgger) or trunk episode; a fresh simulator per episode under the
        proposed setup-count rule, so no warm-up loads (REPLAN §3)."""
        return self.full_episode * self.scale_full

    def screen(self, escalations: float, features: bool) -> float:
        """A screened (record `off`) episode: X-VLA steps, π0.5 calls, and `/features` at mid points for a live
        router-profile decider (none for exploration's profile `none`)."""
        return self.scale_branch * (self.mean_steps * self.step + escalations * self.pi05_call
                                    + (self.mid_points * self.features_call if features else 0.0))


def point_vm_s(p: Prices, parent: str, *, k: int = 2) -> float:
    """VM-s per labelled point with branches 1 and 2 only (amendment 63 (4)), 3 seeds each: a base parent is one
    seed of b1 (2 more); a re-run parent (exploration, DAgger) needs 3 b1 seeds and its share of the re-run (k
    points per re-run episode, amendment 56)."""
    if parent == "base":
        return 2 * p.run_b1 + 3 * p.run_b2
    if parent == "rerun":
        return 3 * p.run_b1 + 3 * p.run_b2 + p.rerun / k
    raise ValueError(parent)


def usd(vm_s: float, hourly_usd: float = HOURLY_USD) -> float:
    return vm_s * hourly_usd / 3600


@dataclass(frozen=True)
class Plan:
    """The coupon's money from now to the end of the DAgger round (amendment 61: evaluation is the other account's)."""
    available_usd: float              # $40 less the ledger's total
    disk_usd: float = 1.65            # ~5 days at ~$0.33 (pause rule: delete between phases if the gap is long)
    storage_usd: float = 0.50         # bucket ~25 GB + Phase 1's records, ~1 month at $0.020/GB-month
    reserve_share: float = 0.10       # of planned VM spend (amendment 61)
    split: float = 2 / 3              # of the label money to Phase 1 (amendment 61's 2:1)
    explore_episodes: int = 500
    dagger_episodes: int = 500
    explore_escalations: float = 3.1  # p = 0.1 at ~31 decision points
    dagger_escalations: float = 3.5  # est.: q 0.7 x ~5 escalations (the plan's evaluation figure)
    extra_boots: int = 3              # preemptions and the DAgger boot (est.)
    k: int = 2
    tasks: int = 10


def coupon_budget(p: Prices, plan: Plan, hourly_usd: float = HOURLY_USD) -> dict[str, Any]:
    """Fixed VM lines, then the label money L from P + reserve x P + disk + storage = available, P = fixed + L; L
    split 2:1 between Phase 1 (n base and n exploration points: n = L1 / (c_b + c_e + r/k)) and DAgger (m points
    in each half, both screened and re-run: m = L2 / (2 c_e + 2 r/k)). 'moved' (DAgger dropped, amendment 58 (b)):
    the DAgger round's screening and label money into Phase 1 points."""
    fixed = {"setup": usd(p.setup_vm_s, hourly_usd), "boots": usd(plan.extra_boots * p.boot_vm_s, hourly_usd),
             "explore_screening": usd(plan.explore_episodes * p.screen(plan.explore_escalations, False), hourly_usd),
             "dagger_screening": usd(plan.dagger_episodes * p.screen(plan.dagger_escalations, True), hourly_usd)}
    other = plan.disk_usd + plan.storage_usd
    vm = (plan.available_usd - other) / (1 + plan.reserve_share)
    labels = vm - sum(fixed.values())
    if labels <= 0:
        raise pilot.PilotError(f"no label money: fixed lines {sum(fixed.values()):.2f} > planned VM {vm:.2f}")
    l1, l2 = labels * plan.split, labels * (1 - plan.split)
    cb, ce = usd(point_vm_s(p, "base"), hourly_usd), usd(point_vm_s(p, "rerun", k=plan.k), hourly_usd)
    pair1 = cb + ce                     # one base + one exploration point (ce carries r/k)
    pair2 = 2 * ce                      # one point in each DAgger half (both re-run, amendment 63 (2))
    n, m = l1 / pair1, l2 / pair2
    moved = (l1 + l2 + fixed["dagger_screening"]) / pair1
    return {"fixed_usd": fixed, "disk_usd": plan.disk_usd, "storage_usd": plan.storage_usd,
            "planned_vm_usd": vm, "reserve_usd": vm * plan.reserve_share, "labels_usd": labels,
            "phase1_labels_usd": l1, "dagger_labels_usd": l2,
            "usd_per_point": {"base": cb, "rerun": ce}, "vm_s_per_point": {"base": point_vm_s(p, "base"),
                                                                           "rerun": point_vm_s(p, "rerun", k=plan.k)},
            "phase1_points": 2 * n, "dagger_points": 2 * m,
            "points_per_task": {"current": (2 * n + 2 * m) / plan.tasks, "moved": 2 * moved / plan.tasks},
            "vm_hours": vm / hourly_usd}


def phase_lines(b: Mapping[str, Any], *, overrun: float = 0.08, margin_usd: float = (900 + 300) / 3600 * HOURLY_USD
                ) -> list[dict[str, Any]]:
    """Amendment 59's lines for the re-plan: branch phases stop at their allocation (graceful tiers), rollout phases
    may overrun 8%; the stop point is the line less the drain margin. A boot's VM time counts in the phase it starts in
    (amendment 59), so the boots' money (preemptions, the DAgger boot) goes to the long branch phases, 2:1."""
    f = b["fixed_usd"]
    rows = [("setup (new disk: images, weights, VM re-run rows, golden check)", f["setup"], overrun),
            ("phase 1: exploration screening", f["explore_screening"], overrun),
            ("phase 1: exploration re-runs and branches (base and exploration points)",
             b["phase1_labels_usd"] + f["boots"] * 2 / 3, 0.0),
            ("DAgger: screening (router and cascader_nospread halves)", f["dagger_screening"], overrun),
            ("DAgger: re-runs and branches", b["dagger_labels_usd"] + f["boots"] / 3, 0.0)]
    return [{"phase": name, "allocation_usd": a, "line_usd": a * (1 + o), "stop_usd": max(0.0, a * (1 + o) - margin_usd)}
            for name, a, o in rows]


# ---------------------------------------------------------------- the other account's evaluation

@dataclass(frozen=True)
class EvalCase:
    q: float = 0.7                     # share of a decider's episodes that ever escalate (PROPOSALS §0.1 (a))
    first_at: float = 0.4              # first escalation, as a share of the episode
    escalations: float = 5.0           # per escalating episode
    baseline_escalations: float = 5.0
    audit_episodes: int = 20           # per decider and round (amendment 55 (6))
    disk_days: float = 2.0
    storage_usd: float = 0.50
    reserve_share: float = 0.10


def evaluation_sheet(p: Prices, case: EvalCase, *, setup: bool = True, hourly_usd: float = HOURLY_USD
                     ) -> dict[str, Any]:
    """Amendment 61's two rounds after the DAgger round, with amendment 63's deciders: pre-DAgger router and
    cascader_nospread; post-DAgger router, cascader_nospread and cascader_full; the baselines; π0.5 alone; the
    audits. Optional: cascader_chunkspread (post-DAgger) and the λ sweep (router and cascader_nospread at two more
    λ). The trunk (record full) is the X-VLA-alone baseline (check (ii) passed) and its features are used as they
    are (check (i) passed). Each suffix runs live from the first escalation."""
    ep = 500
    router_episode = p.mean_steps * p.step + p.mid_points * p.features_call          # without escalations
    chunk_points = 10.9
    episode = {"router": router_episode * p.scale_branch, "cascader_nospread": router_episode * p.scale_branch,
               "cascader_chunkspread": (router_episode + chunk_points * p.sample_call) * p.scale_branch,
               "cascader_full": p.rerun}                       # a live cascader_full makes every record-full call
    esc = lambda n: n * p.pi05_call * p.scale_branch                                # noqa: E731

    def decider(prof: str) -> float:
        return ep * case.q * ((1 - case.first_at) * episode[prof] + esc(case.escalations))

    def audit(prof: str, escalations: float) -> float:
        return case.audit_episodes * (episode[prof] + esc(escalations))

    baseline_episode = episode["router"] + esc(case.baseline_escalations)
    items = {
        "trunk (record full; also the X-VLA-alone baseline)": ep * p.rerun,
        "pre-DAgger router suffixes": decider("router"),
        "pre-DAgger cascader_nospread suffixes": decider("cascader_nospread"),
        "post-DAgger router suffixes": decider("router"),
        "post-DAgger cascader_nospread suffixes": decider("cascader_nospread"),
        "post-DAgger cascader_full suffixes": decider("cascader_full"),
        "fixed-schedule baseline (router profile, whole episodes)": ep * baseline_episode,
        "random baseline matched to the router": ep * baseline_episode,
        "random baseline matched to cascader_nospread": ep * baseline_episode,
        "π0.5 alone (500, setup count replayed)": ep * p.pi05_episode * p.scale_full,
        "audits (20 fully live per decider and round)": (
            2 * audit("router", case.escalations) + 2 * audit("cascader_nospread", case.escalations)
            + audit("cascader_full", case.escalations) + 3 * audit("router", case.baseline_escalations)),
    }
    if setup:
        items["setup (new project's VM and disk: images, weights, VM re-run rows, golden check)"] = p.setup_vm_s
    optional = {"cascader_chunkspread (post-DAgger) with its audit": decider("cascader_chunkspread")
                + audit("cascader_chunkspread", case.escalations),
                "λ sweep: router and cascader_nospread at two more λ": 4 * decider("router")}
    vm_usd = {k: usd(v, hourly_usd) for k, v in items.items()}
    vm = sum(vm_usd.values())
    disk = case.disk_days * 0.33
    return {"case": asdict(case), "vm_usd": vm_usd, "vm_total_usd": vm, "reserve_usd": vm * case.reserve_share,
            "disk_usd": disk, "storage_usd": case.storage_usd,
            "total_usd": vm * (1 + case.reserve_share) + disk + case.storage_usd,
            "optional_usd": {k: usd(v, hourly_usd) * (1 + case.reserve_share) for k, v in optional.items()},
            "vm_hours": sum(items.values()) / 3600}


# ---------------------------------------------------------------- the re-plan report

def projections_by_profile(rows: Sequence[Mapping], strata: Mapping[int, Mapping[str, Any]],
                           budgets: Mapping[str, float], pools: Mapping[str, int], *,
                           profiles: Sequence[str] = PROFILES_COMPARED, n_boot: int = 2000) -> dict[str, Any]:
    """Amendment 58's projections at `budgets` with positives scored under each profile (the rule's is
    RULE_PROFILE; the others are information, amendment 63 (3b)). A point a profile excludes is not positive."""
    out = {}
    for prof in profiles:
        pts = [{**r, "positive": bool(r["positive_by"].get(prof))} for r in rows]
        out[prof] = pilot.project(pts, strata, budgets, pools=pools, n_boot=n_boot)
    return out


def decision_at(proj: Mapping[str, Any]) -> dict[str, Any]:
    cur, mov = proj["current/stratified"], proj["moved/stratified"]
    return {"decision": pilot.decision(cur["passes"], mov["passes"]),
            "shrinkage_decision": pilot.decision(cur["shrinkage_passes"], mov["shrinkage_passes"]),
            "bootstrap_pass_share": {"current": cur["bootstrap_pass_share"], "moved": mov["bootstrap_pass_share"]}}


def report(root: Path, plan_dir: Path, pilot_report: Mapping[str, Any], *, available_usd: float,
           cfg=None, n_boot: int = 2000, hourly_usd: float = HOURLY_USD) -> dict[str, Any]:
    """Everything the re-plan rests on, from the local copies the pilot report used."""
    from vlar_harness import clock as clk
    from vlar_harness.config import load_config
    from vlar_harness.dataset import read_table
    cfg = cfg or load_config()
    if not getattr(cfg, "frozen", False):
        raise pilot.PilotError("labels are scored with the frozen calibration file (set VLAR_CALIBRATION to v1.0's)")
    limits = {int(k): float(v) for k, v in cfg["time_limit"]["per_task_s"].items()}
    charges = clk.Charges.from_config(cfg)
    runs, events = read_table(root, "runs").to_pylist(), read_table(root, "events").to_pylist()
    label_map = json.loads((Path(plan_dir) / "labels.json").read_text())
    selection = json.loads((Path(plan_dir) / "selection.json").read_text())
    rows, problems = pilot.point_rows(runs, events, label_map, limits_s=limits, charges=charges)
    base = base_logs(runs, events)
    prices = Prices.from_pilot(pilot_report["costs"])
    optimistic = replace(prices, scale_branch=6 / 7, scale_full=1.0)
    plan = Plan(available_usd=available_usd)
    budget = {"conservative": coupon_budget(prices, plan, hourly_usd),
              "optimistic": coupon_budget(optimistic, plan, hourly_usd)}
    budgets = {"current": budget["conservative"]["points_per_task"]["current"],
               "moved": budget["conservative"]["points_per_task"]["moved"],
               "optimistic_current": budget["optimistic"]["points_per_task"]["current"],
               "optimistic_moved": budget["optimistic"]["points_per_task"]["moved"]}
    pools = {k: (3 if k.endswith("current") else 2) for k in budgets}
    strata = {int(t): v for t, v in selection["phase1_strata"].items()}
    proj = projections_by_profile(rows, strata, budgets, pools, n_boot=n_boot)
    rule = proj[RULE_PROFILE]
    dec = decision_at(rule)
    opt = {"current/stratified": rule["optimistic_current/stratified"],
           "moved/stratified": rule["optimistic_moved/stratified"]}
    dec["at_optimistic_prices"] = decision_at(opt)["decision"]
    evaluation = {name: evaluation_sheet(prices, case, hourly_usd=hourly_usd) for name, case in
                  (("planned", EvalCase()), ("upper", EvalCase(q=1.0, first_at=0.2, escalations=8.0,
                                                               baseline_escalations=8.0, disk_days=3.0)))}
    return {"rule_profile": RULE_PROFILE, "labelled": len(rows), "problems": len(problems),
            "time_fits": time_fits(base, charges=charges, limits_s=limits),
            "clock": clock_comparison(base, charges=charges, cfg=cfg),
            "prices": {"conservative": asdict(prices), "optimistic": asdict(optimistic)},
            "budget": budget, "phase_lines": phase_lines(budget["conservative"]),
            "budgets_points_per_task": budgets, "pools": pools, "projections": proj, "decision": dec,
            "decision_text": pilot.DECISIONS.get(dec["decision"]), "evaluation": evaluation,
            "rates": pilot.rate_table(rows, PROFILES_COMPARED)}
