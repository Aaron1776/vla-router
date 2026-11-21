"""Handoff rules, controller mode switching, the budget and the run modes (effort step 3; SPEC §3-§4,
§11 step 3: "forced handoffs in both directions must produce no arm jump").

All on the production episode engine (`vlar_harness.episode.Episode`) with the deterministic fake
policies from episode_driver.py. "No arm jump" is checked where a jump would come from, at every
forced handoff:
- X-VLA -> π0.5: the first π0.5 step runs in delta mode, so the controller's goal is within one
  delta step (0.05 m per axis) of the arm;
- π0.5 -> X-VLA and abandoned escalations: X-VLA's request carries the measured pose exactly (the
  controller's cached pose), and its first absolute target is near the arm;
- every executed step runs in its model's controller mode.
Sensitivity checks show these catch a skipped handoff rule. With the real models the same checks run on
the VM (test_model_handoff_continuity_on_the_vm).
"""

import os

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot
from vlar_harness.config import load_config

CFG = load_config()
TASK = 0
INIT = 4
ROLLOUT = 1_000_004
DELTA_REACH = 0.05 * np.sqrt(3)          # robosuite OSC: output_max 0.05 m per axis per step


class Recorder:
    """X-VLA transport wrapper and step hook that record what continuity depends on."""

    def __init__(self, sim, inner):
        self.sim, self.inner = sim, inner
        self.requests = []                 # (step, payload proprio, measured pos, measured 6D rotation)
        self.steps = []                    # per executed step

    def transport(self, payload, seed):
        import json_numpy
        from vlar_harness import reference
        lc = reference.xvla_libero_client()
        pos, mat = self.sim.controller_pose()
        self.requests.append({"step": self.sim.steps, "proprio": np.asarray(json_numpy.loads(payload["proprio"])),
                              "pos": np.array(pos), "rot6d": lc.LiberoAbsActionProcessor().Mat_to_Rotate6D(mat)})
        return self.inner(payload, seed)

    def on_step(self, episode, action, driver):
        c = episode.sim.env.robots[0].controller
        self.steps.append({"step": episode.sim.steps, "driver": driver, "use_delta": bool(c.use_delta),
                           "goal_pos": np.array(c.goal_pos), "eef": np.array(episode.sim.obs["robot0_eef_pos"])})


class FailAt:
    """Remote link whose requests fail on every attempt at the given steps (1 try + 2 retries)."""

    def __init__(self, steps):
        self.fail_steps = set(steps)

    def request(self, call, *, step, rollout_seed):
        from vlar_harness.episode import RemoteOutcome
        if step in self.fail_steps:
            return RemoteOutcome(False, None, [{"attempt": i, "ok": False} for i in range(3)])
        return RemoteOutcome(True, np.asarray(call(), dtype=np.float32), [{"attempt": 0, "ok": True}])


def _spec():
    from vlar_harness.sim.libero_sim import SimSpec
    return SimSpec.from_config(CFG)


@pytest.fixture(scope="module")
def sim():
    from vlar_harness.sim.libero_sim import LiberoSim
    s = LiberoSim(TASK, _spec())
    yield s
    s.close()


@pytest.fixture(scope="module")
def other_sim():
    from vlar_harness.sim.libero_sim import LiberoSim
    s = LiberoSim(TASK, _spec())
    yield s
    s.close()


def make(sim, *, link=None, state_mode="quirk", xvla=None):
    from episode_driver import FakePI05, FakeXVLA
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.episode import Episode

    rec = Recorder(sim, xvla or FakeXVLA())
    ep = Episode(sim, XVLAAdapter(rec.transport, state_mode=state_mode), PI05Adapter(sim), FakePI05(), CFG,
                 remote=link, on_step=rec.on_step)
    return ep, rec


def run_routed(ep, schedule, *, until=520, init=INIT, rollout=ROLLOUT, on_point=None):
    dp = ep.start(init, rollout, "routed")
    while dp is not None and dp.step < until:
        if on_point is not None:
            on_point(ep, dp)
        dp = ep.decide("escalate" if schedule(dp.index) and dp.can_escalate else "local").next_point
    return ep


ESCALATE_AT = {2, 3, 6, 11, 12, 13, 14, 17}       # mid-chunk, consecutive, chunk point, a run of four


def check_continuity(rec, ep, first_target_limit=0.03):
    """Every handoff in `rec`, both directions; returns a list of problems. `first_target_limit` bounds
    X-VLA's first target after π0.5 (3 cm for the fake models; relative for the real ones, SPEC §11 step 3)."""
    problems = []
    by_step = {s["step"]: s for s in rec.steps}
    for s in rec.steps:                                       # every step in its model's controller mode
        if s["use_delta"] != (s["driver"] == "pi05"):
            problems.append(f"step {s['step']}: {s['driver']} ran with use_delta={s['use_delta']}")
    prev = None
    for s in rec.steps:
        start = s["step"] - 1
        if prev is not None and s["driver"] != prev["driver"]:
            if s["driver"] == "pi05":
                gap = np.linalg.norm(s["goal_pos"] - prev["eef"])
                if gap > DELTA_REACH + 0.01:
                    problems.append(f"step {start}: X-VLA->pi0.5 goal {gap:.3f} m from the arm")
            else:
                req = [r for r in rec.requests if r["step"] == start]
                if not req:
                    problems.append(f"step {start}: no fresh X-VLA request at the handoff")
                else:
                    r = req[-1]
                    if not (np.array_equal(r["proprio"][:3], r["pos"]) and np.array_equal(r["proprio"][3:9], r["rot6d"])):
                        problems.append(f"step {start}: X-VLA was not sent the measured pose")
                    gap = np.linalg.norm(s["goal_pos"] - r["pos"])
                    if gap > first_target_limit:
                        problems.append(f"step {start}: pi0.5->X-VLA first target {gap:.3f} m from the arm")
        prev = s
    return problems


# ---------------------------------------------------------------- continuity

@requires_lerobot
@pytest.mark.parametrize("state_mode", ["quirk", "measured"])
def test_forced_handoffs_in_both_directions_have_no_arm_jump(sim, state_mode):
    ep, rec = make(sim, state_mode=state_mode)
    run_routed(ep, lambda d: d in ESCALATE_AT, until=200)
    drivers = [s["driver"] for s in rec.steps]
    switches = sum(a != b for a, b in zip(drivers, drivers[1:]))
    assert switches >= 8, "the schedule should force handoffs in both directions"
    assert check_continuity(rec, ep) == []
    assert ep.state.pi05_steps == 10 * len(ESCALATE_AT)


@requires_lerobot
def test_abandoned_escalation_hands_back_to_xvla_from_the_measured_pose(sim):
    ep, rec = make(sim, link=FailAt({60, 120}))
    budgets = {}

    def watch(e, dp):
        budgets[dp.step] = dp.budget_remaining_steps

    run_routed(ep, lambda d: d in {6, 12, 13}, until=170, on_point=watch)
    assert check_continuity(rec, ep) == []
    assert ep.state.pi05_requests == 3 and ep.state.pi05_failed_requests == 6
    assert ep.state.pi05_steps == 10                           # only the request at step 130 got through
    assert budgets[70] == budgets[60] and budgets[130] == budgets[120]    # failures charge no budget
    for step in (60, 120):
        seg = [s for s in rec.steps if step < s["step"] <= step + 10]
        assert {s["driver"] for s in seg} == {"xvla"} and not any(s["use_delta"] for s in seg)
        fresh = [r for r in rec.requests if r["step"] == step]
        assert np.array_equal(fresh[-1]["proprio"][:3], fresh[-1]["pos"])     # act_fresh: measured pose
    abandoned = [e for e in ep.events if e["t"] == "escalation_abandoned"]
    assert [e["step"] for e in abandoned] == [60, 120]
    assert [e["call"] for e in ep.events if e["t"] == "xvla_act" and e["step"] == 60] == ["act", "act_fresh"]


@requires_lerobot
def test_a_skipped_handoff_rule_is_caught(sim):
    """Sensitivity: forgetting X-VLA's remembered pose is what makes the return smooth."""
    ep, rec = make(sim)
    ep.xvla.forget_predicted_pose = lambda: None              # handoff rule removed
    run_routed(ep, lambda d: d in {2, 3, 6}, until=110)
    problems = check_continuity(rec, ep)
    assert any("measured pose" in p for p in problems), problems
    assert any("first target" in p for p in problems), problems


@requires_lerobot
@pytest.mark.parametrize("skip", ["delta", "absolute"])
def test_a_missing_mode_switch_is_refused(sim, skip):
    from vlar_harness.sim.libero_sim import LiberoSim

    ep, rec = make(sim)
    real = LiberoSim.set_control_mode
    sim.set_control_mode = lambda mode: None if mode == skip else real(sim, mode)
    try:
        with pytest.raises(RuntimeError, match="wrong controller mode"):
            run_routed(ep, lambda d: d in {2, 4}, until=80)
    finally:
        del sim.set_control_mode


# ---------------------------------------------------------------- budget

@requires_lerobot
def test_budget_allows_26_escalations_and_then_refuses(sim):
    from vlar_harness.episode import BudgetExhausted

    ep, rec = make(sim)
    dp = ep.start(INIT, ROLLOUT, "routed")
    escalations = 0
    while dp is not None:
        if dp.can_escalate:
            dp = ep.decide("escalate").next_point
            escalations += 1
        else:
            with pytest.raises(BudgetExhausted):
                ep.decide("escalate")
            assert dp.budget_remaining_steps == 0
            dp = ep.decide("local").next_point
    assert escalations == 26 and ep.state.pi05_steps == 260
    res = ep.result()
    assert res.steps == 520 and res.end_reason == "step_limit" and res.pi05_requests == 26
    assert check_continuity(rec, ep) == []


# ---------------------------------------------------------------- run modes

@requires_lerobot
def test_routed_always_local_is_exactly_xvla_only(sim, other_sim):
    """X-VLA's executed behaviour is the same in every mode (§3)."""
    a, ra = make(sim)
    run_routed(a, lambda d: False)
    b, rb = make(other_sim)
    b.start(INIT, ROLLOUT, "xvla_only")
    b.run_episode()
    assert len(ra.steps) == len(rb.steps) == 520
    for x, y in zip(a.trajectory, b.trajectory):
        assert x["step"] == y["step"] and np.array_equal(x["action"], y["action"])
        assert np.array_equal(x["robot0_eef_pos"], y["robot0_eef_pos"])
    assert [r["proprio"].tobytes() for r in ra.requests] == [r["proprio"].tobytes() for r in rb.requests]


@requires_lerobot
def test_xvla_only_matches_the_authors_client_loop(sim):
    """xvla_only against the authors' own rollout loop (libero_client.py:289-306), same scene seed (142 + k)."""
    from episode_driver import FakeXVLA
    from vlar_harness import reference

    lc = reference.xvla_libero_client()
    seedless = FakeXVLA()
    fake = lambda payload, seed: seedless(payload, None)       # the authors' client sends no seed

    ev = lc.LIBEROEval(task_suite_name="libero_10", eval_horizon=900, act_type="abs", num_episodes=1, init_seed=42)
    env, lang, obs = ev._init_env(ev.task_suite_list[0], TASK, INIT)
    client = lc.ClientModel.__new__(lc.ClientModel)
    client.processor = lc.LiberoAbsActionProcessor()
    client.reset()
    client._post = lambda payload: np.array(fake(payload, None)["action"])
    ref = []
    try:
        for _ in range(90):
            obs["robo_ori"] = client.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            obs["robo_pos"] = env.env.robots[0].controller.ee_pos
            action = client.step(obs, lang)
            obs, _, done, _ = env.step(action)
            ref.append((action, np.array(obs["robot0_eef_pos"])))
    finally:
        env.close()

    ep, _ = make(sim, xvla=fake)
    ep.start(INIT, ROLLOUT, "xvla_only")
    ep.step_limit = 90
    ep.run_episode()
    assert ep.state.scene_seed == 142 + INIT
    for (a_ref, p_ref), x in zip(ref, ep.trajectory):
        assert np.array_equal(a_ref, x["action"]) and np.array_equal(p_ref, x["robot0_eef_pos"])


@requires_lerobot
def test_pi05_only_calls_every_ten_steps_in_delta_mode(sim):
    from episode_driver import FakePI05

    calls = []
    fake = FakePI05()

    def pi05(raw, task, seed):
        out = fake(raw, task, seed)
        calls.append((sim.steps, out))
        return out

    ep, rec = make(sim)
    ep.pi05_call = pi05
    ep.start(INIT, ROLLOUT, "pi05_only")
    ep.step_limit = 60
    ep.run_episode()
    assert [c[0] for c in calls] == [0, 10, 20, 30, 40, 50]
    assert all(s["use_delta"] and s["driver"] == "pi05" for s in rec.steps)
    executed = np.stack([x["action"] for x in ep.trajectory])
    assert np.array_equal(executed, np.concatenate([c[1][:10] for c in calls]))
    assert rec.requests == []                                  # X-VLA never called


# ---------------------------------------------------------------- snapshots through the engine

@requires_lerobot
def test_engine_snapshot_restore_continues_exactly(sim, other_sim):
    from vlar_harness import snapshot as snap

    schedule = lambda d: d in {3, 4, 8}
    snaps = {}
    ep, rec = make(sim)
    run_routed(ep, schedule, until=130,
               on_point=lambda e, dp: snaps.__setitem__(dp.index, snap.to_bytes(e.snapshot())))
    original = {x["step"]: x for x in ep.trajectory}
    for d in (2, 4, 5, 7, 9):
        ep2, rec2 = make(other_sim)
        dp = ep2.restore(snap.from_bytes(snaps[d]))
        assert dp.index == d and dp.step == 10 * d
        while dp is not None and dp.step < 130:
            dp = ep2.decide("escalate" if schedule(dp.index) else "local").next_point
        for x in ep2.trajectory:
            o = original[x["step"]]
            assert np.array_equal(o["action"], x["action"]) and np.array_equal(o["robot0_eef_pos"], x["robot0_eef_pos"])
        assert ep2.state.pi05_steps == ep.state.pi05_steps


# ---------------------------------------------------------------- the real models (VM)

XVLA_URL = os.environ.get("VLAR_XVLA_SERVER")       # host:port of a running xvla_server
PI05_URL = os.environ.get("VLAR_PI05_SERVER")
# First-step target distances (m) X-VLA produced at ordinary chunk boundaries (no handoff) in the base
# rollouts: a JSON list of floats, extracted from their event logs after Phase 0 step 5 (see docs/HANDOFF.md).
BASE_FIRST_TARGETS = os.environ.get("VLAR_BASE_FIRST_TARGETS")


@pytest.mark.skipif(not (XVLA_URL and PI05_URL and BASE_FIRST_TARGETS),
                    reason="needs both model servers (VLAR_XVLA_SERVER, VLAR_PI05_SERVER) and the base rollouts' "
                           "chunk-boundary distances (VLAR_BASE_FIRST_TARGETS)")
@requires_lerobot
@pytest.mark.parametrize("task", [0, 2, 8])
def test_model_handoff_continuity_on_the_vm(task):
    """The same continuity checks with the real models, on the VM (SPEC §11 step 3, amendment 10). X-VLA's
    first target after a handoff must be at or below the 99th percentile of the first-step target distances
    it produces at ordinary chunk boundaries in the base rollouts; the fixed 3 cm is for the fakes only."""
    import json
    ordinary = np.asarray(json.loads(open(BASE_FIRST_TARGETS).read()), dtype=float)
    assert ordinary.size >= 100, "too few ordinary chunk boundaries to take a 99th percentile"
    limit = float(np.percentile(ordinary, 99))
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.episode import Episode
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    from vlar_harness.sim.libero_sim import LiberoSim

    xh, xp = XVLA_URL.split(":")
    ph, pp = PI05_URL.split(":")
    xvla_http, pi05_http = XVLAClient(xh, int(xp)), PI05Client(ph, int(pp))
    s = LiberoSim(task, _spec())
    try:
        rec = Recorder(s, lambda payload, seed: xvla_http.act(payload, seed))
        ep = Episode(s, XVLAAdapter(rec.transport), PI05Adapter(s),
                     lambda raw, task_, seed: pi05_http.act(raw, task_, seed)["actions"], CFG, on_step=rec.on_step)
        run_routed(ep, lambda d: d in ESCALATE_AT, until=200, rollout=ROLLOUT + task)
        problems = check_continuity(rec, ep, first_target_limit=limit)
        assert problems == [], f"limit {limit:.4f} m (99th percentile of ordinary boundaries): {problems}"
    finally:
        s.close()
