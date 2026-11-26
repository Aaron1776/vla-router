#!/usr/bin/env bash
# cap-seconds: 1800
# spend-phase: phase1b
# SPEC §17 amendment 74 (the user, 2026-11-26): before 740 resumes after its first boot's exit 76, re-compare every
# re-run that boot finished, with the fixed comparison (`phase1b.compare_rerun`: the runs table's job_id skipped, the
# re-run's own job id; every other runs column, the steps and the events bit for bit, as before). The first boot's
# mismatch marker (<work>/phase1b/rerun/MISMATCH.json; 740's `mismatch_stop` gate-fails while it exists) is set aside,
# renamed and kept (and copied to the bucket's reports), only if:
#   - it is the marker diagnosed on the Mac: one failed re-run, explore/t6/i43/r2, "runs: column job_id differs first
#     at row 0";
#   - exactly 8 re-runs have markers (those that finished on 740's first boot), all 8 compared, all 8 passed.
# Anything else: exit 76 (the runner stops the VM; the user decides). No model servers, no simulation: compare-b
# reads the bucket's markers and records. Neither the mismatch marker nor the stop flag is written here.
source "${VLAR_LIB:-/opt/vla-router/ops/vm/phase0/lib.sh}"    # VLAR_LIB, VLAR_WORK_ROOT: tests only
need_images
: "${VLAR_STORAGE_URL:?}"
PH=phase1b
W="${VLAR_WORK_ROOT:-/var/lib/vlar}/${PH}"
RD="${W}/rerun"; REP="${W}/reports"
RURL="${VLAR_STORAGE_URL}/${PH}/reports"
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
MISMATCH="${RD}/MISMATCH.json"
OUT="${REP}/recompare-739-${BOOT}.json"
EXPECT_FINISHED=8
EXPECT_MARKER='{"failed": [{"problems": ["runs: column job_id differs first at row 0"], "subject": "explore/t6/i43/r2"}], "n_failed": 1}'
mkdir -p "${RD}" "${REP}"

upload_reports() {
  local rc=$?
  echo "{\"boot_id\": \"${BOOT}\", \"step\": \"739\", \"step_exit\": ${rc}, \"at\": $(date +%s)}" >> "${REP}/step-exits.jsonl"
  gcloud storage rsync -r "${REP}" "${RURL}" >/dev/null 2>&1 || echo "WARNING: ${PH} reports not uploaded"
}
trap upload_reports EXIT
trap 'exit 143' TERM INT

ops_sim() {   # vlar-ops in the sim image (pyarrow, the harness config), the bucket and the work dir mounted
  docker run --rm --network host -v /etc/vlar:/etc/vlar:ro -v /var/lib/vlar:/var/lib/vlar \
    -v "${R}/ops/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml \
    "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" vlar-ops "$@"
}

# ---- the marker must be the one diagnosed (a later difference would not be this defect)
[[ -s "${MISMATCH}" ]] || gate_fail "739: no mismatch marker at ${MISMATCH}: not the state diagnosed on the Mac; nothing set aside"
python3 - "${MISMATCH}" "${EXPECT_MARKER}" <<'EOF' || gate_fail "739: the mismatch marker is not the one diagnosed ($(head -c 300 "${MISMATCH}" | tr '\n' ' ')): nothing set aside"
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])) == json.loads(sys.argv[2]) else 1)
EOF

# ---- the re-run plan: 740's, from the bucket (the VM's copy must be the same file)
gcloud storage cp -q "${VLAR_STORAGE_URL}/${PH}/rerun/plan.json" "${REP}/recompare-plan.json" \
  || gate_fail "739: the re-run plan could not be read from the bucket"
cmp -s "${REP}/recompare-plan.json" "${RD}/plan.json" || gate_fail "739: the VM's re-run plan differs from the bucket's"
rm -f "${REP}/recompare-plan.json"

# ---- every finished re-run, compared afresh (a new --out: no earlier row is taken as passed)
ops_sim phase1 compare-b --plan-dir "${RD}" --out "${OUT}"
rc=$?
[[ ${rc} -eq 76 ]] && gate_fail "739: a re-run still differs under the fixed comparison ($(head -c 400 "${OUT}" | tr '\n' ' ')): the marker stays"
[[ ${rc} -ne 0 ]] && gate_fail "739: compare-b failed (exit ${rc}): the marker stays"
python3 - "${OUT}" "${EXPECT_FINISHED}" <<'EOF' || gate_fail "739: not exactly ${EXPECT_FINISHED} re-runs compared and passed (see ${OUT}): the marker stays"
import json, sys
r, n = json.load(open(sys.argv[1])), int(sys.argv[2])
rows = r["rows"].values()
ok = (r["passed"] and r["compared"] == n and sum(x["passed"] is True for x in rows) == n
      and sum(x["passed"] is None for x in rows) == len(r["rows"]) - n)
print(json.dumps({"compared": r["compared"], "pending": r["pending"], "passed": r["passed"], "ok": ok}))
sys.exit(0 if ok else 1)
EOF

# ---- set the first boot's marker aside, kept on the VM and in the bucket's reports
ASIDE="${RD}/MISMATCH-set-aside-740-boot1.json"
mv "${MISMATCH}" "${ASIDE}" || gate_fail "739: the marker could not be set aside"
cp -p "${ASIDE}" "${REP}/"
reason "739: all ${EXPECT_FINISHED} finished re-runs pass the fixed comparison; the first boot's mismatch marker set aside (${ASIDE##*/}); 740 resumes"
