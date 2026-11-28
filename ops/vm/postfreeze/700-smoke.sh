#!/usr/bin/env bash
# cap-seconds: 4800
# Post-freeze smoke job (the user's item 5, 2026-11-24): one boot, nothing written to the dataset. In order:
#  1. w8's evidence, read-only, to the bucket first: w8 stopped at ~00:48Z on 2026-11-24 during the base rollouts
#     without failing or renewing its lease; the cause is only on this disk (its unit's journal, the kernel log, the
#     queue database's rows).
#  2. Both production servers, the image cross-check (amendment 54) and the golden-set replay. Any mismatch: stop
#     (exit 76), no debugging.
#  3. Probe neutrality with the real models (`vlar-ops smoke probe`), then π0.5 determinism (`smoke branch`).
#  4. Three base episodes re-run by the production worker under the ops waiver into a separate storage root
#     (<storage>/smoke/<date>/), never the dataset's, and compared with the originals (`smoke compare`).
# Every report goes to <storage>/smoke/<date>/reports/. A mismatch in 3 or 4 does not stop the later checks (the boot
# is paid for); the step then ends with exit 76 so nothing is queued behind it.
source /opt/vla-router/ops/vm/phase0/lib.sh
need_images
: "${VLAR_STORAGE_URL:?}"
DAY="$(date -u +%F)"
SURL="${VLAR_STORAGE_URL}/smoke/${DAY}"
SD="/var/lib/vlar/smoke-${DAY}"
REP="${SD}/reports"
SUBJECTS="base/t2/i34,base/t8/i19,base/t1/i15"
mkdir -p "${REP}"
FAILED=()

# ---- 1. w8's evidence (read-only); the runner also mirrors logs/ to the bucket after the step
W8="${P0D}/logs/w8-evidence"; mkdir -p "${W8}"
{
  export TZ=UTC
  journalctl --list-boots --no-pager > "${W8}/boots.txt" 2>&1
  journalctl --disk-usage > "${W8}/journal-disk-usage.txt" 2>&1; ls -la /var/log/journal >> "${W8}/journal-disk-usage.txt" 2>&1
  W=(--since "2026-11-23 20:00:00" --until "2026-11-24 09:00:00" --no-pager -o short-iso-precise)
  journalctl -u vlar-worker@8 "${W[@]}" > "${W8}/vlar-worker-8.journal.txt" 2>&1
  journalctl _TRANSPORT=kernel "${W[@]}" > "${W8}/kernel.journal.txt" 2>&1   # not -k: it implies this boot only
  journalctl -u docker -u containerd "${W[@]}" > "${W8}/docker.journal.txt" 2>&1
  journalctl --since "2026-11-24 00:30:00" --until "2026-11-24 01:15:00" --no-pager -o short-iso-precise \
    > "${W8}/all-units-0030-0115.journal.txt" 2>&1
  for f in /var/log/kern.log*; do [[ -e "${f}" ]] && cp -p "${f}" "${W8}/"; done
  grep -hE '^2026-11-24T0(0:[3-5]|1:0)' /var/log/syslog /var/log/syslog.1 > "${W8}/syslog-0030-0110.txt" 2>/dev/null
  python3 - /var/lib/vlar/queue.sqlite "${W8}" <<'PY'
import json, sqlite3, sys
src, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
dst = sqlite3.connect(f"{out}/queue.sqlite"); con.backup(dst); dst.close()
con.row_factory = sqlite3.Row
rows = lambda q, *a: [dict(r) for r in con.execute(q, a).fetchall()]
cols = [r["name"] for r in rows("PRAGMA table_info(jobs)")]
body = lambda r: {k: v for k, v in r.items() if k != "body"} | {"subject": json.loads(r["body"])["subject"]}
json.dump({"columns": cols,
           "base_t1_i15": [body(r) for r in rows("SELECT * FROM jobs WHERE job_id=?", "d01957c2fd54254403b371d0")],
           "attempted_or_errored": [body(r) for r in rows("SELECT * FROM jobs WHERE attempts > 0 OR last_error IS NOT NULL")],
           "status_counts": rows("SELECT phase, status, COUNT(*) AS n FROM jobs GROUP BY phase, status")},
          open(f"{out}/queue-rows.json", "w"), indent=1, default=str)
PY
} || true
if gcloud storage rsync -r "${W8}" "${VLAR_PHASE0_URL}/w8-evidence" >/dev/null 2>&1 \
   && [[ "$(gcloud storage ls "${VLAR_PHASE0_URL}/w8-evidence/" | wc -l)" -ge "$(ls "${W8}" | wc -l)" ]]; then
  echo "w8 evidence in the bucket: $(ls "${W8}" | tr '\n' ' ')"
else
  echo "WARNING: w8 evidence NOT confirmed in the bucket (keep the disk)"; FAILED+=("w8-evidence-upload")
fi

# ---- 2. servers, image cross-check, golden replay
[[ -s /etc/vlar/batching.env ]] || write_batching_env
calibration_frozen || fail "the calibration file is not frozen"
systemctl start vlar-xvla vlar-pi05 || fail "model servers did not start"
wait_servers || { systemctl stop vlar-xvla vlar-pi05; fail "model servers not answering /info after 10 minutes"; }
golden_check || { systemctl stop vlar-xvla vlar-pi05; gate_fail "golden-set replay or image cross-check failed: stopped, no debugging"; }
cp /var/lib/vlar/golden-pass.json "${REP}/golden-pass.json"
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m["passed"] and m.get("image_check") else 1)' \
  "${REP}/golden-pass.json" || gate_fail "the golden-pass marker records no image check"
echo "golden set and image cross-check passed on boot $(cat /proc/sys/kernel/random/boot_id)"

# ---- 3. probe neutrality and π0.5 determinism, through the production servers
smoke_in_sim() {
  docker run --rm --gpus all --network host -v /etc/vlar:/etc/vlar:ro -v /var/lib/vlar:/var/lib/vlar \
    -v "${R}/ops/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl "${CAL_ENV[@]}" "${OPS_RUN[@]}" "vlar-sim:${C}" vlar-ops smoke "$@"
}
for what in probe branch; do
  smoke_in_sim "${what}" --golden "${CAL_DIR}/golden/golden.json" --out "${REP}/${what}.json" || FAILED+=("${what}")
done

# ---- 4. three base episodes re-run by the production worker under the waiver, into the smoke root
for a in $(gcloud storage ls "${VLAR_STORAGE_URL}/phase0/src/" | grep -E '/[0-9a-f]{40}\.tar\.gz$'); do
  gcloud storage cp -n -q "${a}" "${SURL}/phase0/src/" || fail "archive ${a} not copied to the smoke root"
done
sed -e "s#^storage_url: .*#storage_url: ${SURL}#" -e "s#^local_root: .*#local_root: ${SD}#" "${R}/ops/ops.yaml" \
  > "${SD}/ops.yaml"
ln -sfn /var/lib/vlar/STOP "${SD}/STOP"            # the spend guard's stop flag reaches these workers too
smoke_in_sim plan --subjects "${SUBJECTS}" --out "${SD}/plan.json" || fail "smoke plan"
VLAR_OPS_CONFIG="${SD}/ops.yaml" vlar-ops queue add "${SD}/plan.json" || fail "smoke queue add"
VLAR_OPS_CONFIG="${SD}/ops.yaml" vlar-ops queue resume >/dev/null
for i in 1 2 3; do
  docker run --rm --name "vlar-smoke-s${i}" --gpus all --network host -v /etc/vlar:/etc/vlar:ro \
    -v "${SD}/ops.yaml:/etc/vlar-ops/ops.yaml:ro" -v /var/lib/vlar:/var/lib/vlar -e VLAR_OPS_CONFIG=/etc/vlar-ops/ops.yaml \
    "${OPS_RUN[@]}" "${CAL_ENV[@]}" -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    "vlar-sim:${C}" vlar-ops worker --worker-id "s${i}" --phase base > "${SD}/worker-s${i}.log" 2>&1 &
done
wait
cp "${SD}"/worker-s*.log "${REP}/"
VLAR_OPS_CONFIG="${SD}/ops.yaml" vlar-ops queue sync >/dev/null
echo "re-runs: $(VLAR_OPS_CONFIG="${SD}/ops.yaml" vlar-ops queue status --phase base | tail -n 1)"
systemctl stop vlar-xvla vlar-pi05
smoke_in_sim compare --subjects "${SUBJECTS}" --smoke-url "${SURL}" --out "${REP}/compare.json" || FAILED+=("compare")

gcloud storage rsync -r "${REP}" "${SURL}/reports" >/dev/null 2>&1 || echo "WARNING: reports not uploaded"
(( ${#FAILED[@]} == 0 )) || gate_fail "smoke: ${FAILED[*]} failed (reports in ${SURL}/reports): no debugging on the VM"
reason "smoke passed: golden set and image check, probe neutrality, π0.5 repeat, 3 base re-runs bit for bit"
