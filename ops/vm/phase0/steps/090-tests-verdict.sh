#!/usr/bin/env bash
# cap-seconds: 600
# The re-run rows' verdict: every row passed -> done; any failed -> failed (the VM stops, nothing is debugged);
# rows not run (e.g. no tokenizer yet) -> needs the user. Prints each row's last line (pytest's pass/skip counts).
source /opt/vla-router/ops/vm/phase0/lib.sh
failed=(); notrun=()
for f in "${RESULTS}"/*.status; do
  [[ -e "${f}" ]] || continue
  n="$(basename "${f}" .status)"; s="$(cat "${f}")"
  printf '%-28s %-22s %s\n' "${n}" "${s}" "$(tail -n 1 "${RESULTS}/${n}.log" 2>/dev/null)"
  case "${s}" in passed) ;; "not run"*) notrun+=("${n}") ;; *) failed+=("${n}") ;; esac
done
(( ${#failed[@]} == 0 )) || fail "re-run rows failed: ${failed[*]} (logs/results/<row>.log)"
(( ${#notrun[@]} == 0 )) || need_input "re-run rows not run: ${notrun[*]} (see their .status)"
echo "every re-run row passed"
