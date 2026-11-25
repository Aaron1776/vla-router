#!/usr/bin/env bash
# cap-seconds: 14400
# spend-phase: phase1b-topup
# Phase 1, Tranche B, the one pre-approved top-up of the screen (SPEC §17 amendment 68): 730's screen run again under
# its own spend phase, on the enlarged screen plan uploaded to <storage>/phase1b/screen/plan.json (each short task's
# size grown by `top_up_decision`; the same sequence, so every episode already screened keeps its job id). `queue add`
# leaves known job ids as they are, so only the new episodes run; the golden gate runs again on this boot, and the
# step ends with the yield over the whole screen. Boots only after the top-up decision passed (<= $1.00, no task's
# observed failure rate under 2%) or with the user's approval.
exec bash /opt/vla-router/ops/vm/postfreeze/730-phase1b-screen.sh
