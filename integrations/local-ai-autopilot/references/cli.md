# LocalPilot CLI contract

Run commands from the LocalPilot repository root unless LOCALPILOT_HOME points
to another project state directory.

## Read-only inspection

    localpilot doctor --json
    localpilot profile

Doctor reports real_execution_ready. A false value means real deployment must
not be claimed.

## Planning and execution

    localpilot recommend --task coding --mode auto
    localpilot autopilot "user goal" --mode auto --json

Modes:

- auto chooses OpenVINO only when an Intel CPU and OpenVINO devices are detected.
- mock runs a deterministic development simulation.
- openvino requires installed OpenVINO packages and LOCALPILOT_MODEL_PATH.

Every candidate and benchmark carries a simulated flag.

## State and service

    localpilot status --json
    localpilot benchmark
    localpilot demo --mode mock
    localpilot stop
    localpilot serve

Stop preserves profile files. Serve is foreground-only in the MVP and binds to
127.0.0.1 by default.

## Result interpretation

- status success means the individual candidate executed.
- score is assigned only after quality and stability gates pass.
- profile_reused means a matching profile passed a fresh health check.
- READY with simulated true is a development result, not production readiness.
- Recovery candidates include fallback_of and recovery_action. Do not retry
  beyond the configured recovery limit.
