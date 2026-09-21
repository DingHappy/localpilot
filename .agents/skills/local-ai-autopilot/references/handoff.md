# Configuration handoff

Use this when the user wants a reusable result rather than ongoing service
management. Match the deliverable to what was requested and actually verified.
Applications may call the existing inference engine directly; no LocalPilot
inference gateway, telemetry daemon or automatic model switch is required.

## Deliverable

Provide the applicable items, using actual verified values rather than guessed
paths, flags or compatibility:

- Execution target, engine, served model name and engine endpoint. For a remote
  engine bound to loopback, that address belongs to the node, not the controller.
  Explain the authorised connection or SSH tunnel needed by the application.
- Configuration file and launch command, or identify the already-running service
  to attach to. Mark an unexecuted launch command as proposed, not deployed.
- Request example with the needed prompt, input modality, response_format and
  output budget. Keep user documents and credentials out of public artifacts.
- Acceptance configuration, data/version hash, gates, run ID and report paths;
  distinguish measured, simulated and proposed portions.
- Changes from baseline, observed tradeoffs, rejected alternatives and remaining
  validation gaps. Describe any explicitly requested apply/recovery procedure.

For an OpenAI-compatible engine, the application can send a prepared request
file directly (the endpoint/model must have been verified):

```bash
curl "$ENGINE_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

The request carries `model`, `messages`, and the needed generation parameters.
For image tasks preserve the image content block; for constrained output include
response_format in the actual request. Do not include expected_fields or
mock_answer, which belong only to the evaluation dataset. A schema declares
structure and types; unseen inputs still require validation.

## Completion boundaries

- Proposal only: deliver a candidate and explain what has not been measured.
- Acceptance: deliver the measured result, including rejection if gates failed.
- Handoff: provide the configuration and usable invocation instructions; execute
  a small request only if runtime verification is requested or needed and authorised.
- A saved Profile remembers a decision; it does not automatically apply a request
  template to the application, keep a service online, or establish production quality.

If the user separately requests LocalPilot's existing API or guarded switching,
read [service-management.md](service-management.md). Otherwise complete handoff
without introducing an online control layer.
