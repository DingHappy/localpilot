# Optional workflow composition

LocalPilot owns target inspection, candidate decisions, acceptance evidence,
and reusable profiles. Another Skill may own a specialised workflow after
LocalPilot establishes the configuration or encounters a boundary.

Use a specialised Skill only when it is discoverable in the current Agent
environment:

- A Dynamo Skill may continue into serving recipes or routing after LocalPilot
  selects and validates a configuration.
- A Jetson Skill should own JetPack, driver, and Jetson-specific serving or
  tuning work. LocalPilot does not configure Jetson systems.
- An engine-specific troubleshooting Skill may continue after LocalPilot's
  bounded recovery fails for vLLM, TensorRT-LLM, SGLang, NIM, Ollama, or
  llama.cpp.

If no matching Skill is available, report the boundary, preserve LocalPilot's
structured error and evidence, and give the smallest concrete next step. Do
not claim that a missing Skill was invoked or silently substitute a different
workflow.

Passing work to another Skill does not change the evidence owner: measurements
remain attached to the target and run that produced them.
