# Skill installation

The canonical project Skill is `.agents/skills/local-ai-autopilot`. Installing
it makes the instructions discoverable; it does not install the LocalPilot
Python package, inference engines, model weights, drivers, or remote-node
software.

Install the Skill for another supported Agent with:

```bash
python3 .agents/skills/local-ai-autopilot/scripts/install.py install \
  --agent claude-code --target .
```

Supported agents are `codex`, `claude-code`, `cursor`, `gemini`, and `all`.
The installer refuses to overwrite or remove unmanaged or locally modified
Skill directories.

After installation, verify the separate runtime prerequisite on every target:

```bash
command -v localpilot
localpilot --version
localpilot --target ssh://user@host --version
```

This Skill targets the LocalPilot `0.2.x` CLI contract. A different major or
minor series requires a compatibility review before the workflow is executed.
