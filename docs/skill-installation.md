# Skill installation

The installable Skill package is [`skills/local-ai-autopilot`](../skills/local-ai-autopilot/SKILL.md).
The matching `.agents/skills/local-ai-autopilot` directory is a generated
project-local discovery copy, checked by `python3 scripts/sync_skill.py`.
Installing the Skill makes its instructions discoverable; it does not install the LocalPilot
Python package, inference engines, model weights, drivers, or remote-node
software.

In Codex, ask `$skill-installer` to install the repository path
`https://github.com/DingHappy/localpilot/tree/main/skills/local-ai-autopilot`.
From a cloned repository, install for another supported Agent with:

```bash
python3 scripts/install_skill.py install \
  --agent claude-code --target .
```

Supported agents are `codex`, `claude-code`, `cursor`, `gemini`, and `all`.
The installer refuses to overwrite or remove unmanaged or locally modified
Skill directories.

After installation, verify the separate runtime prerequisite on the controller
and every execution target:

```bash
command -v localpilot
localpilot --version
localpilot --target ssh://user@host --version
```

If a target lacks the CLI, the Skill recommends an isolated installation from
the project repository using the steps in
[`references/cli.md`](../skills/local-ai-autopilot/references/cli.md#installing-a-missing-cli).
It does not silently download or install the CLI. An explicit request to set
up LocalPilot can proceed through those steps and verify the installed version.

This Skill targets the LocalPilot `0.2.x` CLI contract. A different major or
minor series requires a compatibility review before the workflow is executed.
