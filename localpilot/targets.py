from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urlsplit


class TargetError(RuntimeError):
    pass


@dataclass(frozen=True)
class SSHTarget:
    destination: str
    port: int | None = None

    @classmethod
    def parse(cls, value: str) -> "SSHTarget":
        parsed = urlsplit(value)
        if parsed.scheme != "ssh" or not parsed.hostname:
            raise TargetError(
                "Remote target must use ssh://[user@]host[:port]"
            )
        if parsed.hostname.startswith("-"):
            raise TargetError("SSH target host cannot start with '-'")
        if parsed.password is not None:
            raise TargetError(
                "Do not put passwords in the target URI; use SSH keys or an "
                "interactive SSH prompt"
            )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise TargetError(
                "Remote target may contain only user, host, and optional port"
            )
        username = unquote(parsed.username) if parsed.username else None
        destination = f"{username}@{parsed.hostname}" if username else parsed.hostname
        try:
            port = parsed.port
        except ValueError as exc:
            raise TargetError(f"Invalid SSH target port: {exc}") from exc
        return cls(destination=destination, port=port)

    def command(self, remote_argv: List[str], executable: str) -> List[str]:
        if not remote_argv:
            raise TargetError("A LocalPilot command is required for a remote target")
        if remote_argv[0] == "serve":
            raise TargetError(
                "Remote `serve` is not proxied. Run it on the node and use an "
                "SSH tunnel to keep the dashboard bound to 127.0.0.1."
            )
        ssh = shutil.which("ssh")
        if ssh is None:
            raise TargetError("ssh was not found on this controller")
        executable_argv = shlex.split(executable)
        if not executable_argv:
            raise TargetError("Remote LocalPilot command cannot be empty")
        command = [ssh]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.extend(
            [
                self.destination,
                shlex.join([*executable_argv, *remote_argv]),
            ]
        )
        return command


def extract_target_options(argv: List[str]) -> tuple[str, str, List[str]]:
    """Extract controller-only flags wherever the user placed them."""
    target = os.environ.get("LOCALPILOT_TARGET", "local")
    executable = os.environ.get("LOCALPILOT_REMOTE_COMMAND", "localpilot")
    remaining: List[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in {"--target", "--remote-command"}:
            if index + 1 >= len(argv):
                raise TargetError(f"{item} requires a value")
            value = argv[index + 1]
            if item == "--target":
                target = value
            else:
                executable = value
            index += 2
            continue
        if item.startswith("--target="):
            target = item.split("=", 1)[1]
            index += 1
            continue
        if item.startswith("--remote-command="):
            executable = item.split("=", 1)[1]
            index += 1
            continue
        remaining.append(item)
        index += 1
    return target, executable, remaining


def run_on_target(target: str, remote_argv: List[str], executable: str) -> int:
    if target == "local":
        raise TargetError("The local target must execute in-process")
    command = SSHTarget.parse(target).command(remote_argv, executable)
    try:
        return int(subprocess.run(command, check=False).returncode)
    except OSError as exc:
        raise TargetError(f"Could not start SSH: {exc}") from exc
