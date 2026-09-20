#!/usr/bin/env python3
"""Harvests a day's facts into a paste-ready journal block.

A development journal has to be written while the development happens; on
the last day nobody can reconstruct which run taught them what. This pulls
together what the repository already knows about a given day -- commits,
runs, exported reports, whether anything real was measured -- so the only
part left to write by hand is the part a tool cannot know: what it meant.

    python3 scripts/devlog.py              # today
    python3 scripts/devlog.py 2026-09-23   # a specific day
    python3 scripts/devlog.py --since 2026-09-20 --until 2026-09-29
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPETITION_START = date(2026, 9, 20)


def run(command: list, merge_stderr: bool = False) -> str:
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    output = result.stdout
    if merge_stderr:
        output += result.stderr
    return output.strip()


def commits(since: str, until: str) -> list:
    raw = run([
        "git", "log",
        f"--since={since} 00:00", f"--until={until} 23:59",
        "--date=short", "--pretty=%h|%ad|%s",
    ])
    rows = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            rows.append({"sha": parts[0], "date": parts[1], "subject": parts[2]})
    return rows


def diffstat(since: str, until: str) -> str:
    raw = run([
        "git", "log",
        f"--since={since} 00:00", f"--until={until} 23:59",
        "--shortstat", "--pretty=",
    ])
    files = insertions = deletions = 0
    for line in raw.splitlines():
        for chunk in line.split(","):
            chunk = chunk.strip()
            number = chunk.split(" ", 1)[0]
            if not number.isdigit():
                continue
            if "file" in chunk:
                files += int(number)
            elif "insertion" in chunk:
                insertions += int(number)
            elif "deletion" in chunk:
                deletions += int(number)
    if not files:
        return "no commits"
    return f"{files} files, +{insertions} / -{deletions}"


def _in_window(stamp: str, since: str, until: str) -> bool:
    if not stamp:
        return False
    day = stamp[:10]
    return since <= day <= until


def runs(since: str, until: str) -> list:
    directory = ROOT / "profiles" / "runs"
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stamp = data.get("finished_at") or data.get("started_at") or ""
        if not _in_window(stamp, since, until):
            continue
        hardware = data.get("hardware") or {}
        best = data.get("best_profile") or {}
        benchmark = best.get("benchmark") or {}
        intent = data.get("intent") or {}
        found.append({
            "when": stamp[11:19],
            "simulated": bool(hardware.get("simulated")),
            "platform": hardware.get("platform_id"),
            "task": intent.get("task"),
            "priority": intent.get("priority"),
            "goal": (intent.get("raw_text") or "")[:60],
            "candidates": len(data.get("candidates") or []),
            "reused": bool(data.get("profile_reused")),
            "winner": (best.get("candidate") or {}).get("candidate_id", ""),
            "engine": (best.get("candidate") or {}).get("engine"),
            "ttft_ms": benchmark.get("ttft_ms"),
            "tok_s": benchmark.get("throughput_tokens_s"),
            "memory_gb": benchmark.get("peak_memory_gb"),
            "memory_source": (benchmark.get("raw") or {}).get("peak_memory_source"),
            "quality_judge": benchmark.get("quality_judge"),
            "score": best.get("score"),
        })
    return found


def reports(since: str, until: str) -> list:
    directory = ROOT / "results"
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.md")):
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        if since <= stamp <= until:
            found.append(path.name)
    return found


def failures(since: str, until: str) -> list:
    """Failed candidates are the part worth writing about."""
    path = ROOT / "logs" / "localpilot.jsonl"
    if not path.exists():
        return []
    found = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") not in {
                    "candidate_failed", "recovery_started"
                }:
                    continue
                if not _in_window(record.get("timestamp", ""), since, until):
                    continue
                found.append({
                    "when": record["timestamp"][11:19],
                    "event": record["event"],
                    "details": record.get("details", {}),
                })
    except OSError:
        return []
    return found


def test_count() -> str:
    # unittest writes its summary to stderr.
    raw = run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        merge_stderr=True,
    )
    for line in reversed(raw.splitlines()):
        if line.startswith("Ran "):
            return line
    return "test run produced no summary"


def emit(since: str, until: str) -> None:
    label = since if since == until else f"{since} .. {until}"
    day_number = (date.fromisoformat(since) - COMPETITION_START).days + 1
    header = f"## Day {day_number} — {label}" if 1 <= day_number <= 10 else f"## {label}"

    print(header)
    print()

    day_runs = runs(since, until)
    real = [item for item in day_runs if not item["simulated"]]
    print(f"- 提交：{diffstat(since, until)}")
    print(f"- 运行：{len(day_runs)} 次"
          + (f"，其中 **{len(real)} 次真实硬件**" if real else "（全部为模拟）"))
    if reports(since, until):
        print(f"- 导出报告：{', '.join(reports(since, until))}")
    print(f"- 测试：{test_count()}")
    print()

    if commits(since, until):
        print("<details><summary>当天提交</summary>")
        print()
        for item in commits(since, until):
            print(f"- `{item['sha']}` {item['subject']}")
        print()
        print("</details>")
        print()

    if day_runs:
        collapsed = {}
        for item in sorted(day_runs, key=lambda row: row["when"]):
            key = (
                item["simulated"], item["task"], item["priority"],
                item["winner"], item["engine"],
            )
            if key in collapsed:
                collapsed[key]["count"] += 1
                collapsed[key]["when"] = item["when"]
            else:
                collapsed[key] = {**item, "count": 1}
        # Real measurements first: they are the ones worth writing about.
        shown = sorted(
            collapsed.values(),
            key=lambda row: (row["simulated"], row["when"]),
        )
        hidden = max(0, len(shown) - 14)
        shown = shown[:14]

        print("| 时间 | 任务/优先级 | 冠军配置 | 引擎 | TTFT | tok/s | 显存 | 分数 | 次数 | 实测? |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for item in shown:
            def fmt(value, digits=1, suffix=""):
                return "-" if value is None else f"{value:.{digits}f}{suffix}"
            flag = "模拟" if item["simulated"] else "**实测**"
            if item["reused"]:
                flag += " · 命中"
            print(
                f"| {item['when']} | {item['task']}/{item['priority']} "
                f"| `{item['winner'][:34]}` | {item['engine']} "
                f"| {fmt(item['ttft_ms'], 0, ' ms')} | {fmt(item['tok_s'])} "
                f"| {fmt(item['memory_gb'], 1, ' GB')} | {fmt(item['score'])} "
                f"| {item['count']} | {flag} |"
            )
        if hidden:
            print()
            print(f"> 另有 {hidden} 组配置未列出（同配置重复运行已合并）")
        print()

        if real:
            sources = {item["memory_source"] for item in real if item["memory_source"]}
            if sources:
                print(f"> 峰值显存来源：{', '.join(sorted(sources))}")
            ungraded = [item for item in real if item["quality_judge"] is None]
            if ungraded:
                print(f"> ⚠ {len(ungraded)} 次实测没有 rubric 评分，"
                      "质量这一维是弱信号（judge 端点未配置？）")
            print()

    day_failures = failures(since, until)
    if day_failures:
        print("<details><summary>失败与恢复（写进征文的素材通常在这里）</summary>")
        print()
        for item in day_failures:
            print(f"- `{item['when']}` **{item['event']}** — {item['details']}")
        print()
        print("</details>")
        print()

    print("**今天想清楚了什么**（一句话，工具答不了这个）：")
    print()
    print("**今天错在哪**（预测和实测差多少？为什么？）：")
    print()
    print("**明天第一件事**：")
    print()
    print("---")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", nargs="?", default=None, help="YYYY-MM-DD，默认今天")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument(
        "--all", action="store_true", help="逐日输出整个比赛窗口"
    )
    args = parser.parse_args(argv)

    if args.all:
        for offset in range(10):
            day = (COMPETITION_START + timedelta(days=offset)).isoformat()
            if day > datetime.now(timezone.utc).date().isoformat():
                break
            emit(day, day)
            print()
        return 0

    if args.since or args.until:
        since = args.since or args.until
        until = args.until or args.since
    else:
        day = args.day or datetime.now().date().isoformat()
        since = until = day
    emit(since, until)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
