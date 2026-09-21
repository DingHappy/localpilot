from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from localpilot.agents.base import Agent
from localpilot.benchmark.prompts import keyword_hit, prompts_for_task, load_benchmark_config
from localpilot.benchmark.structured import score_fields, summarize_fields
from localpilot.utils import stable_hash
from localpilot.planner.policies import PolicyEngine
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import CandidatePlan


RUBRIC = """You are grading one answer from a code-review assistant.

Task given to the assistant:
{prompt}

The answer is correct only if it identifies this issue: {expected}

Answer to grade:
{answer}

Score three things from 0.0 to 1.0:
  correctness   - does it identify the actual issue, with no wrong claims
  completeness  - does it cover the issue rather than gesture at it
  actionability - could a developer fix the code from this answer alone

Reply with JSON only, no prose:
{{"correctness": <float>, "completeness": <float>, "actionability": <float>, "note": "<10 words>"}}"""


class JudgeAgent(Agent):
    """Scores answer quality against a rubric instead of matching keywords.

    Keyword matching cannot separate a correct review from one that merely
    contains the right word: "this has no race condition" matches "race".
    That flaw sits directly under the quality gate, so a wrong quality
    score picks the wrong winner. This agent asks a model to grade against
    an explicit rubric, blends that with the keyword signal, and records
    which signals were actually available.
    """

    name = "judge"
    role = "Grades candidate output quality against a rubric"

    def __init__(self, policies: PolicyEngine = None) -> None:
        super().__init__()
        self.policies = policies or PolicyEngine()
        self.config = self.policies.judge

    # ------------------------------------------------------------------

    def evaluate(
        self,
        runtime: RuntimeProvider,
        candidate: CandidatePlan,
        task: str,
    ) -> Dict[str, Any]:
        """Runs while the candidate's server is still up.

        Called by the executor rather than after it, because grading needs
        the model that produced the answers to still be loaded.
        """
        prompts = prompts_for_task(task)
        if not prompts:
            return self._result(None, None, "no benchmark prompts for task")

        if any("expected_fields" in prompt for prompt in prompts):
            if not all(prompt.get("expected_fields") for prompt in prompts):
                raise ValueError("Do not mix structured fields and keyword-only prompts in one task")
            return self._evaluate_fields(runtime, prompts)

        samples = []
        keyword_hits = 0
        for prompt in prompts:
            try:
                answer = runtime.generate(
                    prompt,
                    max_new_tokens=int(self.config.get("max_new_tokens", 256)),
                )
            except Exception as exc:
                samples.append(
                    {
                        "prompt_id": prompt.get("id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            hit = keyword_hit(answer, prompt)
            keyword_hits += 1 if hit else 0
            samples.append(
                {
                    "prompt_id": prompt.get("id"),
                    "keyword_hit": hit,
                    "answer": answer[:600],
                    "expected_terms": prompt["expected_terms"],
                }
            )

        graded = [item for item in samples if "answer" in item]
        keyword_quality = keyword_hits / max(1, len(prompts))

        if not self.config.get("enabled", False):
            return self._result(
                keyword_quality, None, "judge disabled in policies", samples
            )
        if not graded:
            return self._result(
                keyword_quality, None, "no answers to grade", samples
            )

        judge_scores = []
        for item in graded:
            verdict = self._grade(
                prompt=self._prompt_text(prompts, item["prompt_id"]),
                expected=", ".join(item["expected_terms"]),
                answer=item["answer"],
                runtime=runtime,
                candidate=candidate,
            )
            if verdict is None:
                continue
            item["judge"] = verdict
            judge_scores.append(verdict["score"])

        if not judge_scores:
            self.record(
                "judge_unavailable",
                status="degraded",
                detail=(
                    "No judge model answered; the keyword score stands alone "
                    "and the profile records that."
                ),
            )
            return self._result(
                keyword_quality, None, "judge model unavailable", samples
            )

        judge_quality = sum(judge_scores) / len(judge_scores)
        blended = self._blend(keyword_quality, judge_quality)
        self.record(
            "grade_candidate",
            detail=(
                f"{candidate.candidate_id}: keyword {keyword_quality:.2f}, "
                f"rubric {judge_quality:.2f}, blended {blended:.2f}"
            ),
            data={
                "candidate_id": candidate.candidate_id,
                "keyword": round(keyword_quality, 3),
                "judge": round(judge_quality, 3),
                "blended": round(blended, 3),
                "graded": len(judge_scores),
            },
        )
        return self._result(
            keyword_quality, judge_quality, "rubric grading applied", samples
        )

    # ------------------------------------------------------------------

    def _evaluate_fields(self, runtime, prompts):
        samples = []
        config = load_benchmark_config()
        for prompt in prompts:
            error = None
            try:
                answer = runtime.generate(prompt, max_new_tokens=int(config.get("max_new_tokens", 768)))
            except Exception as exc:
                answer = ""
                error = type(exc).__name__
            sample = score_fields(answer, prompt["expected_fields"])
            sample["prompt_id"] = prompt.get("id")
            if error:
                sample["error"] = error
            # Store verdicts, never the document text or model answer.
            samples.append(sample)
        summary = summarize_fields(samples)
        summary["dataset_sha256"] = stable_hash(prompts)
        self.record("grade_fields", detail="Deterministic JSON field acceptance", data=summary)
        return {
            "keyword": None, "judge": None,
            "blended": summary["document_accuracy"],
            "detail": "Exact JSON document match; not a semantic judge score",
            "structured": summary, "samples": samples,
        }

    def _prompt_text(self, prompts: List[Dict[str, Any]], prompt_id: Any) -> str:
        for prompt in prompts:
            if prompt.get("id") == prompt_id:
                return prompt["text"]
        return ""

    def _blend(self, keyword: float, judge: float) -> float:
        keyword_weight = float(self.config.get("keyword_weight", 0.4))
        judge_weight = float(self.config.get("judge_weight", 0.6))
        total = keyword_weight + judge_weight
        if total <= 0:
            return judge
        return (keyword * keyword_weight + judge * judge_weight) / total

    def _result(
        self,
        keyword: Optional[float],
        judge: Optional[float],
        detail: str,
        samples: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if keyword is None:
            blended = None
        elif judge is None:
            blended = keyword
        else:
            blended = self._blend(keyword, judge)
        return {
            "keyword": None if keyword is None else round(keyword, 3),
            "judge": None if judge is None else round(judge, 3),
            "blended": None if blended is None else round(blended, 3),
            "detail": detail,
            "samples": samples or [],
        }

    def _grade(
        self,
        prompt: str,
        expected: str,
        answer: str,
        runtime: RuntimeProvider,
        candidate: CandidatePlan,
    ) -> Optional[Dict[str, Any]]:
        rendered = RUBRIC.format(prompt=prompt, expected=expected, answer=answer)

        raw = self._ask_external_judge(rendered)
        source = "external"
        if raw is None and candidate.simulated:
            return self._simulated_verdict(answer, expected)
        if raw is None:
            return None

        parsed = self._parse_verdict(raw)
        if parsed is None:
            return None
        parsed["source"] = source
        return parsed

    def _ask_external_judge(self, rendered: str) -> Optional[str]:
        """Asks a judge endpoint, which must be a separate deployment.

        Grading a candidate with the candidate itself would let a weak model
        certify its own answers, so there is no fallback to the candidate's
        own server here.
        """
        base_url = os.environ.get("LOCALPILOT_JUDGE_BASE_URL")
        if not base_url:
            return None
        model = os.environ.get("LOCALPILOT_JUDGE_MODEL", "localpilot-judge")
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": rendered}],
                "max_tokens": 200,
                "temperature": 0.0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            self.record(
                "judge_request_failed",
                status="degraded",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return None

    def _parse_verdict(self, raw: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        try:
            parts = [
                float(data["correctness"]),
                float(data["completeness"]),
                float(data["actionability"]),
            ]
        except (KeyError, TypeError, ValueError):
            return None
        parts = [min(1.0, max(0.0, value)) for value in parts]
        return {
            "correctness": parts[0],
            "completeness": parts[1],
            "actionability": parts[2],
            "score": round(sum(parts) / 3, 3),
            "note": str(data.get("note", ""))[:120],
        }

    def _simulated_verdict(self, answer: str, expected: str) -> Dict[str, Any]:
        """Deterministic stand-in so the rubric path is exercised off-target.

        Rewards a specific, actionable answer over one that merely contains
        the expected term, which is the distinction the real judge exists to
        make.
        """
        lowered = answer.lower()
        terms = [term.strip().lower() for term in expected.split(",")]
        correctness = 1.0 if any(term in lowered for term in terms) else 0.2
        if any(
            negation in lowered
            for negation in ("no issue", "not a problem", "no race", "is safe")
        ):
            correctness = min(correctness, 0.1)
        completeness = min(1.0, len(answer.split()) / 25)
        actionability = (
            0.9
            if any(
                cue in lowered
                for cue in ("use ", "replace", "instead", "guard", "wrap")
            )
            else 0.4
        )
        parts = [correctness, completeness, actionability]
        return {
            "correctness": round(correctness, 3),
            "completeness": round(completeness, 3),
            "actionability": round(actionability, 3),
            "score": round(sum(parts) / 3, 3),
            "note": "simulated rubric",
            "source": "simulated",
        }
