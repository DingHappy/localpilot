from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from localpilot.engines.introspect import ServerConfig, introspect, reconcile
from localpilot.engines.registry import EngineRegistry, EngineSpec
from localpilot.models.selector import ModelSelector, SelectionOutcome
from localpilot.sizing import MemoryModel, decode_roofline_tokens_s
from localpilot.planner.policies import PolicyEngine
from localpilot.schemas import CandidatePlan, HardwareProfile, Intent, ModelSpec


class Planner:
    """Builds the candidate set for one intent.

    The search space on a unified-memory machine is not "which device" --
    there is one accelerator. It is the serving configuration: engine,
    weight precision, KV precision, context, batch width, and whether to
    spend spare compute on speculative decoding.
    """

    def __init__(
        self,
        policies: PolicyEngine = None,
        engines: EngineRegistry = None,
        memory_model: MemoryModel = None,
    ) -> None:
        self.policies = policies or PolicyEngine()
        self.engines = engines or EngineRegistry()
        self.memory = memory_model or MemoryModel(self.policies.memory_config)
        self.selector = ModelSelector(self.memory)
        self.last_selection: Optional[SelectionOutcome] = None
        self.last_restriction: Optional[Dict[str, Any]] = None
        self.attached_config: Optional[ServerConfig] = None
        self.last_reconciliation: Optional[Dict[str, Any]] = None

    def plan(
        self,
        intent: Intent,
        hardware: HardwareProfile,
        models: List[ModelSpec],
        runtime: str,
    ) -> List[CandidatePlan]:
        engine_ids = self._candidate_engines(hardware, runtime)
        if not engine_ids:
            raise RuntimeError(
                "No inference engine is available. Install vLLM, TensorRT-LLM, "
                "SGLang or NIM, or run with --mode mock."
            )

        models, restriction = self._restrict_to_served(models, engine_ids, runtime)
        self.last_restriction = restriction

        selection = self.selector.select(intent, hardware, models, engine_ids)
        self.last_selection = selection
        if not selection.selected:
            raise RuntimeError(self._no_model_message(intent, selection))

        policy = self.policies.priority(intent.priority)
        preferred_engines = [
            engine_id
            for engine_id in policy.get("prefer_engine", engine_ids)
            if engine_id in engine_ids
        ] or engine_ids

        candidates: List[Tuple[float, CandidatePlan]] = []
        for model in selection.selected:
            for engine_rank, engine_id in enumerate(preferred_engines):
                if engine_id not in model.engines:
                    continue
                try:
                    engine = self.engines.get(engine_id)
                except KeyError:
                    continue
                for knobs in self._knob_combinations(
                    engine, model, intent, hardware, policy
                ):
                    candidate = self._build_candidate(
                        model=model,
                        engine=engine,
                        engine_rank=engine_rank,
                        intent=intent,
                        hardware=hardware,
                        runtime=runtime,
                        knobs=knobs,
                    )
                    if candidate is None:
                        continue
                    prior = self._candidate_prior(
                        model, candidate, hardware, intent, engine_rank
                    )
                    candidates.append((prior, candidate))

        if not candidates:
            raise RuntimeError(
                "Every model and engine combination was gated out before "
                "execution. Run `localpilot recommend --json` to see why."
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = self._diversify(candidates)
        return self._reconcile_with_server(chosen)

    def _reconcile_with_server(
        self, candidates: List[CandidatePlan]
    ) -> List[CandidatePlan]:
        """Keeps only candidates the attached server would really run.

        A knob in a candidate is an instruction to an engine LocalPilot
        starts. Against a server someone else started it is only a claim:
        the server has one cache dtype, one utilization, one context limit,
        and answers every request with those. Measuring a candidate whose
        knobs differ from the server's produces a number correctly labelled
        as something it is not -- so an attached run collapses to the one
        configuration that is actually running.
        """
        config = self.attached_config
        if config is None or not config.raw_available or not candidates:
            self.last_reconciliation = None
            return candidates

        # Utilization is the one knob worth adopting rather than rejecting
        # over: it describes how much of the pool the server was given, not
        # what the candidate is. Adopting it keeps the candidate honest
        # about the configuration it will actually be measured under, and
        # the trace records that it was adopted rather than requested.
        adopted = []
        if config.gpu_memory_utilization is not None:
            for candidate in candidates:
                declared = (candidate.runtime_config or {}).get(
                    "gpu_memory_utilization"
                )
                if (
                    declared is not None
                    and abs(float(declared) - config.gpu_memory_utilization) > 0.01
                ):
                    candidate.runtime_config = dict(candidate.runtime_config)
                    candidate.runtime_config["gpu_memory_utilization"] = (
                        config.gpu_memory_utilization
                    )
                    candidate.knobs = dict(candidate.knobs)
                    candidate.knobs["gpu_memory_utilization"] = (
                        config.gpu_memory_utilization
                    )
                    adopted.append(
                        f"{candidate.candidate_id}: utilization "
                        f"{declared} -> {config.gpu_memory_utilization}"
                    )

        kept, rejected = [], []
        for candidate in candidates:
            mismatches = reconcile(candidate, config)
            if mismatches:
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "mismatches": mismatches,
                    }
                )
            else:
                kept.append(candidate)

        self.last_reconciliation = {
            "server": config.to_dict(),
            "kept": [item.candidate_id for item in kept],
            "rejected": rejected,
            "adopted": adopted,
        }
        if not kept:
            raise RuntimeError(
                "No candidate matches the attached server's actual "
                "configuration:\n  "
                + "\n  ".join(
                    f"{item['candidate_id']}: {'; '.join(item['mismatches'])}"
                    for item in rejected
                )
                + "\nEither restart the server with the configuration you "
                "want measured, or set LOCALPILOT_ALLOW_ENGINE_LAUNCH=1 so "
                "LocalPilot can configure one per candidate."
            )
        return kept

    def _diversify(
        self, ranked: List[Tuple[float, CandidatePlan]]
    ) -> List[CandidatePlan]:
        """Spend the candidate budget on distinct models before knob variants.

        Ranking by prior alone fills every slot with one model's near
        identical configurations, and measuring four of those answers a
        narrower question than the user asked. Covering distinct models
        first makes the comparison informative; leftover slots then go to
        the knob variants of the strongest model.
        """
        limit = self.policies.max_candidates
        per_model = max(1, self.policies.max_candidates_per_model)

        by_model: Dict[str, List[CandidatePlan]] = {}
        model_order: List[str] = []
        seen_ids = set()
        for _, candidate in ranked:
            if candidate.candidate_id in seen_ids:
                continue
            seen_ids.add(candidate.candidate_id)
            if candidate.model_id not in by_model:
                by_model[candidate.model_id] = []
                model_order.append(candidate.model_id)
            by_model[candidate.model_id].append(candidate)

        # An engine that compiles per configuration charges a build for every
        # candidate it appears in. Four of those turns one run into an hour of
        # compiling before a single token is measured, so a plan may only
        # spend a limited number of slots on them.
        #
        # A model whose best variant is priced out must fall back to its next
        # one rather than forfeit its slot, so each model carries its own
        # cursor instead of every model sharing a depth index.
        rebuild_budget = self.policies.max_rebuilding_candidates
        rebuilding = 0
        cursors = {model_id: 0 for model_id in model_order}
        taken = {model_id: 0 for model_id in model_order}

        selected: List[CandidatePlan] = []
        for _ in range(per_model):
            progressed = False
            for model_id in model_order:
                if len(selected) >= limit:
                    return selected
                if taken[model_id] >= per_model:
                    continue
                bucket = by_model[model_id]
                while cursors[model_id] < len(bucket):
                    candidate = bucket[cursors[model_id]]
                    cursors[model_id] += 1
                    rebuilds = self._needs_rebuild(candidate.engine)
                    if rebuilds and rebuilding >= rebuild_budget:
                        continue
                    if rebuilds:
                        rebuilding += 1
                    selected.append(candidate)
                    taken[model_id] += 1
                    progressed = True
                    break
            if not progressed:
                break
        return selected[:limit]

    def _restrict_to_served(
        self, models: List[ModelSpec], engine_ids: List[str], runtime: str
    ) -> Tuple[List[ModelSpec], Optional[Dict[str, Any]]]:
        """Drops models the attached server cannot serve.

        In attach mode LocalPilot talks to a server someone else started,
        and that server has one model loaded. Planning candidates for other
        checkpoints does not compare them: every request is answered by
        whatever is actually resident, so four candidates return four
        measurements of one configuration. The numbers look plausible and
        the comparison is meaningless, which is worse than an error.

        So the plan is cut to what the server serves, and the reason is
        recorded rather than left for the reader to infer from four
        suspiciously similar rows.
        """
        if runtime == "mock":
            return models, None

        served: List[str] = []
        self.attached_config = None
        for engine_id in engine_ids:
            try:
                spec = self.engines.get(engine_id)
            except KeyError:
                continue
            config = introspect(engine_id, spec.openai_base_path or "/v1")
            if config is None:
                continue
            if self.attached_config is None and config.served_models:
                self.attached_config = config
            served.extend(config.served_models)
        if not served:
            return models, None

        lowered = {name.lower() for name in served}

        def matches(model: ModelSpec) -> bool:
            return (
                model.model_id.lower() in lowered
                or model.source_id.lower() in lowered
                or any(
                    model.model_id.lower() in name or name in model.model_id.lower()
                    for name in lowered
                )
            )

        kept = [model for model in models if matches(model)]
        dropped = [model.model_id for model in models if not matches(model)]
        restriction = {
            "served": served,
            "kept": [model.model_id for model in kept],
            "dropped": dropped,
        }
        if not kept:
            raise RuntimeError(
                "The attached server serves "
                f"{served}, which matches no model registered for this task. "
                "Point LOCALPILOT_*_BASE_URL at a server running a "
                "registered model, or add this one to config/models.yaml."
            )
        return kept, restriction

    def _needs_rebuild(self, engine_id: str) -> bool:
        try:
            return self.engines.get(engine_id).rebuild_per_configuration
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # candidate construction

    def _candidate_engines(
        self, hardware: HardwareProfile, runtime: str
    ) -> List[str]:
        if runtime == "mock":
            # Mock mode models every engine so the search space can be
            # developed and demonstrated without the target machine.
            return [
                engine_id
                for engine_id in self.engines.ids()
                if self.engines.servable(engine_id)
            ]
        installed = [
            engine_id
            for engine_id in hardware.stack.get("engines_available", [])
            if self.engines.servable(engine_id)
        ]
        if runtime in installed:
            return [runtime]
        return installed

    def _knob_combinations(
        self,
        engine: EngineSpec,
        model: ModelSpec,
        intent: Intent,
        hardware: HardwareProfile,
        policy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        space = self.policies.search_space
        concurrency = intent.concurrency or int(policy.get("target_concurrency", 1))

        kv_dtypes = ["auto"]
        if engine.supports_feature("fp8_kv_cache"):
            kv_dtypes = [
                dtype
                for dtype in space.get("kv_cache_dtypes", ["auto"])
                if dtype == "auto" or engine.supports_feature("fp8_kv_cache")
            ]

        speculative_options = [False]
        if (
            model.supports_speculative_decoding
            and engine.supports_feature("speculative_decoding")
            and policy.get("prefer_speculative_decoding", True)
        ):
            # Only worth trying when a stream is latency-bound. Under heavy
            # batching the spare compute it needs is already committed.
            speculative_options = [True, False] if concurrency <= 4 else [False]

        utilizations = space.get("gpu_memory_utilization", [0.90])

        combinations = []
        for kv_dtype in kv_dtypes:
            for speculative in speculative_options:
                for utilization in utilizations:
                    combinations.append(
                        {
                            "kv_cache_dtype": kv_dtype,
                            "speculative_decoding": speculative,
                            "gpu_memory_utilization": utilization,
                            "max_num_seqs": max(concurrency, 1),
                            "enable_prefix_caching": bool(
                                space.get("enable_prefix_caching", True)
                            )
                            and engine.supports_feature("prefix_caching"),
                            "num_speculative_tokens": int(
                                space.get("speculative_num_tokens", 3)
                            ),
                            "concurrency": max(concurrency, 1),
                        }
                    )
        return combinations

    def _build_candidate(
        self,
        model: ModelSpec,
        engine: EngineSpec,
        engine_rank: int,
        intent: Intent,
        hardware: HardwareProfile,
        runtime: str,
        knobs: Dict[str, Any],
    ) -> Optional[CandidatePlan]:
        context_length = min(intent.context_length, model.context_length)
        concurrency = int(knobs["concurrency"])
        speculative = bool(knobs["speculative_decoding"])

        estimate = self.memory.estimate(
            model,
            hardware,
            context_length=context_length,
            concurrency=concurrency,
            kv_cache_dtype=knobs["kv_cache_dtype"],
            speculative_decoding=speculative,
        )
        if not estimate.fits:
            return None

        parts = [model.model_id, engine.engine_id, model.precision.lower()]
        if knobs["kv_cache_dtype"] != "auto":
            parts.append(f"kv{knobs['kv_cache_dtype']}")
        if speculative:
            parts.append("spec")
        if concurrency > 1:
            parts.append(f"b{concurrency}")
        candidate_id = "-".join(parts)

        reason_bits = [
            f"{engine.display_name} ranks #{engine_rank + 1} for "
            f"{intent.priority}"
        ]
        if model.is_mixture_of_experts:
            reason_bits.append(
                f"{model.active_parameter_count_b:g}B of "
                f"{model.parameter_count_b:g}B parameters are read per token"
            )
        if speculative:
            reason_bits.append("speculative decoding trades compute for bandwidth")
        if knobs["kv_cache_dtype"] == "fp8":
            reason_bits.append("FP8 KV cache halves cache cost")
        reason_bits.append(estimate.detail)

        runtime_config = {
            "max_num_seqs": knobs["max_num_seqs"],
            "gpu_memory_utilization": knobs["gpu_memory_utilization"],
            "kv_cache_dtype": knobs["kv_cache_dtype"],
            "enable_prefix_caching": knobs["enable_prefix_caching"],
        }
        if speculative:
            runtime_config["speculative"] = {
                "draft_source_id": model.draft_source_id,
                "num_speculative_tokens": knobs["num_speculative_tokens"],
            }

        confidence = max(
            0.45,
            0.90
            - engine_rank * 0.07
            - (0.08 if speculative else 0.0)
            - (0.05 if model.validation != "verified_on_target" else 0.0),
        )

        return CandidatePlan(
            candidate_id=candidate_id,
            model_id=model.model_id,
            source_id=model.source_id,
            engine=engine.engine_id,
            device="CUDA" if "CUDA" in model.devices else "CPU",
            precision=model.precision,
            context_length=context_length,
            runtime=runtime,
            expected_memory_gb=estimate.total_gb,
            quality_score=model.quality_score,
            reason="; ".join(reason_bits),
            confidence=round(confidence, 2),
            simulated=hardware.simulated,
            concurrency=concurrency,
            kv_cache_dtype=knobs["kv_cache_dtype"],
            active_parameter_count_b=model.active_parameter_count_b,
            parameter_count_b=model.parameter_count_b,
            memory_estimate=estimate,
            model_path=os.environ.get("LOCALPILOT_MODEL_PATH") or None,
            runtime_config=runtime_config,
            knobs=dict(knobs),
        )

    def _candidate_prior(
        self,
        model: ModelSpec,
        candidate: CandidatePlan,
        hardware: HardwareProfile,
        intent: Intent,
        engine_rank: int,
    ) -> float:
        roofline = decode_roofline_tokens_s(
            model.active_parameter_count_b,
            model.precision,
            hardware.memory_bandwidth_gbps,
            efficiency=self.policies.bandwidth_efficiency,
            memory_model=self.memory,
        )
        speed = roofline or 40.0
        if candidate.runtime_config.get("speculative"):
            speed *= self.policies.speculative_gain
        speed_term = min(1.0, speed / 200.0)
        engine_term = max(0.0, 1.0 - engine_rank * 0.12)
        weights = self.policies.priority(intent.priority).get("weights", {})
        quality_weight = float(weights.get("quality", 0.25))
        speed_weight = float(weights.get("latency", 0.2)) + float(
            weights.get("throughput", 0.2)
        )
        return (
            model.quality_score * quality_weight
            + speed_term * speed_weight
            + engine_term * 0.15
            + candidate.confidence * 0.10
        )

    def _no_model_message(
        self, intent: Intent, selection: SelectionOutcome
    ) -> str:
        if not selection.rejected:
            return f"No model in the registry serves task '{intent.task}'"
        lines = [
            f"No model can serve task '{intent.task}' on this machine:",
        ]
        for rejection in selection.rejected[:6]:
            lines.append(
                f"  - {rejection.model_id} [{rejection.gate}] {rejection.reason}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # bounded recovery

    def recovery_plan(
        self,
        failed_candidates: List[CandidatePlan],
        hardware: HardwareProfile,
        models: List[ModelSpec],
    ) -> List[CandidatePlan]:
        if not failed_candidates:
            return []

        recovery = self.policies.recovery
        maximum = max(0, int(recovery.get("max_attempts", 2)))
        minimum_context = max(
            1024, int(recovery.get("minimum_context_length", 4096))
        )
        primary = failed_candidates[0]
        model = next(
            (item for item in models if item.model_id == primary.model_id), None
        )
        if model is None:
            return []

        seen = {item.candidate_id for item in failed_candidates}
        attempts: List[CandidatePlan] = []

        # Step one: the same configuration, smaller. Most first failures on a
        # shared memory pool are the cache, not the weights.
        reduced_context = max(minimum_context, primary.context_length // 4)
        if reduced_context < primary.context_length:
            estimate = self.memory.estimate(
                model,
                hardware,
                context_length=reduced_context,
                concurrency=1,
                kv_cache_dtype="fp8",
            )
            candidate_id = f"{primary.candidate_id}-ctx{reduced_context}"
            if candidate_id not in seen:
                attempts.append(
                    self._recovery_candidate(
                        primary,
                        model,
                        candidate_id=candidate_id,
                        context_length=reduced_context,
                        concurrency=1,
                        kv_cache_dtype="fp8",
                        estimate=estimate,
                        action="reduce_context_and_quantize_kv",
                        reason=(
                            f"Recovery after {primary.candidate_id}: context "
                            f"{primary.context_length} to {reduced_context} "
                            "with an FP8 KV cache"
                        ),
                        engine=primary.engine,
                    )
                )

        # Step two: the smallest viable model on the fallback engine.
        fallback_engine = str(recovery.get("fallback_engine", "vllm"))
        smallest = min(
            (
                item
                for item in models
                if fallback_engine in item.engines and item.model_id != model.model_id
            ),
            key=lambda item: self.memory.weights_gb(item),
            default=None,
        )
        if smallest is not None:
            estimate = self.memory.estimate(
                smallest,
                hardware,
                context_length=minimum_context,
                concurrency=1,
            )
            candidate_id = f"{smallest.model_id}-{fallback_engine}-recovery"
            if estimate.fits and candidate_id not in seen:
                attempts.append(
                    self._recovery_candidate(
                        primary,
                        smallest,
                        candidate_id=candidate_id,
                        context_length=minimum_context,
                        concurrency=1,
                        kv_cache_dtype="auto",
                        estimate=estimate,
                        action="fallback_to_smallest_model",
                        reason=(
                            f"Recovery after {primary.candidate_id}: smallest "
                            f"registered model on {fallback_engine}"
                        ),
                        engine=fallback_engine,
                    )
                )

        return attempts[:maximum]

    def _recovery_candidate(
        self,
        primary: CandidatePlan,
        model: ModelSpec,
        candidate_id: str,
        context_length: int,
        concurrency: int,
        kv_cache_dtype: str,
        estimate,
        action: str,
        reason: str,
        engine: str,
    ) -> CandidatePlan:
        return CandidatePlan(
            candidate_id=candidate_id,
            model_id=model.model_id,
            source_id=model.source_id,
            engine=engine,
            device=primary.device,
            precision=model.precision,
            context_length=context_length,
            runtime=primary.runtime,
            expected_memory_gb=estimate.total_gb,
            quality_score=model.quality_score,
            reason=reason,
            confidence=max(0.35, primary.confidence - 0.15),
            simulated=primary.simulated,
            concurrency=concurrency,
            kv_cache_dtype=kv_cache_dtype,
            active_parameter_count_b=model.active_parameter_count_b,
            parameter_count_b=model.parameter_count_b,
            memory_estimate=estimate,
            model_path=primary.model_path,
            runtime_config={
                "max_num_seqs": concurrency,
                "gpu_memory_utilization": 0.85,
                "kv_cache_dtype": kv_cache_dtype,
                "enable_prefix_caching": False,
            },
            knobs={
                "kv_cache_dtype": kv_cache_dtype,
                "speculative_decoding": False,
                "concurrency": concurrency,
                "max_num_seqs": concurrency,
            },
            fallback_of=primary.candidate_id,
            recovery_action=action,
        )
