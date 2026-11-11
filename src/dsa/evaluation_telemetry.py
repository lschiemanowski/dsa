"""Mirror completed benchmark scores to Logfire without running the agent again."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from dsa.evaluation import EvaluationPrediction, score_prediction
from dsa.pack import LoadedEvaluationPack
from dsa.reporting import (
    LogfireConfigurationMissing,
    LogfireReporting,
    configure_logfire,
    flush_analysis_telemetry,
)

if TYPE_CHECKING:
    from pydantic_evals.evaluators import EvaluatorContext


def record_experiment(
    pack: LoadedEvaluationPack,
    predictions: tuple[EvaluationPrediction, ...],
    *,
    pack_id: str,
    model_id: str,
    metadata: dict[str, str],
) -> LogfireReporting:
    """Publish a bounded projection from the isolated cell worker."""
    try:
        sdk = configure_logfire(local=False)
        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import Evaluator

        by_id = {prediction.case_id: prediction for prediction in predictions}
        scores: dict[str, dict[str, bool]] = {}
        for case in pack.cases:
            prediction = by_id[case.case_id]
            summary = score_prediction(prediction, case.expected_answer, pack.manifest.scorer)
            scores[case.case_id] = {
                "completion": prediction.accepted,
                **{name: value for name, value in summary.items() if isinstance(value, bool)},
            }

        def retained_case(case_id: str) -> dict[str, str]:
            prediction = by_id[case_id]
            output = {"run_id": prediction.run_id}
            if prediction.reporting.trace_id:
                output["analysis_trace_id"] = prediction.reporting.trace_id
            return output

        class RetainedScores(Evaluator[str, dict[str, str], None]):
            def evaluate(self, ctx: EvaluatorContext[str, dict[str, str], None]) -> dict[str, bool]:
                return scores[ctx.inputs]

        dataset = Dataset[str, dict[str, str], None](
            name=f"{pack_id}-{pack.reference.manifest_sha256}",
            cases=[Case(name=case.case_id, inputs=case.case_id) for case in pack.cases],
            evaluators=[RetainedScores()],
        )
        report = asyncio.run(dataset.evaluate(
            retained_case, name=model_id, progress=False, max_concurrency=1, metadata=metadata
        ))
        if report.failures or len(report.cases) != len(predictions) or not report.trace_id:
            raise ValueError("incomplete experiment projection")
        experiment_flushed = sdk.force_flush()
        analysis_flushed = flush_analysis_telemetry()
        if not (experiment_flushed and analysis_flushed):
            return LogfireReporting(
                status="failed", trace_id=report.trace_id, failure_code="logfire_flush_failed"
            )
        return LogfireReporting(status="recorded", trace_id=report.trace_id)
    except LogfireConfigurationMissing:
        return LogfireReporting(status="failed", failure_code="logfire_token_missing")
    except ModuleNotFoundError:
        return LogfireReporting(status="failed", failure_code="logfire_dependency_missing")
    except Exception:
        return LogfireReporting(status="failed", failure_code="logfire_experiment_failed")
