#!/usr/bin/env python3
"""Full evaluation test script.

Tests:
1. Ragas evaluation with multiple metrics
2. Langfuse integration (trace logging)
3. SQLite local storage

Usage:
    uv run python scripts/test_full_evaluation.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    """Run full evaluation test."""
    print("=" * 70)
    print("EvalVault Full Evaluation Test")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Import after path setup
    from evalvault.adapters.outbound.dataset import JSONDatasetLoader
    from evalvault.adapters.outbound.llm.openai_adapter import OpenAIAdapter
    from evalvault.adapters.outbound.storage.sqlite_adapter import SQLiteStorageAdapter
    from evalvault.adapters.outbound.tracker.langfuse_adapter import LangfuseAdapter
    from evalvault.config.settings import Settings
    from evalvault.domain.services.evaluator import RagasEvaluator

    # Configuration
    dataset_path = Path(__file__).parent.parent / "tests/fixtures/e2e/evaluation_test_sample.json"
    db_path = Path(__file__).parent.parent / "data/evaluations.db"

    # Metrics to test (all 6 metrics)
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "factual_correctness",
        "semantic_similarity",
    ]

    print("\n[1/4] Loading Dataset...")
    print("-" * 50)

    loader = JSONDatasetLoader()
    dataset = loader.load(str(dataset_path))

    print(f"  Dataset: {dataset.name}")
    print(f"  Version: {dataset.version}")
    print(f"  Test Cases: {len(dataset.test_cases)}")
    for tc in dataset.test_cases:
        print(f"    - {tc.id}: {tc.question[:40]}...")

    print("\n[2/4] Running Ragas Evaluation...")
    print("-" * 50)
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Model: {os.environ.get('OPENAI_MODEL', 'unknown')}")
    print()

    # Initialize components
    settings = Settings()
    llm_adapter = OpenAIAdapter(settings)
    evaluator = RagasEvaluator()

    # Run evaluation
    start_time = datetime.now()
    run = await evaluator.evaluate(dataset, metrics, llm_adapter)
    end_time = datetime.now()

    print(f"\n  Evaluation completed in {(end_time - start_time).total_seconds():.1f}s")
    print(f"  Run ID: {run.run_id}")
    print(f"  Dataset: {run.dataset_name} v{run.dataset_version}")
    print(f"  Model: {run.model_name}")
    print(f"  Pass Rate: {run.pass_rate * 100:.1f}%")
    print(f"  Total Tokens: {run.total_tokens}")

    print("\n  Metric Scores (Aggregated):")
    for metric_name in run.metrics_evaluated:
        avg_score = run.get_avg_score(metric_name)
        if avg_score is not None:
            status = "PASS" if avg_score >= 0.7 else "FAIL"
            print(f"    {metric_name}: {avg_score:.4f} [{status}]")

    print("\n  Per-TestCase Results:")
    for result in run.results:
        print(f"    {result.test_case_id}:")
        for metric in result.metrics:
            print(
                f"      {metric.name}: {metric.score:.4f} [{'PASS' if metric.passed else 'FAIL'}]"
            )

    print("\n[3/4] Sending to Langfuse...")
    print("-" * 50)

    # Check if Langfuse is configured
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    langfuse_configured = bool(langfuse_public_key and langfuse_secret_key)

    if langfuse_configured:
        try:
            tracker = LangfuseAdapter(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host,
            )
            trace_id = tracker.log_evaluation_run(run)
            print(f"  Trace ID: {trace_id}")
            print(f"  Langfuse Host: {langfuse_host}")
            print("  Status: SUCCESS - Check Langfuse dashboard for details")

            # Flush to ensure data is sent
            tracker.flush()
        except Exception as e:
            print(f"  Status: ERROR - {e}")
    else:
        print("  Status: SKIPPED - Langfuse not configured")
        print("  Missing: LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY")

    print("\n[4/4] Saving to Local SQLite DB...")
    print("-" * 50)

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = SQLiteStorageAdapter(str(db_path))
        storage.save_run(run)

        # Verify by retrieving
        retrieved = storage.get_run(run.run_id)

        print(f"  DB Path: {db_path}")
        print(f"  Run ID: {retrieved.run_id}")
        print("  Status: SUCCESS - Data saved and verified")

        # List recent runs
        recent_runs = storage.list_runs(limit=5)
        print(f"\n  Recent Runs in DB ({len(recent_runs)} shown):")
        for r in recent_runs:
            print(
                f"    - {r.run_id[:8]}... | {r.dataset_name} | {r.pass_rate*100:.0f}% | {r.started_at}"
            )

    except Exception as e:
        print(f"  Status: ERROR - {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    # Return summary
    aggregated = {m: run.get_avg_score(m) for m in run.metrics_evaluated}
    return {
        "run_id": run.run_id,
        "pass_rate": run.pass_rate,
        "metrics": aggregated,
        "langfuse_configured": langfuse_configured,
        "db_saved": True,
    }


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nResult Summary: {json.dumps(result, indent=2, default=str)}")
