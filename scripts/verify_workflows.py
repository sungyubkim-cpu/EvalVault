"""Helper script to orchestrate integration and end-to-end checks."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run shell command and stream output."""
    print(f"\n[verify] Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Langfuse integration and KG workflows.")
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip pytest integration checks.",
    )
    parser.add_argument(
        "--kg-source",
        type=Path,
        default=Path("docs/USER_GUIDE.md"),
        help="Document or directory for evalvault kg stats.",
    )
    parser.add_argument(
        "--kg-report",
        type=Path,
        default=Path("reports/kg_stats_report.json"),
        help="Output path for KG stats report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/e2e/insurance_qa_korean.json"),
        help="Dataset for evalvault run.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness",
        help="Metrics for evalvault run.",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Enable Langfuse logging for CLI commands.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="After commands, fetch Langfuse traces via langfuse-dashboard.",
    )
    parser.add_argument(
        "--dashboard-limit",
        type=int,
        default=5,
        help="Trace count for langfuse-dashboard.",
    )
    parser.add_argument(
        "--dashboard-event",
        type=str,
        default="ragas_evaluation",
        help="event_type filter for langfuse-dashboard.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_integration:
        run_command(["pytest", "tests/integration", "-k", "langfuse", "-v"])

    kg_cmd = [
        "evalvault",
        "kg",
        "stats",
        str(args.kg_source),
        "--report-file",
        str(args.kg_report),
    ]
    if not args.langfuse:
        kg_cmd.append("--no-langfuse")
    run_command(kg_cmd)

    run_cmd = [
        "evalvault",
        "run",
        str(args.dataset),
        "--metrics",
        args.metrics,
    ]
    if args.langfuse:
        run_cmd.append("--langfuse")
    run_command(run_cmd)

    if args.dashboard:
        dash_cmd = [
            "evalvault",
            "langfuse-dashboard",
            "--limit",
            str(args.dashboard_limit),
            "--event-type",
            args.dashboard_event,
        ]
        run_command(dash_cmd)

    print("\n[verify] All requested steps completed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
