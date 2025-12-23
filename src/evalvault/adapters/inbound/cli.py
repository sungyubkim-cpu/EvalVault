"""CLI interface for EvalVault using Typer."""

import asyncio
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evalvault.adapters.outbound.dataset import get_loader
from evalvault.adapters.outbound.llm.openai_adapter import OpenAIAdapter
from evalvault.adapters.outbound.tracker.langfuse_adapter import LangfuseAdapter
from evalvault.config.settings import Settings
from evalvault.domain.services.evaluator import RagasEvaluator

app = typer.Typer(
    name="evalvault",
    help="RAG evaluation system using Ragas with Langfuse tracing.",
    add_completion=False,
)
console = Console()

# Available metrics
AVAILABLE_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        rprint("[bold]EvalVault[/bold] version [cyan]0.1.0[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """EvalVault - RAG evaluation system."""
    pass


@app.command()
def run(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset file (CSV, Excel, or JSON).",
        exists=True,
        readable=True,
    ),
    metrics: str = typer.Option(
        "faithfulness,answer_relevancy",
        "--metrics",
        "-m",
        help="Comma-separated list of metrics to evaluate.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model to use for evaluation (overrides settings).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format).",
    ),
    langfuse: bool = typer.Option(
        False,
        "--langfuse",
        "-l",
        help="Log results to Langfuse.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed output.",
    ),
):
    """Run RAG evaluation on a dataset."""
    # Parse metrics
    metric_list = [m.strip() for m in metrics.split(",")]

    # Validate metrics
    invalid_metrics = [m for m in metric_list if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        console.print(
            f"[red]Error:[/red] Invalid metrics: {', '.join(invalid_metrics)}"
        )
        console.print(f"Available metrics: {', '.join(AVAILABLE_METRICS)}")
        raise typer.Exit(1)

    # Load settings
    settings = Settings()
    if model:
        settings.openai_model = model

    # Check for API key
    if not settings.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set.")
        console.print("Set it in your environment or .env file.")
        raise typer.Exit(1)

    console.print("\n[bold]EvalVault[/bold] - RAG Evaluation")
    console.print(f"Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"Metrics: [cyan]{', '.join(metric_list)}[/cyan]")
    console.print(f"Model: [cyan]{settings.openai_model}[/cyan]\n")

    # Load dataset
    with console.status("[bold green]Loading dataset..."):
        try:
            loader = get_loader(dataset)
            ds = loader.load(dataset)
            console.print(f"[green]Loaded {len(ds)} test cases[/green]")
        except Exception as e:
            console.print(f"[red]Error loading dataset:[/red] {e}")
            raise typer.Exit(1)

    # Initialize components
    llm = OpenAIAdapter(settings)
    evaluator = RagasEvaluator()
    thresholds = settings.get_all_thresholds()

    # Run evaluation
    with console.status("[bold green]Running evaluation..."):
        try:
            result = asyncio.run(
                evaluator.evaluate(
                    dataset=ds,
                    metrics=metric_list,
                    llm=llm,
                    thresholds=thresholds,
                )
            )
        except Exception as e:
            console.print(f"[red]Error during evaluation:[/red] {e}")
            raise typer.Exit(1)

    # Display results
    _display_results(result, verbose)

    # Log to Langfuse if requested
    if langfuse:
        _log_to_langfuse(settings, result)

    # Save to file if requested
    if output:
        _save_results(output, result)


def _display_results(result, verbose: bool = False):
    """Display evaluation results in a formatted table."""
    # Calculate duration safely
    duration = result.duration_seconds
    duration_str = f"{duration:.2f}s" if duration is not None else "N/A"

    # Summary panel
    summary = f"""
[bold]Evaluation Summary[/bold]
  Run ID: {result.run_id}
  Dataset: {result.dataset_name} v{result.dataset_version}
  Model: {result.model_name}
  Duration: {duration_str}

[bold]Results[/bold]
  Total Test Cases: {result.total_test_cases}
  Passed: [green]{result.passed_test_cases}[/green]
  Failed: [red]{result.total_test_cases - result.passed_test_cases}[/red]
  Pass Rate: {'[green]' if result.pass_rate >= 0.7 else '[red]'}{result.pass_rate:.1%}[/]
"""
    console.print(Panel(summary, title="Evaluation Results", border_style="blue"))

    # Metrics table
    table = Table(title="Metric Scores", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Average Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for metric in result.metrics_evaluated:
        avg_score = result.get_avg_score(metric)
        threshold = result.thresholds.get(metric, 0.7)
        passed = avg_score >= threshold

        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        score_color = "green" if passed else "red"

        table.add_row(
            metric,
            f"[{score_color}]{avg_score:.3f}[/{score_color}]",
            f"{threshold:.2f}",
            status,
        )

    console.print(table)

    # Detailed results if verbose
    if verbose:
        console.print("\n[bold]Detailed Results[/bold]\n")
        for tc_result in result.results:
            status = "[green]PASS[/green]" if tc_result.all_passed else "[red]FAIL[/red]"
            console.print(f"  {tc_result.test_case_id}: {status}")
            for metric in tc_result.metrics:
                m_status = "[green]+[/green]" if metric.passed else "[red]-[/red]"
                console.print(
                    f"    {m_status} {metric.name}: {metric.score:.3f} (threshold: {metric.threshold})"
                )


def _log_to_langfuse(settings: Settings, result):
    """Log results to Langfuse."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        console.print(
            "[yellow]Warning:[/yellow] Langfuse credentials not configured. "
            "Skipping Langfuse logging."
        )
        return

    with console.status("[bold green]Logging to Langfuse..."):
        try:
            tracker = LangfuseAdapter(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            trace_id = tracker.log_evaluation_run(result)
            console.print(f"[green]Logged to Langfuse[/green] (trace_id: {trace_id})")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to log to Langfuse: {e}")


def _save_results(output: Path, result):
    """Save results to JSON file."""
    import json

    with console.status(f"[bold green]Saving to {output}..."):
        try:
            data = result.to_summary_dict()
            data["results"] = [
                {
                    "test_case_id": r.test_case_id,
                    "all_passed": r.all_passed,
                    "metrics": [
                        {
                            "name": m.name,
                            "score": m.score,
                            "threshold": m.threshold,
                            "passed": m.passed,
                        }
                        for m in r.metrics
                    ],
                }
                for r in result.results
            ]

            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            console.print(f"[green]Results saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving results:[/red] {e}")


@app.command()
def metrics():
    """List available evaluation metrics."""
    console.print("\n[bold]Available Metrics[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Description")
    table.add_column("Requires Ground Truth", justify="center")

    table.add_row(
        "faithfulness",
        "Measures factual accuracy of the answer based on contexts",
        "[red]No[/red]",
    )
    table.add_row(
        "answer_relevancy",
        "Measures how relevant the answer is to the question",
        "[red]No[/red]",
    )
    table.add_row(
        "context_precision",
        "Measures ranking quality of retrieved contexts",
        "[green]Yes[/green]",
    )
    table.add_row(
        "context_recall",
        "Measures if all relevant info is in retrieved contexts",
        "[green]Yes[/green]",
    )

    console.print(table)
    console.print(
        "\n[dim]Use --metrics flag with 'run' command to specify metrics.[/dim]"
    )
    console.print("[dim]Example: evalvault run data.csv --metrics faithfulness,answer_relevancy[/dim]\n")


@app.command()
def config():
    """Show current configuration."""
    settings = Settings()

    console.print("\n[bold]Current Configuration[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    # OpenAI settings
    api_key_status = "[green]Set[/green]" if settings.openai_api_key else "[red]Not set[/red]"
    table.add_row("OpenAI API Key", api_key_status)
    table.add_row("OpenAI Model", settings.openai_model)
    table.add_row("OpenAI Base URL", settings.openai_base_url or "[dim]Default[/dim]")

    # Langfuse settings
    langfuse_status = (
        "[green]Configured[/green]"
        if settings.langfuse_public_key and settings.langfuse_secret_key
        else "[yellow]Not configured[/yellow]"
    )
    table.add_row("Langfuse", langfuse_status)
    table.add_row("Langfuse Host", settings.langfuse_host)

    console.print(table)

    # Thresholds
    console.print("\n[bold]Metric Thresholds[/bold]\n")

    threshold_table = Table(show_header=True, header_style="bold cyan")
    threshold_table.add_column("Metric", style="bold")
    threshold_table.add_column("Threshold", justify="right")

    thresholds = settings.get_all_thresholds()
    for metric, threshold in thresholds.items():
        threshold_table.add_row(metric, f"{threshold:.2f}")

    console.print(threshold_table)
    console.print(
        "\n[dim]Set thresholds via environment variables:[/dim]"
    )
    console.print("[dim]  THRESHOLD_FAITHFULNESS=0.8[/dim]")
    console.print("[dim]  THRESHOLD_ANSWER_RELEVANCY=0.7[/dim]\n")


if __name__ == "__main__":
    app()
