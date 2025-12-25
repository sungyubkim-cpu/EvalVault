# Ragas Evaluation Service Usage

This guide demonstrates how to use the Ragas evaluation service in EvalVault.

## Quick Start

### 1. Setup Configuration

Create a `.env` file in your project root:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Custom endpoint
OPENAI_MODEL=gpt-5-nano                     # Default model

# Metric Thresholds (0.0 - 1.0)
THRESHOLD_FAITHFULNESS=0.7
THRESHOLD_ANSWER_RELEVANCY=0.7
THRESHOLD_CONTEXT_PRECISION=0.7
THRESHOLD_CONTEXT_RECALL=0.7
```

### 2. Basic Usage

```python
import asyncio
from evalvault.domain.entities import Dataset, TestCase
from evalvault.domain.services import RagasEvaluator
from evalvault.adapters.outbound.llm import OpenAIAdapter
from evalvault.config import Settings

async def main():
    # Initialize settings
    settings = Settings()

    # Create LLM adapter
    llm = OpenAIAdapter(settings)

    # Prepare your dataset
    dataset = Dataset(
        name="my-rag-evaluation",
        version="1.0.0",
        test_cases=[
            TestCase(
                id="tc-001",
                question="What is the capital of France?",
                answer="The capital of France is Paris.",
                contexts=["Paris is the capital and largest city of France."],
                ground_truth="Paris"
            ),
            TestCase(
                id="tc-002",
                question="What is Python?",
                answer="Python is a high-level programming language.",
                contexts=[
                    "Python is a high-level, interpreted programming language.",
                    "It was created by Guido van Rossum in 1991."
                ],
                ground_truth="A programming language"
            ),
        ]
    )

    # Create evaluator
    evaluator = RagasEvaluator()

    # Run evaluation
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    thresholds = settings.get_all_thresholds()

    result = await evaluator.evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        thresholds=thresholds
    )

    # Display results
    print(f"Evaluation Run ID: {result.run_id}")
    print(f"Model: {result.model_name}")
    print(f"Total Test Cases: {result.total_test_cases}")
    print(f"Passed: {result.passed_test_cases}")
    print(f"Pass Rate: {result.pass_rate:.2%}")
    print(f"Duration: {result.duration_seconds:.2f}s")

    # Print per-metric averages
    for metric in metrics:
        avg_score = result.get_avg_score(metric)
        print(f"Average {metric}: {avg_score:.3f}")

    # Print detailed results
    for test_result in result.results:
        print(f"\nTest Case: {test_result.test_case_id}")
        print(f"  Passed: {test_result.all_passed}")
        for metric_score in test_result.metrics:
            status = "✓" if metric_score.passed else "✗"
            print(f"  {status} {metric_score.name}: {metric_score.score:.3f} (threshold: {metric_score.threshold})")

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Metrics

EvalVault supports the following Ragas metrics:

### 1. Faithfulness
Measures how factually accurate the generated answer is compared to the retrieved contexts.
- **Range**: 0.0 - 1.0 (higher is better)
- **Use case**: Detect hallucinations

### 2. Answer Relevancy
Measures how relevant the generated answer is to the user's question.
- **Range**: 0.0 - 1.0 (higher is better)
- **Use case**: Ensure answers stay on topic

### 3. Context Precision
Measures whether the relevant contexts are ranked higher than irrelevant ones.
- **Range**: 0.0 - 1.0 (higher is better)
- **Use case**: Evaluate retrieval quality
- **Requires**: ground_truth in TestCase

### 4. Context Recall
Measures whether all relevant information from the ground truth is present in the retrieved contexts.
- **Range**: 0.0 - 1.0 (higher is better)
- **Use case**: Ensure comprehensive retrieval
- **Requires**: ground_truth in TestCase

## Data Format

### TestCase Structure

```python
TestCase(
    id="unique-id",
    question="User's question",           # Required
    answer="Generated answer",            # Required
    contexts=["context1", "context2"],    # Required
    ground_truth="Expected answer",       # Optional (required for precision/recall)
    metadata={"key": "value"}             # Optional
)
```

### Loading from Files

```python
from evalvault.adapters.outbound.dataset import LoaderFactory

# CSV file
loader = LoaderFactory.get_loader("dataset.csv")
dataset = loader.load("dataset.csv")

# Excel file
loader = LoaderFactory.get_loader("dataset.xlsx")
dataset = loader.load("dataset.xlsx")

# JSON file
loader = LoaderFactory.get_loader("dataset.json")
dataset = loader.load("dataset.json")
```

## Custom Thresholds

You can customize thresholds per evaluation:

```python
custom_thresholds = {
    "faithfulness": 0.8,      # Stricter
    "answer_relevancy": 0.6,  # More lenient
    "context_precision": 0.75,
    "context_recall": 0.7
}

result = await evaluator.evaluate(
    dataset=dataset,
    metrics=["faithfulness", "answer_relevancy"],
    llm=llm,
    thresholds=custom_thresholds
)
```

## Advanced: Custom LLM Provider

To use a custom LLM provider, implement the `LLMPort` interface:

```python
from evalvault.ports.outbound.llm_port import LLMPort
from langchain_openai import ChatOpenAI

class CustomLLMAdapter(LLMPort):
    def get_model_name(self) -> str:
        return "custom-model-name"

    def as_ragas_llm(self):
        # Return LangChain-compatible LLM instance
        return ChatOpenAI(
            api_key="your-key",
            base_url="https://custom-endpoint.com/v1",
            model="custom-model"
        )
```

## Integration with Langfuse

Results can be logged to Langfuse for tracking and visualization:

```python
from evalvault.adapters.outbound.tracker import LangfuseAdapter

# Initialize Langfuse
tracker = LangfuseAdapter(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Start trace
trace_id = tracker.start_trace(
    name="rag-evaluation",
    metadata={"dataset": dataset.name}
)

# Run evaluation
result = await evaluator.evaluate(...)

# Log results to Langfuse
tracker.log_evaluation_run(result, trace_id)
tracker.end_trace(trace_id)
```

## Best Practices

1. **Use appropriate metrics**: Choose metrics based on your use case
   - Hallucination detection: `faithfulness`
   - Response quality: `answer_relevancy`
   - Retrieval quality: `context_precision`, `context_recall`

2. **Set realistic thresholds**: Start with 0.7 and adjust based on your domain

3. **Batch evaluations**: For large datasets, consider running evaluations in batches to manage API costs

4. **Version your datasets**: Use dataset versioning to track improvements over time

5. **Monitor costs**: Ragas evaluation uses LLM calls, which can add up. Use smaller models (gpt-5-nano) for cost efficiency.

## Troubleshooting

### API Key Errors
Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

### Import Errors
Make sure you've installed all dependencies:
```bash
pip install -e ".[dev]"
```

### Slow Evaluation
Ragas makes multiple LLM calls per test case. Consider:
- Using faster models (gpt-5-nano for cost efficiency)
- Evaluating fewer metrics
- Reducing dataset size for quick iterations
