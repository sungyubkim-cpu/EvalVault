📚 Core Concepts
 Experimentation

Learn how to systematically evaluate your AI applications using experiments.

Track changes, measure improvements, and compare results across different versions of your application.

 Datasets

Understand how to create, manage, and use evaluation datasets.

Learn about dataset structure, storage backends, and best practices for maintaining your test data.

: Ragas Metrics

Use our library of available metrics or create custom metrics tailored to your use case.

Metrics for evaluating RAG, Agentic workflows and more...

 Test Data Generation

Generate high-quality datasets for comprehensive testing.

Algorithms for synthesizing data to test RAG, Agentic workflows

----
Experiments
What is an experiment?
An experiment is a deliberate change made to your application to test a hypothesis or idea. For example, in a Retrieval-Augmented Generation (RAG) system, you might replace the retriever model to evaluate how a new embedding model impacts chatbot responses.

Principles of a Good Experiment
Define measurable metrics: Use metrics like accuracy, precision, or recall to quantify the impact of your changes.
Systematic result storage: Ensure results are stored in an organized manner for easy comparison and tracking.
Isolate changes: Make one change at a time to identify its specific impact. Avoid making multiple changes simultaneously, as this can obscure the results.
Iterative process: Follow a structured approach: *Make a change → Run evaluations → Observe results →
Make a change

Run evaluations

Observe results

Hypothesize next change

Experiments in Ragas
Components of an Experiment
Test dataset: The data used to evaluate the system.
Application endpoint: The application, component or model being tested.
Metrics: Quantitative measures to assess performance.
Execution Process
Setup: Define the experiment parameters and load the test dataset.
Run: Execute the application on each sample in the dataset.
Evaluate: Apply metrics to measure performance.
Store: Save results for analysis and comparison.
Creating Experiments with Ragas
Ragas provides an @experiment decorator to streamline the experiment creation process. If you prefer a hands-on intro first, see the Quick Start guide.

Basic Experiment Structure

from ragas import experiment
import asyncio

@experiment()
async def my_experiment(row):
    # Process the input through your system
    response = await asyncio.to_thread(my_system_function, row["input"])

    # Return results for evaluation
    return {
        **row,  # Include original data
        "response": response,
        "experiment_name": "baseline_v1",
        # Add any additional metadata
        "model_version": "gpt-4o",
        "timestamp": datetime.now().isoformat()
    }
Running Experiments

from ragas import Dataset

# Load your test dataset
dataset = Dataset.load(name="test_data", backend="local/csv", root_dir="./data")

# Run the experiment
results = await my_experiment.arun(dataset)
Parameterized Experiments
You can create parameterized experiments to test different configurations:


@experiment()
async def model_comparison_experiment(row, model_name: str, temperature: float):
    # Configure your system with the parameters
    response = await my_system_function(
        row["input"], 
        model=model_name, 
        temperature=temperature
    )

    return {
        **row,
        "response": response,
        "experiment_name": f"{model_name}_temp_{temperature}",
        "model_name": model_name,
        "temperature": temperature
    }

# Run with different parameters
results_gpt4 = await model_comparison_experiment.arun(
    dataset, 
    model_name="gpt-4o", 
    temperature=0.1
)

results_gpt35 = await model_comparison_experiment.arun(
    dataset, 
    model_name="gpt-3.5-turbo", 
    temperature=0.1
)
Experiment Management Best Practices
1. Consistent Naming
Use descriptive names that include: - What changed (model, prompt, parameters) - Version numbers - Date/time if relevant


experiment_name = "gpt4o_v2_prompt_temperature_0.1_20241201"
2. Result Storage
Experiments automatically save results to CSV files in the experiments/ directory with timestamps:


experiments/
├── 20241201-143022-baseline_v1.csv
├── 20241201-143515-gpt4o_improved_prompt.csv
└── 20241201-144001-comparison.csv
3. Metadata Tracking
Include relevant metadata in your experiment results:


return {
    **row,
    "response": response,
    "experiment_name": "baseline_v1",
    "git_commit": "a1b2c3d",
    "environment": "staging",
    "model_version": "gpt-4o-2024-08-06",
    "total_tokens": response.usage.total_tokens,
    "response_time_ms": response_time
}
Advanced Experiment Patterns
A/B Testing
Test two different approaches simultaneously:


@experiment()
async def ab_test_experiment(row, variant: str):
    if variant == "A":
        response = await system_variant_a(row["input"])
    else:
        response = await system_variant_b(row["input"])

    return {
        **row,
        "response": response,
        "variant": variant,
        "experiment_name": f"ab_test_variant_{variant}"
    }

# Run both variants
results_a = await ab_test_experiment.arun(dataset, variant="A")
results_b = await ab_test_experiment.arun(dataset, variant="B")
Multi-Stage Experiments
For complex systems with multiple components:


@experiment()
async def multi_stage_experiment(row):
    # Stage 1: Retrieval
    retrieved_docs = await retriever(row["query"])

    # Stage 2: Generation
    response = await generator(row["query"], retrieved_docs)

    return {
        **row,
        "retrieved_docs": retrieved_docs,
        "response": response,
        "num_docs_retrieved": len(retrieved_docs),
        "experiment_name": "multi_stage_v1"
    }
Error Handling in Experiments
Handle errors gracefully to avoid losing partial results:


@experiment()
async def robust_experiment(row):
    try:
        response = await my_system_function(row["input"])
        error = None
    except Exception as e:
        response = None
        error = str(e)

    return {
        **row,
        "response": response,
        "error": error,
        "success": error is None,
        "experiment_name": "robust_v1"
    }
Integrating with Metrics
Experiments work seamlessly with Ragas metrics:


from ragas.metrics import FactualCorrectness

@experiment()
async def evaluated_experiment(row):
    response = await my_system_function(row["input"])

    # Calculate metrics inline
    factual_score = FactualCorrectness().score(
        response=response,
        reference=row["expected_output"]
    )

    return {
        **row,
        "response": response,
        "factual_correctness": factual_score.value,
        "factual_reason": factual_score.reason,
        "experiment_name": "evaluated_v1"
    }
This integration allows you to automatically calculate and store metric scores alongside your experiment results, making it easy to track performance improvements over time.
----
Datasets and Experiment Results
When we evaluate AI systems, we typically work with two main types of data:

Evaluation Datasets: These are stored under the datasets directory.
Evaluation Results: These are stored under the experiments directory.
Evaluation Datasets
A dataset for evaluations contains:

Inputs: a set of inputs that the system will process.
Expected outputs (Optional): the expected outputs or responses from the system for the given inputs.
Metadata (Optional): additional information that can be stored alongside the dataset.
For example, in a Retrieval-Augmented Generation (RAG) system it might include query (input to the system), Grading notes (to grade the output from the system), and metadata like query complexity.

Metadata is particularly useful for slicing and dicing the dataset, allowing you to analyze results across different facets. For instance, you might want to see how your system performs on complex queries versus simple ones, or how it handles different languages.

Experiment Results
Experiment results include:

All attributes from the dataset.
The response from the evaluated system.
Results of metrics.
Optional metadata, such as a URI pointing to the system trace for a given input.
For example, in a RAG system, the results might include Query, Grading notes, Response, Accuracy score (metric), link to the system trace, etc.

Working with Datasets in Ragas
Ragas provides a Dataset class to work with evaluation datasets. Here's how you can use it:

Creating a Dataset

from ragas import Dataset

# Create a new dataset
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir="./data")

# Add a sample to the dataset
dataset.append({
    "id": "sample_1",
    "query": "What is the capital of France?",
    "expected_answer": "Paris",
    "metadata": {"complexity": "simple", "language": "en"}
})
Loading an Existing Dataset

# Load an existing dataset
dataset = Dataset.load(
    name="my_evaluation",
    backend="local/csv",
    root_dir="./data"
)
Dataset Structure
Datasets in Ragas are flexible and can contain any fields you need for your evaluation. Common fields include:

id: Unique identifier for each sample
query or input: The input to your AI system
expected_output or ground_truth: The expected response (if available)
metadata: Additional information about the sample
Best Practices for Dataset Creation
Representative Samples: Ensure your dataset represents the real-world scenarios your AI system will encounter.

Balanced Distribution: Include samples across different difficulty levels, topics, and edge cases.

Quality Over Quantity: It's better to have fewer high-quality, well-curated samples than many low-quality ones.

Metadata Rich: Include relevant metadata that allows you to analyze performance across different dimensions.

Version Control: Track changes to your datasets over time to ensure reproducibility.

Dataset Storage and Management
Local Storage
For local development and small datasets, you can use CSV files:


dataset = Dataset(name="my_eval", backend="local/csv", root_dir="./datasets")
Cloud Storage
For larger datasets or team collaboration, consider cloud backends:


# Google Drive (experimental)
dataset = Dataset(name="my_eval", backend="gdrive", root_dir="folder_id")

# Other backends can be added as needed
Dataset Versioning
Keep track of dataset versions for reproducible experiments:


# Include version in dataset name
dataset = Dataset(name="my_eval_v1.2", backend="local/csv", root_dir="./datasets")
Integration with Evaluation Workflows
Datasets integrate seamlessly with Ragas evaluation workflows:


from ragas import experiment, Dataset

# Load your dataset
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir="./data")

# Define your experiment
@experiment()
async def my_experiment(row):
    # Process the input through your AI system
    response = await my_ai_system(row["query"])

    # Return results for metric evaluation
    return {
        **row,  # Include original data
        "response": response,
        "experiment_name": "baseline_v1"
    }

# Run evaluation on the dataset
results = await my_experiment.arun(dataset)
This integration allows you to maintain a clear separation between your test data (datasets) and your evaluation results (experiments), making it easier to track progress and compare different approaches.
----
Overview of Metrics
Why Metrics Matter
You can't improve what you don't measure. Metrics are the feedback loop that makes iteration possible.

In AI systems, progress depends on running many experiments—each a hypothesis about how to improve performance. But without a clear, reliable metric, you can't tell the difference between a successful experiment (a positive delta between the new score and the old one) and a failed one.

Metrics give you a compass. They let you quantify improvement, detect regressions, and align optimization efforts with user impact and business value.

A metric is a quantitative measure used to evaluate the performance of a AI application. Metrics help in assessing how well the application and individual components that makes up application is performing relative to the given test data. They provide a numerical basis for comparison, optimization, and decision-making throughout the application development and deployment process. Metrics are crucial for:

Component Selection: Metrics can be used to compare different components of the AI application like LLM, Retriever, Agent configuration, etc with your own data and select the best one from different options.
Error Diagnosis and Debugging: Metrics help identify which part of the application is causing errors or suboptimal performance, making it easier to debug and refine.
Continuous Monitoring and Maintenance: Metrics enable the tracking of an AI application's performance over time, helping to detect and respond to issues such as data drift, model degradation, or changing user requirements.
Types of Metrics in AI Applications
1. End-to-End Metrics
End-to-end metrics evaluate the overall system performance from the user's perspective, treating the AI application as a black box. These metrics quantify key outcomes users care deeply about, based solely on the system's final outputs.

Examples:

Answer correctness: Measures if the provided answers from a Retrieval-Augmented Generation (RAG) system are accurate.
Citation accuracy: Evaluates whether the references cited by the RAG system are correctly identified and relevant.
Optimizing end-to-end metrics ensures tangible improvements aligned directly with user expectations.

2. Component-Level Metrics
Component-level metrics assess the individual parts of an AI system independently. These metrics are immediately actionable and facilitate targeted improvements but do not necessarily correlate directly with end-user satisfaction.

Example:

Retrieval accuracy: Measures how effectively a RAG system retrieves relevant information. A low retrieval accuracy (e.g., 50%) signals that improving this component can enhance overall system performance. However, improving a component alone doesn't guarantee better end-to-end outcomes.
3. Business Metrics
Business metrics align AI system performance with organizational objectives and quantify tangible business outcomes. These metrics are typically lagging indicators, calculated after a deployment period (days/weeks/months).

Example:

Ticket deflection rate: Measures the percentage reduction of support tickets due to the deployment of an AI assistant.
Types of Metrics in Ragas
Component-wise Evaluation
Metrics Mind map
Metrics can be classified into two categories based on the mechanism used underneath the hood:

     LLM-based metrics: These metrics use LLM underneath to do the evaluation. There might be one or more LLM calls that are performed to arrive at the score or result. These metrics can be somewhat non-deterministic as the LLM might not always return the same result for the same input. On the other hand, these metrics has shown to be more accurate and closer to human evaluation.

All LLM based metrics in ragas are inherited from MetricWithLLM class. These metrics expects a LLM object to be set before scoring.


from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluation_llm)
Each LLM based metrics also will have prompts associated with it written using Prompt Object. You can customize these prompts to suit your domain and use-case. Learn more in the Modifying Prompts in Metrics guide.

     Non-LLM-based metrics: These metrics do not use LLM underneath to do the evaluation. These metrics are deterministic and can be used to evaluate the performance of the AI application without using LLM. These metrics rely on traditional methods to evaluate the performance of the AI application, such as string similarity, BLEU score, etc. Due to the same, these metrics are known to have a lower correlation with human evaluation.

All Non-LLM-based metrics in ragas are inherited from Metric class.

Metrics can be broadly classified into two categories based on the type of data they evaluate:

     Single turn metrics: These metrics evaluate the performance of the AI application based on a single turn of interaction between the user and the AI. All metrics in ragas that supports single turn evaluation are inherited from SingleTurnMetric class and scored using single_turn_ascore method. It also expects a Single Turn Sample object as input.


from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness()
await scorer.single_turn_ascore(sample)
     Multi-turn metrics: These metrics evaluate the performance of the AI application based on multiple turns of interaction between the user and the AI. All metrics in ragas that supports multi turn evaluation are inherited from MultiTurnMetric class and scored using multi_turn_ascore method. It also expects a Multi Turn Sample object as input.


from ragas.metrics import AgentGoalAccuracy
from ragas import MultiTurnSample

scorer = AgentGoalAccuracy()
await scorer.multi_turn_ascore(sample)
Output Types
In Ragas, we categorize metrics based on the type of output they produce. This classification helps clarify how each metric behaves and how its results can be interpreted or aggregated. The three types are:

1. Discrete Metrics
These return a single value from a predefined list of categorical classes. There is no implicit ordering among the classes. Common use cases include classifying outputs into categories such as pass/fail or good/okay/bad. Discrete metrics accept custom prompts directly, making them ideal for quick custom evaluations.

Example:


from ragas.metrics import discrete_metric

@discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
For modifying prompts in existing collection metrics (like Faithfulness, FactualCorrectness), see Modifying prompts in metrics.

2. Numeric Metrics
These return an integer or float value within a specified range. Numeric metrics support aggregation functions such as mean, sum, or mode, making them useful for statistical analysis.


from ragas.metrics import numeric_metric

@numeric_metric(name="response_accuracy", allowed_values=(0, 1))
def my_metric(predicted: float, expected: float) -> float:
    return abs(predicted - expected) / max(expected, 1e-5)

my_metric.score(predicted=0.8, expected=1.0)  # Returns a float value
3. Ranking Metrics
These evaluate multiple outputs at once and return a ranked list based on a defined criterion. They are useful when the goal is to compare multiple outputs from the same pipeline relative to one another.


from ragas.metrics import ranking_metric
@ranking_metric(name="response_ranking", allowed_values=[0,1])
def my_metric(responses: list) -> list:
    response_lengths = [len(response) for response in responses]
    sorted_indices = sorted(range(len(response_lengths)), key=lambda i: response_lengths[i])
    return sorted_indices

my_metric.score(responses=["short", "a bit longer", "the longest response"])  # Returns a ranked list of indices
Metric Design Principles
Designing effective metrics for AI applications requires following to a set of core principles to ensure their reliability, interpretability, and relevance. Here are five key principles we follow in ragas when designing metrics:

1. Single-Aspect Focus A single metric should target only one specific aspect of the AI application's performance. This ensures that the metric is both interpretable and actionable, providing clear insights into what is being measured.

2. Intuitive and Interpretable Metrics should be designed to be easy to understand and interpret. Clear and intuitive metrics make it simpler to communicate results and draw meaningful conclusions.

3. Effective Prompt Flows When developing metrics using large language models (LLMs), use intelligent prompt flows that align closely with human evaluation. Decomposing complex tasks into smaller sub-tasks with specific prompts can improve the accuracy and relevance of the metric.

4. Robustness Ensure that LLM-based metrics include sufficient few-shot examples that reflect the desired outcomes. This enhances the robustness of the metric by providing context and guidance for the LLM to follow.

5.Consistent Scoring Ranges It is crucial to normalize metric score values or ensure they fall within a specific range, such as 0 to 1. This facilitates comparison between different metrics and helps maintain consistency and interpretability across the evaluation framework.

These principles serve as a foundation for creating metrics that are not only effective but also practical and meaningful in evaluating AI applications.

Choosing the Right Metrics for Your Application
1. Prioritize End-to-End Metrics
Focus first on metrics reflecting overall user satisfaction. While many aspects influence user satisfaction—such as factual correctness, response tone, and explanation depth—concentrate initially on the few dimensions delivering maximum user value (e.g., answer and citation accuracy in a RAG-based assistant).

2. Ensure Interpretability
Design metrics clear enough for the entire team to interpret and reason about. For example:

Execution accuracy in a text-to-SQL system: Does the SQL query generated return precisely the same dataset as the ground truth query crafted by domain experts?
3. Emphasize Objective Over Subjective Metrics
Prioritize metrics with objective criteria, minimizing subjective judgment. Assess objectivity by independently labeling samples across team members and measuring agreement levels. A high inter-rater agreement (≥80%) indicates greater objectivity.

4. Few Strong Signals over Many Weak Signals
Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals. For instance:

In a conversational AI, using a single metric such as goal accuracy (whether the user's objective for interacting with the AI was met) provides strong proxy for the performance of the system than multiple weak proxies like coherence or helpfulness.
----
List of available metrics
Ragas provides a set of evaluation metrics that can be used to measure the performance of your LLM application. These metrics are designed to help you objectively measure the performance of your application. Metrics are available for different applications and tasks, such as RAG and Agentic workflows.

Each metric are essentially paradigms that are designed to evaluate a particular aspect of the application. LLM Based metrics might use one or more LLM calls to arrive at the score or result. One can also modify or write your own metrics using ragas.

Retrieval Augmented Generation
Context Precision
Context Recall
Context Entities Recall
Noise Sensitivity
Response Relevancy
Faithfulness
Multimodal Faithfulness
Multimodal Relevance
Nvidia Metrics
Answer Accuracy
Context Relevance
Response Groundedness
Agents or Tool use cases
Topic adherence
Tool call Accuracy
Tool Call F1
Agent Goal Accuracy
Natural Language Comparison
Factual Correctness
Semantic Similarity
Non LLM String Similarity
BLEU Score
CHRF Score
ROUGE Score
String Presence
Exact Match
SQL
Execution based Datacompy Score
SQL query Equivalence
General purpose
Aspect critic
Simple Criteria Scoring
Rubrics based scoring
Instance specific rubrics scoring
Other tasks
Summarization
----
Testset Generation
Curating a high quality test dataset is crucial for evaluating the performance of your AI application.

Characteristics of an Ideal Test Dataset
Contains high quality data samples
Covers wide variety of scenarios as observed in real world.
Contains enough number of samples to derive statistically significant conclusions.
Continually updated to prevent data drift
Curating such a dataset manually can be time-consuming and expensive. Ragas provides a set of tools to generate synthetic test datasets for evaluating your AI applications.
----
Testset Generation for RAG
In RAG application, when a user interacts through your application to a set of documents, there can be different patterns of queries that the system can encounter. Let's first understand the different types of queries that can be encountered in RAG application.

Query types in RAG
Queries

Single-Hop Query

Multi-Hop Query

Specific Query

Abstract Query

Specific Query

Abstract Query

Single-Hop Query
A single-hop query is a straightforward question that requires retrieving information from a single document or source to provide a relevant answer. It involves only one step to arrive at the answer.

Example (Specific Query):

“What year did Albert Einstein publish the theory of relativity?”
This is a specific, fact-based question that can be answered with a single retrieval from a document containing that information.

Example (Abstract Query):

“How did Einstein’s theory change our understanding of time and space?”
While this query still refers to a single concept (the theory of relativity), it requires a more abstract or interpretive explanation from the source material.

Multi-Hop Query
A multi-hop query involves multiple steps of reasoning, requiring information from two or more sources. The system must retrieve information from various documents and connect the dots to generate an accurate answer.

Example (Specific Query):

“Which scientist influenced Einstein’s work on relativity, and what theory did they propose?”
This requires the system to retrieve information about both the scientist who influenced Einstein and the specific theory, potentially from two different sources.

Example (Abstract Query):

“How have scientific theories on relativity evolved since Einstein’s original publication?”
This abstract query requires the retrieval of multiple pieces of information over time and across different sources to form a broad, interpretive response about the evolution of the theory.

Specific vs. Abstract Queries in a RAG
Specific Query: Focuses on clear, fact-based retrieval. The goal in RAG is to retrieve highly relevant information from one or more documents that directly address the specific question.

Abstract Query: Requires a broader, more interpretive response. In RAG, abstract queries challenge the retrieval system to pull from documents that contain higher-level reasoning, explanations, or opinions, rather than simple facts.

In both single-hop and multi-hop cases, the distinction between specific and abstract queries shapes the retrieval and generation process by determining whether the focus is on precision (specific) or on synthesizing broader ideas (abstract).

Different types of queries requires different contexts to be synthesized. To solve this problem, Ragas uses a Knowledge Graph based approach to Test set Generation.

Knowledge Graph Creation
Given that we want to manufacture different types of queries from the given set of documents, our major challenge is to identify the right set of chunks or documents to enable LLMs to create the queries. To solve this problem, Ragas uses a Knowledge Graph based approach to Test set Generation.

knowledge graph creation
knowledge graph creation
The knowledge graph is created by using the following components:

Document Splitter
The documents are chunked to form hierarchical nodes. The chunking can be done by using different splitters. For example, in the case of financial documents, the chunking can be done by using the splitter that splits the document based on the sections like Income Statement, Balance Sheet, Cash Flow Statement etc. You can write your own custom splitters to split the document based on the sections that are relevant to your domain.

Example

from ragas.testset.graph import Node

sample_nodes = [Node(
    properties={"page_content": "Einstein's theory of relativity revolutionized our understanding of space and time. It introduced the concept that time is not absolute but can change depending on the observer's frame of reference."}
),Node(
    properties={"page_content": "Time dilation occurs when an object moves close to the speed of light, causing time to pass slower relative to a stationary observer. This phenomenon is a key prediction of Einstein's special theory of relativity."}
)]
sample_nodes
Output:

[Node(id: 4f6b94, type: , properties: ['page_content']),
 Node(id: 952361, type: , properties: ['page_content'])]
Properties

Properties

Node: 4f6b94

page_content

Node: 952361

page_content

Extractors
Different extractors are used to extract information from each node that can be used to establish the relationship between the nodes. For example, in the case of financial documents, the extractor that can be used are entity extractor to extract the entities like Company Name, Keyphrase extractor to extract important key phrases present in each node, etc. You can write your own custom extractors to extract the information that is relevant to your domain.

Extractors can be LLM based which are inherited from LLMBasedExtractor or rule based which are inherited from Extractor.

Example
Let's say we have a sample node from the knowledge graph. We can use the NERExtractor to extract the named entities from the node.


from ragas.testset.transforms.extractors import NERExtractor

extractor = NERExtractor()
output = [await extractor.extract(node) for node in sample_nodes]
output[0]
Returns a tuple of the type of the extractor and the extracted information.

('entities', ['Einstein', 'theory of relativity', 'space', 'time', "observer's frame of reference"])
Let's add the extracted information to the node.


_ = [node.properties.update({key:val}) for (key,val), node in zip(output, sample_nodes)]
sample_nodes[0].properties
Output:


{'page_content': "Einstein's theory of relativity revolutionized our understanding of space and time. It introduced the concept that time is not absolute but can change depending on the observer's frame of reference.", 
'entities': ['Einstein', 'theory of relativity', 'space', 'time', 'observer']}
Properties

Properties

Properties

Properties

Node: 4f6b94

page_content

entities

Node: 952361

page_content

entities

Relationship builder
The extracted information is used to establish the relationship between the nodes. For example, in the case of financial documents, the relationship can be established between the nodes based on the entities present in the nodes. You can write your own custom relationship builder to establish the relationship between the nodes based on the information that is relevant to your domain.

Example

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder

kg = KnowledgeGraph(nodes=sample_nodes)
rel_builder = JaccardSimilarityBuilder(property_name="entities", key_name="PER", new_property_name="entity_jaccard_similarity")
relationships = await rel_builder.transform(kg)
relationships
Output:

[Relationship(Node(id: 4f6b94) <-> Node(id: 952361), type: jaccard_similarity, properties: ['entity_jaccard_similarity'])]
Since both the nodes have the same entity "Einstein", the relationship is established between the nodes based on the entity similarity.
Properties

Properties

Properties

Properties

entity_jaccard_similarity

Node: 4f6b94

page_content

entities

Node: 952361

page_content

entities

Now let's understand how to build the knowledge graph using the above components with a transform, that would make your job easier.

Transforms
All of the components used to build the knowledge graph can be combined into a single transform that can be applied to the knowledge graph to build the knowledge graph. Transforms is made of up of a list of components that are applied to the knowledge graph in a sequence. It can also handle parallel processing of the components. The apply_transforms method is used to apply the transforms to the knowledge graph.

Example
Let's build the above knowledge graph using the above components with a transform.


from ragas.testset.transforms import apply_transforms
transforms = [
    extractor,
    rel_builder
    ]

apply_transforms(kg,transforms)
To apply few of the components in parallel, you can wrap them in Parallel class.


from ragas.testset.transforms import KeyphraseExtractor, NERExtractor
from ragas.testset.transforms import apply_transforms, Parallel

tranforms = [
    Parallel(
        KeyphraseExtractor(),
        NERExtractor()
    ),
    rel_builder
]

apply_transforms(kg,transforms)
Once the knowledge graph is created, the different types of queries can be generated by traversing the graph. For example, to generate the query “Compare the revenue growth of Company X and Company Y from FY2020 through FY2023”, the graph can be traversed to find the nodes that contain the information about the revenue growth of Company X and Company Y from FY2020 through FY2023.

Scenario Generation
Now we have the knowledge graph that can be used to manufacture the right context to generate any type of query. When a population of users interact with RAG system, they may formulate the queries in various ways depending upon their persona (eg, Senior Engineer, Junior Engineer, etc), Query length (Short, Long, etc), Query style (Formal, Informal, etc). To generate the queries that cover all these scenarios, Ragas uses a Scenario based approach to Test set Generation.

Each Scenario in Test set Generation is a combination of following parameters.

Nodes : The nodes that are used to generate the query
Query Length : The length of the desired query, it can be short, medium or long, etc.
Query Style : The style of the query, it can be web search, chat, etc.
Persona : The persona of the user, it can be Senior Engineer, Junior Engineer, etc. (Coming soon)
Scenario in Test Generation
Scenario in Test Generation
Query Synthesizer
The QuerySynthesizer is responsible for generating different scenarios for a single query type. The generate_scenarios method is used to generate the scenarios for a single query type. The generate_sample method is used to generate the query and reference answer for a single scenario. Let's understand this with an example.

Example
In the previous example, we have created a knowledge graph that contains two nodes that are related to each other based on the entity similarity. Now imagine that you have 20 such pairs of nodes in your KG that are related to each other based on the entity similarity.

Imagine your goal is to create 50 different queries where each query is about some abstract question comparing two entities. We first have to query the KG to get the pairs of nodes that are related to each other based on the entity similarity. Then we have to generate the scenarios for each pair of nodes until we get 50 different scenarios. This logic is implemented in generate_scenarios method.


from dataclasses import dataclass
from ragas.testset.synthesizers.base_query import QuerySynthesizer

@dataclass
class EntityQuerySynthesizer(QuerySynthesizer):

    async def _generate_scenarios( self, n, knowledge_graph, callbacks):
        """
        logic to query nodes with entity
        logic describing how to combine nodes,styles,length,persona to form n scenarios
        """

        return scenarios

    async def _generate_sample(
        self, scenario, callbacks
    ):

        """
        logic on how to use tranform each scenario to EvalSample (Query,Context,Reference)
        you may create singleturn or multiturn sample
        """

        return SingleTurnSample(user_input=query, reference_contexs=contexts, reference=reference)

----
