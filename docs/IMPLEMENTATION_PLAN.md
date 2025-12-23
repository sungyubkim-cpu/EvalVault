# EvalVault 구현 계획서

> 한국어/영어 보험 문서 RAG 평가 시스템 - RAGAS + Langfuse 기반

## 1. 현재 상태 분석

### 1.1 구현 완료된 부분

| 컴포넌트 | 상태 | 파일 |
|---------|------|------|
| Domain Entities | ✅ 완료 | `dataset.py`, `result.py` |
| RagasEvaluator | ✅ 기본 구현 | `evaluator.py` |
| Langfuse Adapter | ✅ 완료 | `langfuse_adapter.py` |
| Dataset Loaders | ✅ CSV/JSON/Excel | `adapters/outbound/dataset/` |
| OpenAI Adapter | ✅ 완료 | `openai_adapter.py` |

### 1.2 현재 지원 메트릭

```python
METRIC_MAP = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
}
```

### 1.3 개선이 필요한 영역

1. **다국어(한국어/영어) 지원**: 프롬프트 커스터마이징 필요
2. **보험 도메인 특화 메트릭**: 도메인 용어, 규정 준수 평가
3. **Testset Generation**: 보험 문서 특화 테스트셋 자동 생성
4. **Experiment 관리**: RAGAS @experiment 데코레이터 통합
5. **고급 메트릭**: FactualCorrectness, SemanticSimilarity 등 추가

---

## 2. 다국어(Multilingual) 지원 전략

### 2.1 지원 언어 우선순위

보험 문서에 포함될 수 있는 언어의 우선순위:

| 우선순위 | 언어 | 코드 | 예상 비율 | 비고 |
|---------|------|------|----------|------|
| 1순위 | 한국어 | `ko` | ~70% | 주요 언어 |
| 2순위 | 영어 | `en` | ~20% | 보조 언어, 기술 용어 |
| 3순위 | 중국어 | `zh` | ~5% | 중국인 고객 대상 |
| 4순위 | 일본어 | `ja` | ~3% | 일본인 고객 대상 |
| 5순위 | 베트남어 | `vi` | ~2% | 베트남인 고객 대상 |
| 기타 | 다국어 | `*` | - | 필요시 확장 |

### 2.2 언어 감지 유틸리티

```python
# src/evalvault/utils/language.py

from dataclasses import dataclass
from typing import Literal

# 지원 언어 타입
SupportedLanguage = Literal["ko", "en", "zh", "ja", "vi", "other"]

# 언어별 유니코드 범위
LANGUAGE_RANGES = {
    "ko": [('\uac00', '\ud7af'), ('\u1100', '\u11ff')],  # 한글
    "zh": [('\u4e00', '\u9fff')],  # 중국어 (CJK)
    "ja": [('\u3040', '\u309f'), ('\u30a0', '\u30ff')],  # 히라가나, 카타카나
    "vi": [('\u0100', '\u017f')],  # 베트남어 특수문자 (Latin Extended)
}

@dataclass
class LanguageDetectionResult:
    """언어 감지 결과."""
    primary: SupportedLanguage
    confidence: float
    distribution: dict[str, float]  # 언어별 비율


def detect_language(text: str) -> LanguageDetectionResult:
    """텍스트의 주요 언어를 감지합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        언어 감지 결과 (주요 언어, 신뢰도, 분포)
    """
    if not text:
        return LanguageDetectionResult("en", 0.0, {})

    # 언어별 문자 수 계산
    char_counts = {lang: 0 for lang in LANGUAGE_RANGES}
    total_chars = len([c for c in text if not c.isspace()])

    for char in text:
        for lang, ranges in LANGUAGE_RANGES.items():
            for start, end in ranges:
                if start <= char <= end:
                    char_counts[lang] += 1
                    break

    # 분포 계산
    distribution = {
        lang: count / total_chars if total_chars > 0 else 0
        for lang, count in char_counts.items()
    }

    # 한국어 우선 판정 (CJK 범위 중첩 처리)
    if distribution.get("ko", 0) > 0.2:
        primary = "ko"
    elif distribution.get("ja", 0) > 0.1:
        primary = "ja"
    elif distribution.get("zh", 0) > 0.2:
        primary = "zh"
    elif distribution.get("vi", 0) > 0.05:
        primary = "vi"
    else:
        primary = "en"  # 기본값

    confidence = distribution.get(primary, 0) if primary != "en" else 1.0 - sum(distribution.values())

    return LanguageDetectionResult(
        primary=primary,
        confidence=confidence,
        distribution=distribution,
    )


def get_dominant_language(texts: list[str]) -> SupportedLanguage:
    """여러 텍스트에서 주요 언어를 판단합니다."""
    if not texts:
        return "en"

    # 전체 텍스트 합쳐서 분석
    combined = " ".join(texts)
    result = detect_language(combined)
    return result.primary


def is_mixed_language(text: str, threshold: float = 0.1) -> bool:
    """혼합 언어 문서인지 판단합니다.

    Args:
        text: 분석할 텍스트
        threshold: 혼합으로 판단할 최소 비율

    Returns:
        True if 2개 이상 언어가 threshold 이상 포함
    """
    result = detect_language(text)
    significant_languages = [
        lang for lang, ratio in result.distribution.items()
        if ratio >= threshold
    ]
    return len(significant_languages) >= 2
```

### 2.3 다국어 프롬프트 커스터마이징

RAGAS의 LLM 기반 메트릭은 `Prompt Object`를 통해 프롬프트를 커스터마이징할 수 있습니다.

```python
# src/evalvault/domain/prompts/multilingual_prompts.py

from ragas.metrics import Faithfulness
from evalvault.utils.language import SupportedLanguage, detect_language

class MultilingualFaithfulness(Faithfulness):
    """다국어 문서를 위한 Faithfulness 메트릭."""

    # 언어별 프롬프트 템플릿
    PROMPTS = {
        "ko": {
            "nli": "다음 주장이 주어진 맥락에서 지지되는지 판단하세요...",
            "instruction": "답변의 각 문장이 컨텍스트에 기반하는지 평가합니다.",
        },
        "en": {
            "nli": "Determine if the following claim is supported by the context...",
            "instruction": "Evaluate if each sentence in the answer is grounded in the context.",
        },
        "zh": {
            "nli": "判断以下声明是否得到上下文的支持...",
            "instruction": "评估答案中的每个句子是否基于上下文。",
        },
        "ja": {
            "nli": "以下の主張がコンテキストによって支持されているかどうかを判断してください...",
            "instruction": "回答の各文がコンテキストに基づいているかどうかを評価します。",
        },
    }

    def __init__(self, language: SupportedLanguage = "auto"):
        super().__init__()
        self.language = language

    def _get_prompt_for_language(self, lang: SupportedLanguage) -> dict:
        """언어에 맞는 프롬프트 반환."""
        return self.PROMPTS.get(lang, self.PROMPTS["en"])

    async def single_turn_ascore(self, sample):
        # 언어 자동 감지
        if self.language == "auto":
            detected = detect_language(sample.response)
            lang = detected.primary
        else:
            lang = self.language

        # 언어별 프롬프트 설정
        prompts = self._get_prompt_for_language(lang)
        self._customize_prompts(prompts)

        return await super().single_turn_ascore(sample)
```

### 2.4 구현 우선순위

| 우선순위 | 작업 | 설명 |
|---------|------|------|
| P0 | 언어 감지 유틸리티 | 다국어 자동 감지 (ko, en, zh, ja, vi) |
| P0 | 한국어 프롬프트 | Faithfulness, AnswerRelevancy 한국어화 |
| P0 | 영어 프롬프트 | 기본 RAGAS 프롬프트 (이미 지원) |
| P1 | 혼합 언어 처리 | 한영 혼합 문서 등 처리 전략 |
| P1 | 중국어/일본어 프롬프트 | 3순위/4순위 언어 지원 |
| P2 | 베트남어 프롬프트 | 5순위 언어 지원 |
| P2 | 동적 언어 전환 | 문서별 자동 프롬프트 전환 |
| P3 | 기타 언어 확장 | 필요시 추가 언어 지원 |

### 2.5 혼합 언어 문서 처리 전략

보험 문서에서 흔히 나타나는 혼합 언어 패턴:

```
패턴 1: 한글 + 영어 기술용어
  예: "이 보험의 deductible(자기부담금)은 10만원입니다."

패턴 2: 한글 + 영어 브랜드명
  예: "Samsung Life Insurance 종신보험 상품"

패턴 3: 다국어 약관
  예: 영문/중문/국문 병기 약관
```

```python
# 혼합 언어 처리 전략
class MixedLanguageStrategy:
    """혼합 언어 문서 처리 전략."""

    def process(self, text: str) -> ProcessedText:
        if is_mixed_language(text):
            # 주요 언어 기준으로 평가하되, 보조 언어 컨텍스트 유지
            primary = get_dominant_language([text])
            return ProcessedText(
                text=text,
                primary_language=primary,
                evaluation_language=primary,  # 주요 언어로 평가
                preserve_foreign_terms=True,   # 외래어 보존
            )
        else:
            primary = detect_language(text).primary
            return ProcessedText(
                text=text,
                primary_language=primary,
                evaluation_language=primary,
            )
```

---

## 3. 보험 도메인 특화 메트릭

### 3.1 추가 메트릭 목록

RAGAS Core Concepts에서 제공하는 메트릭 중 보험 도메인에 적합한 것들:

#### RAG 품질 메트릭 (기존 + 추가)

| 메트릭 | 용도 | 우선순위 |
|--------|------|----------|
| `Faithfulness` | 답변이 컨텍스트에 충실한지 | ✅ 구현됨 |
| `AnswerRelevancy` | 답변이 질문과 관련있는지 | ✅ 구현됨 |
| `ContextPrecision` | 검색된 컨텍스트의 정밀도 | ✅ 구현됨 |
| `ContextRecall` | 필요한 정보가 검색되었는지 | ✅ 구현됨 |
| `FactualCorrectness` | 사실적 정확성 (보험 규정) | P0 |
| `ContextEntitiesRecall` | 엔티티 수준 recall | P1 |
| `NoiseSensitivity` | 노이즈에 대한 민감도 | P2 |

#### 자연어 비교 메트릭

| 메트릭 | 용도 | 우선순위 |
|--------|------|----------|
| `SemanticSimilarity` | 의미적 유사도 | P0 |
| `RougeScore` | 요약 품질 (보험 요약) | P1 |
| `BleuScore` | 번역 품질 | P2 |

### 3.2 커스텀 보험 도메인 메트릭

```python
# src/evalvault/domain/metrics/insurance_metrics.py

from ragas.metrics import AspectCritic

class InsuranceTermAccuracy(AspectCritic):
    """보험 용어 정확성 평가 메트릭.

    보험 답변에서 사용된 전문 용어가 정확한지 평가합니다.
    - 보험료, 보험금, 면책사항, 보장범위 등
    """

    name = "insurance_term_accuracy"

    definition = """
    보험 관련 답변에서 전문 용어가 정확하게 사용되었는지 평가합니다.

    평가 기준:
    1. 보험 용어의 정확한 정의 사용
    2. 법적/규제적 용어의 올바른 적용
    3. 수치 정보(보험료, 보장금액)의 정확성
    """


class RegulatoryCompliance(AspectCritic):
    """규제 준수 여부 평가 메트릭.

    답변이 보험 관련 규제를 준수하는지 평가합니다.
    - 금융소비자보호법
    - 보험업법
    - 약관 설명의무
    """

    name = "regulatory_compliance"

    definition = """
    보험 관련 답변이 규제 요건을 준수하는지 평가합니다.

    평가 기준:
    1. 중요사항 고지 여부
    2. 면책사항 설명 포함 여부
    3. 소비자 권리 안내 포함 여부
    """


class DisclaimerPresence(AspectCritic):
    """면책사항/주의사항 포함 여부 메트릭.

    보험 관련 답변에 필수 면책사항이 포함되어 있는지 확인합니다.
    """

    name = "disclaimer_presence"

    definition = """
    보험 관련 답변에 적절한 면책사항이나 주의사항이 포함되어 있는지 평가합니다.
    """
```

### 3.3 Rubrics 기반 평가

보험 도메인 특화 평가를 위한 rubrics 정의:

```python
# src/evalvault/domain/metrics/insurance_rubrics.py

INSURANCE_ANSWER_RUBRIC = {
    1: "답변이 완전히 부정확하거나 위험한 정보를 포함",
    2: "답변에 중요한 누락이 있거나 부분적으로 부정확",
    3: "답변이 대체로 정확하나 면책사항/주의사항 누락",
    4: "답변이 정확하고 필요한 정보 대부분 포함",
    5: "답변이 완전히 정확하고 면책사항, 추가 안내까지 포함"
}

INSURANCE_CONTEXT_RUBRIC = {
    1: "검색된 컨텍스트가 질문과 전혀 무관",
    2: "컨텍스트가 부분적으로 관련있으나 핵심 정보 누락",
    3: "컨텍스트가 관련있으나 최신 정보가 아닐 수 있음",
    4: "컨텍스트가 질문에 적절히 대응",
    5: "컨텍스트가 완벽하게 질문을 커버하고 관련 규정까지 포함"
}
```

---

## 4. Testset Generation 전략

RAGAS의 Knowledge Graph 기반 테스트셋 생성을 보험 도메인에 적용합니다.

### 4.1 보험 문서 쿼리 유형

```
보험 문서 쿼리 유형
├── Single-Hop Query (단일 문서)
│   ├── Specific: "이 보험의 보장금액은 얼마인가요?"
│   └── Abstract: "이 보험 상품의 장단점은 무엇인가요?"
│
└── Multi-Hop Query (복수 문서)
    ├── Specific: "A보험과 B보험의 보장금액 차이는?"
    └── Abstract: "종합보험과 단독보험 중 어떤 것이 유리한가요?"
```

### 4.2 Knowledge Graph 구축

```python
# src/evalvault/testset/insurance_kg.py

from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms import (
    Parallel,
    apply_transforms,
)
from ragas.testset.transforms.extractors import (
    NERExtractor,
    KeyphraseExtractor,
)

class InsuranceKnowledgeGraphBuilder:
    """보험 문서용 Knowledge Graph 빌더."""

    def __init__(self, llm):
        self.llm = llm
        self.extractors = self._setup_extractors()

    def _setup_extractors(self):
        """보험 도메인 특화 추출기 설정."""
        return Parallel(
            # 보험 용어 추출
            NERExtractor(entity_types=["INSURANCE_TERM", "MONEY", "DATE"]),
            # 핵심 문구 추출
            KeyphraseExtractor(),
            # 커스텀: 보험 상품명 추출
            InsuranceProductExtractor(),
        )

    async def build(self, documents: list[str]) -> KnowledgeGraph:
        """문서로부터 Knowledge Graph 구축."""
        nodes = [
            Node(properties={"page_content": doc})
            for doc in documents
        ]

        kg = KnowledgeGraph(nodes=nodes)

        transforms = [
            self.extractors,
            InsuranceRelationshipBuilder(),
        ]

        await apply_transforms(kg, transforms)
        return kg


class InsuranceProductExtractor:
    """보험 상품명 및 유형 추출기."""

    async def extract(self, node):
        # 보험 상품 유형 패턴
        patterns = [
            r"(종신|정기|변액|연금|건강|실손|자동차|화재|배상책임)\s*보험",
            r"(플랜|특약|담보)",
        ]
        # ... 추출 로직
        return ("insurance_products", extracted_products)
```

### 4.3 시나리오 기반 쿼리 생성

```python
# src/evalvault/testset/insurance_synthesizer.py

from ragas.testset.synthesizers.base_query import QuerySynthesizer
from dataclasses import dataclass

@dataclass
class InsuranceQuerySynthesizer(QuerySynthesizer):
    """보험 도메인 특화 쿼리 생성기."""

    # 보험 고객 페르소나
    personas = [
        {"name": "신규 가입자", "traits": "보험 초보, 용어에 익숙하지 않음"},
        {"name": "기존 가입자", "traits": "갱신/변경 관심, 비교 분석 선호"},
        {"name": "청구 고객", "traits": "보험금 청구 절차 문의, 구체적 질문"},
        {"name": "해지 검토자", "traits": "불만족, 비용 민감, 대안 탐색"},
    ]

    # 쿼리 스타일
    query_styles = [
        "formal",      # 공식적 문의
        "casual",      # 일상적 대화
        "urgent",      # 긴급 문의
        "comparison",  # 비교 질문
    ]

    async def _generate_scenarios(self, n, knowledge_graph, callbacks):
        """시나리오 생성."""
        scenarios = []

        # KG에서 관련 노드 조합 탐색
        for node_pair in knowledge_graph.get_related_nodes():
            for persona in self.personas:
                for style in self.query_styles:
                    scenarios.append({
                        "nodes": node_pair,
                        "persona": persona,
                        "style": style,
                        "language": "ko",  # 기본 한국어
                    })

        return scenarios[:n]

    async def _generate_sample(self, scenario, callbacks):
        """시나리오로부터 테스트 샘플 생성."""
        # LLM을 사용하여 쿼리 + 기대 답변 생성
        query = await self._synthesize_query(scenario)
        reference = await self._synthesize_reference(scenario)

        return SingleTurnSample(
            user_input=query,
            reference_contexts=[n.properties["page_content"] for n in scenario["nodes"]],
            reference=reference,
        )
```

### 4.4 구현 우선순위

| 단계 | 작업 | 산출물 |
|-----|------|--------|
| 1 | 기본 문서 파서 | `InsuranceDocumentParser` |
| 2 | 엔티티 추출기 | `InsuranceProductExtractor`, `InsuranceTermExtractor` |
| 3 | KG 빌더 | `InsuranceKnowledgeGraphBuilder` |
| 4 | 쿼리 생성기 | `InsuranceQuerySynthesizer` |
| 5 | CLI 통합 | `evalvault testset generate` 명령 |

---

## 5. Experiment 관리 전략

### 5.1 RAGAS @experiment 데코레이터 통합

```python
# src/evalvault/experiments/runner.py

from ragas import experiment, Dataset
from datetime import datetime

@experiment()
async def insurance_rag_experiment(row, model_name: str, retriever_type: str):
    """보험 RAG 시스템 실험.

    Args:
        row: 데이터셋 행
        model_name: 사용할 LLM 모델
        retriever_type: 검색기 유형 (dense, sparse, hybrid)
    """
    # RAG 파이프라인 실행
    response = await run_insurance_rag_pipeline(
        query=row["user_input"],
        model=model_name,
        retriever=retriever_type,
    )

    return {
        **row,
        "response": response.answer,
        "retrieved_contexts": response.contexts,
        "experiment_name": f"{model_name}_{retriever_type}_{datetime.now():%Y%m%d}",
        "model_name": model_name,
        "retriever_type": retriever_type,
        "latency_ms": response.latency_ms,
        "tokens_used": response.tokens_used,
    }
```

### 5.2 실험 비교 및 분석

```python
# src/evalvault/experiments/analysis.py

class ExperimentAnalyzer:
    """실험 결과 비교 분석기."""

    def compare_experiments(
        self,
        experiment_ids: list[str],
        metrics: list[str] = None,
    ) -> ComparisonReport:
        """여러 실험 결과를 비교합니다.

        Returns:
            각 메트릭별 평균, 표준편차, 통계적 유의성 포함 보고서
        """
        pass

    def detect_regression(
        self,
        baseline_id: str,
        candidate_id: str,
        threshold: float = 0.05,
    ) -> RegressionReport:
        """성능 회귀 감지.

        Returns:
            회귀 발생 메트릭 및 심각도 보고서
        """
        pass
```

### 5.3 실험 결과 저장 구조

```
experiments/
├── 20241224-143022-gpt4o_dense_baseline.csv
├── 20241224-150515-gpt4o_hybrid_v1.csv
├── 20241224-160000-claude_dense_comparison.csv
└── metadata/
    ├── experiment_registry.json
    └── comparison_reports/
        └── 20241224_baseline_vs_hybrid.json
```

---

## 6. Langfuse 통합 강화

### 6.1 현재 구현 상태

- ✅ 기본 Trace 생성
- ✅ Span 추가
- ✅ Score 로깅
- ✅ EvaluationRun 전체 로깅

### 6.2 추가 통합 필요 사항

#### 6.2.1 Ragas-Langfuse 네이티브 통합

```python
# src/evalvault/adapters/outbound/tracker/langfuse_ragas_adapter.py

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from ragas.metrics import Faithfulness

class LangfuseRagasIntegration:
    """Ragas + Langfuse 네이티브 통합."""

    def __init__(self, langfuse_client: Langfuse):
        self.langfuse = langfuse_client

    @observe(name="ragas_evaluation")
    async def evaluate_with_tracing(
        self,
        sample,
        metrics: list,
    ):
        """Langfuse trace와 함께 Ragas 평가 실행."""
        results = {}

        for metric in metrics:
            with langfuse_context.update_current_trace(
                metadata={"metric": metric.name}
            ):
                score = await metric.single_turn_ascore(sample)

                # 자동으로 Langfuse에 score 기록
                langfuse_context.score_current_trace(
                    name=metric.name,
                    value=score,
                )

                results[metric.name] = score

        return results
```

#### 6.2.2 Dataset 관리 통합

```python
# Langfuse Dataset과 EvalVault Dataset 연동

class LangfuseDatasetSync:
    """Langfuse와 EvalVault 데이터셋 동기화."""

    async def push_to_langfuse(self, dataset: Dataset) -> str:
        """EvalVault 데이터셋을 Langfuse에 업로드."""
        lf_dataset = self.langfuse.create_dataset(
            name=f"{dataset.name}_v{dataset.version}",
            description=f"Insurance RAG evaluation dataset",
        )

        for test_case in dataset.test_cases:
            lf_dataset.add_item(
                input={"query": test_case.question},
                expected_output=test_case.ground_truth,
                metadata=test_case.metadata,
            )

        return lf_dataset.id

    async def pull_from_langfuse(self, dataset_name: str) -> Dataset:
        """Langfuse 데이터셋을 EvalVault로 가져오기."""
        pass
```

### 6.3 대시보드 연동

Langfuse의 기본 대시보드 외에 커스텀 뷰 설정:

1. **Evaluation Overview**: 전체 평가 결과 요약
2. **Metric Trends**: 시간에 따른 메트릭 변화
3. **Test Case Details**: 개별 테스트 케이스 분석
4. **Model Comparison**: 모델별 성능 비교

---

## 7. 구현 로드맵

### Phase 1: 기반 강화 (현재)

```
Week 1-2:
├── [x] 기본 Ragas 통합
├── [x] Langfuse 어댑터
├── [ ] 언어 감지 유틸리티
├── [ ] 한국어 프롬프트 커스터마이징
└── [ ] FactualCorrectness 메트릭 추가
```

### Phase 2: 도메인 특화

```
Week 3-4:
├── [ ] InsuranceTermAccuracy 메트릭
├── [ ] RegulatoryCompliance 메트릭
├── [ ] 보험 도메인 Rubrics 정의
└── [ ] 샘플 보험 데이터셋 생성
```

### Phase 3: Testset Generation

```
Week 5-6:
├── [ ] InsuranceDocumentParser
├── [ ] Knowledge Graph 빌더
├── [ ] InsuranceQuerySynthesizer
└── [ ] CLI 통합 (evalvault testset generate)
```

### Phase 4: 실험 관리

```
Week 7-8:
├── [ ] @experiment 데코레이터 통합
├── [ ] 실험 비교 분석기
├── [ ] Langfuse 네이티브 통합
└── [ ] 대시보드 커스터마이징
```

---

## 8. 파일 구조 (목표)

```
src/evalvault/
├── domain/
│   ├── entities/
│   │   ├── dataset.py          # ✅ 완료
│   │   └── result.py           # ✅ 완료
│   ├── services/
│   │   └── evaluator.py        # ✅ 기본 완료
│   ├── metrics/                 # 🆕 신규
│   │   ├── __init__.py
│   │   ├── bilingual.py        # 다국어 메트릭
│   │   ├── insurance.py        # 보험 특화 메트릭
│   │   └── rubrics.py          # Rubrics 정의
│   └── prompts/                 # 🆕 신규
│       ├── __init__.py
│       ├── korean.py           # 한국어 프롬프트
│       └── insurance.py        # 보험 도메인 프롬프트
├── testset/                     # 🆕 신규
│   ├── __init__.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   └── insurance.py        # 보험 엔티티 추출기
│   ├── kg_builder.py           # Knowledge Graph 빌더
│   └── synthesizer.py          # 쿼리 생성기
├── experiments/                 # 🆕 신규
│   ├── __init__.py
│   ├── runner.py               # 실험 실행기
│   └── analyzer.py             # 결과 분석기
├── adapters/
│   ├── outbound/
│   │   └── tracker/
│   │       └── langfuse_adapter.py  # ✅ 완료 (확장 예정)
│   └── inbound/
│       └── cli.py              # ✅ 기본 완료 (확장 예정)
└── utils/                       # 🆕 신규
    ├── __init__.py
    └── language.py             # 언어 감지 유틸리티
```

---

## 9. 성공 기준 (SLA)

### 9.1 메트릭 임계값

| 메트릭 | 최소 기준 | 목표 | 우수 |
|--------|----------|------|------|
| Faithfulness | 0.60 | 0.80 | 0.90 |
| Answer Relevancy | 0.65 | 0.80 | 0.90 |
| Context Precision | 0.60 | 0.75 | 0.85 |
| Context Recall | 0.60 | 0.80 | 0.90 |
| FactualCorrectness | 0.70 | 0.85 | 0.95 |
| InsuranceTermAccuracy | 0.75 | 0.90 | 0.95 |

### 9.2 시스템 요구사항

- **평가 처리량**: 100 test cases / 5분 이내
- **결과 저장**: 모든 결과 SQLite + Langfuse 이중 저장
- **재현성**: 동일 입력 → 동일 결과 (temperature=0)

---

## 10. 참고 자료

- [RAGAS Documentation](https://docs.ragas.io/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [금융위원회 보험업법](https://www.law.go.kr/)
- [금융소비자보호법](https://www.law.go.kr/)
