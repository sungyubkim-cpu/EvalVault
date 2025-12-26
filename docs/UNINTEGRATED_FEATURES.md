# 미연결 기능 분석 보고서

> **작성일**: 2025-12-25
> **최종 수정**: 2025-12-25
> **목적**: 개발되었으나 파이프라인에 연결되지 않은 기능 목록 (이후 작업용 백데이터)

---

## 요약

### Active (모두 완료)

| ID | 기능 | 상태 | 완료일 |
|----|-----|------|-------|
| UNIT-01 | Thinking/Extended Reasoning | ✅ 완료 | 2025-12-25 |
| UNIT-02 | InsuranceTermAccuracy 메트릭 | ✅ 완료 | 2025-12-25 |
| UNIT-03 | KnowledgeGraph 생성 서비스 | ✅ 완료 | 2025-12-25 |

### Deferred (보류 - 필요 시 작업)

| 기능 | 사유 | 테스트 상태 |
|-----|------|-----------|
| 다국어 프롬프트 | 현재 사용 사례에서 불필요 | 44개 Pass |
| Anthropic/Azure LLM 어댑터 | OpenAI, Ollama만 사용 | 37개 Pass |
| PostgreSQL Storage | SQLite로 충분 | 19개 Pass |
| MLflow Tracker | Langfuse만 사용 | 17개 Pass |

---

## Active Work Units

---

### UNIT-01: Thinking/Extended Reasoning 파이프라인 연결

**상태**: ✅ 완료 (2025-12-25)
**우선순위**: P0 (Critical)
**의존성**: 없음 (독립 작업)

#### 현황

`gpt-oss-safeguard:20b` 모델의 thinking 기능이 설정에만 정의되어 있고 실제 평가에 사용되지 않음.

#### 관련 파일

| 파일 | 라인 | 설명 |
|-----|------|-----|
| `src/evalvault/config/settings.py` | 58-61 | `ollama_think_level` 설정 정의 |
| `src/evalvault/adapters/outbound/llm/ollama_adapter.py` | 47, 162-168 | `get_think_level()` 메서드 |
| `config/models.yaml` | 37-38 | `think_level: medium` 프로필 설정 |

#### 문제점

1. `get_think_level()` 메서드가 어디에서도 호출되지 않음
2. Ragas LLM 초기화 시 thinking level이 전달되지 않음
3. 모델의 thinking 기능이 활용되지 않음

#### 작업 내용

```yaml
files_to_modify:
  - path: src/evalvault/adapters/outbound/llm/ollama_adapter.py
    changes:
      - location: "_create_ragas_llm() 또는 LLM 호출 부분"
        action: "think_level을 options에 주입"
        details: |
          Ollama API 호출 시 thinking 관련 파라미터 전달
          - options.think = true
          - options.think_level = self._think_level
```

#### 구현 상세

```python
# ollama_adapter.py 수정안
def _create_ragas_llm(self) -> BaseLLM:
    """Ragas용 LLM 인스턴스 생성 (thinking 지원)"""
    options = {}
    if self._think_level:
        # Ollama의 thinking 모드 활성화
        options["think"] = True
        options["think_level"] = self._think_level

    return llm_factory(
        model=self._model_name,
        provider="openai",  # Ollama는 OpenAI 호환 API 사용
        client=self._client,
        run_config=RunConfig(
            timeout=self._timeout,
            options=options,
        ),
    )
```

#### 검증 기준

```bash
# 테스트 명령
pytest tests/unit/test_ollama_adapter.py -v -k "think"

# 수동 검증
evalvault config  # think_level 표시 확인
evalvault run sample.json --profile prod  # thinking 동작 확인
```

#### 완료 조건

- [x] `get_think_level()` 반환값이 LLM 호출에 적용됨
- [x] 평가 실행 시 thinking이 동작함 (응답 품질 향상)
- [x] 기존 테스트 통과

---

### UNIT-02: InsuranceTermAccuracy 메트릭 등록

**상태**: ✅ 완료 (2025-12-25)
**우선순위**: P1 (High)
**의존성**: 없음 (독립 작업)

#### 현황

보험 용어 정확성 메트릭이 완전히 구현되어 있으나 CLI와 evaluator에 등록되지 않음.

#### 관련 파일

| 파일 | 라인 | 설명 |
|-----|------|-----|
| `src/evalvault/domain/metrics/insurance.py` | 전체 | InsuranceTermAccuracy 클래스 |
| `src/evalvault/adapters/inbound/cli.py` | 32-37 | AVAILABLE_METRICS 상수 |
| `src/evalvault/domain/services/evaluator.py` | 46-53 | METRIC_MAP 딕셔너리 |

#### 문제점

1. `AVAILABLE_METRICS`에 포함되지 않음
2. `METRIC_MAP`에 포함되지 않음
3. CLI에서 `--metrics insurance_term_accuracy` 사용 불가

#### 테스트 현황

| 테스트 파일 | 테스트 수 | 상태 |
|-----------|---------|------|
| `tests/unit/test_insurance_metric.py` | 18 | ✅ Pass |

#### 작업 내용

```yaml
files_to_modify:
  - path: src/evalvault/adapters/inbound/cli.py
    changes:
      - location: "AVAILABLE_METRICS 상수"
        action: "'insurance_term_accuracy' 추가"

  - path: src/evalvault/domain/services/evaluator.py
    changes:
      - location: "상단 import 및 METRIC_MAP"
        action: "InsuranceTermAccuracy import 및 등록"
      - location: "evaluate() 메서드"
        action: "커스텀 메트릭 처리 분기 추가"
```

#### 구현 상세

```python
# cli.py 수정
AVAILABLE_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "insurance_term_accuracy",  # 추가
]

# evaluator.py 수정
from evalvault.domain.metrics.insurance import InsuranceTermAccuracy

# Ragas 메트릭
METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

# 커스텀 메트릭 (별도 처리 필요)
CUSTOM_METRIC_MAP = {
    "insurance_term_accuracy": InsuranceTermAccuracy,
}

async def evaluate(self, dataset: Dataset, metrics: list[str]) -> EvaluationRun:
    ragas_metrics = [m for m in metrics if m in METRIC_MAP]
    custom_metrics = [m for m in metrics if m in CUSTOM_METRIC_MAP]

    results = {}

    # Ragas 메트릭 평가
    if ragas_metrics:
        ragas_results = await self._evaluate_ragas(dataset, ragas_metrics)
        results.update(ragas_results)

    # 커스텀 메트릭 평가
    for metric_name in custom_metrics:
        metric_cls = CUSTOM_METRIC_MAP[metric_name]
        metric = metric_cls()
        custom_results = await self._evaluate_custom(dataset, metric)
        results[metric_name] = custom_results

    return self._build_evaluation_run(dataset, results)
```

#### 검증 기준

```bash
# 단위 테스트
pytest tests/unit/test_insurance_metric.py -v

# CLI 테스트
evalvault metrics  # insurance_term_accuracy 표시 확인
evalvault run sample.json --metrics insurance_term_accuracy
evalvault run sample.json --metrics faithfulness,insurance_term_accuracy
```

#### 완료 조건

- [x] `evalvault metrics` 출력에 insurance_term_accuracy 포함
- [x] 단독 실행 가능: `--metrics insurance_term_accuracy`
- [x] Ragas 메트릭과 혼합 실행 가능
- [x] 기존 테스트 통과

---

### UNIT-03: KnowledgeGraph 생성 서비스 CLI 연결

**상태**: ✅ 완료 (2025-12-25)
**우선순위**: P1 (High)
**의존성**: 없음 (독립 작업)

#### 현황

지식 그래프 기반 테스트셋 생성기가 구현되어 있으나 CLI에서 BasicTestsetGenerator만 사용 중.

#### 관련 파일

| 파일 | 라인 | 설명 |
|-----|------|-----|
| `src/evalvault/domain/services/kg_generator.py` | 전체 | KnowledgeGraphGenerator 클래스 |
| `src/evalvault/domain/services/entity_extractor.py` | 전체 | EntityExtractor 클래스 |
| `src/evalvault/adapters/inbound/cli.py` | 503 | generate 명령 (BasicTestsetGenerator만 사용) |

#### 문제점

1. CLI `generate` 명령은 `BasicTestsetGenerator`만 사용
2. `--method knowledge_graph` 옵션 없음
3. Multi-hop 질문 생성 기능 미활용
4. `EntityExtractor`가 단독 호출되지 않음

#### 테스트 현황

| 테스트 파일 | 테스트 수 | 상태 |
|-----------|---------|------|
| `tests/unit/test_kg_generator.py` | 27 | ✅ Pass |
| `tests/unit/test_entity_extractor.py` | 20 | ✅ Pass |

#### 작업 내용

```yaml
files_to_modify:
  - path: src/evalvault/adapters/inbound/cli.py
    changes:
      - location: "generate 명령 정의"
        action: "--method 옵션 추가 (basic, knowledge_graph)"
      - location: "generate 명령 본문"
        action: "method에 따른 생성기 선택 로직"

files_to_create:
  - path: src/evalvault/domain/services/generator_factory.py
    content: "생성기 팩토리 (선택사항)"
```

#### 구현 상세

```python
# cli.py generate 명령 수정
from evalvault.domain.services.kg_generator import KnowledgeGraphGenerator
from evalvault.domain.services.testset_generator import BasicTestsetGenerator

@app.command()
def generate(
    documents_path: Path = typer.Argument(...),
    output_path: Path = typer.Option(...),
    count: int = typer.Option(10, "--count", "-n"),
    method: str = typer.Option(
        "basic",
        "--method", "-m",
        help="Generation method: basic (simple QA), knowledge_graph (multi-hop)",
    ),
):
    """문서로부터 테스트 데이터셋 생성"""

    # 문서 로드
    documents = load_documents(documents_path)

    # 생성기 선택
    if method == "knowledge_graph":
        console.print("[cyan]Using Knowledge Graph generator (multi-hop questions)[/cyan]")
        generator = KnowledgeGraphGenerator()
        config = KGGenerationConfig(
            num_questions=count,
            include_multi_hop=True,
            max_hop_depth=3,
        )
    else:
        console.print("[cyan]Using Basic generator (simple QA)[/cyan]")
        generator = BasicTestsetGenerator()
        config = GenerationConfig(num_questions=count)

    # 생성 실행
    with console.status("Generating test cases..."):
        dataset = generator.generate(documents, config)

    # 저장
    save_dataset(dataset, output_path)
    console.print(f"[green]Generated {len(dataset.test_cases)} test cases[/green]")
```

#### 검증 기준

```bash
# 단위 테스트
pytest tests/unit/test_kg_generator.py -v
pytest tests/unit/test_entity_extractor.py -v

# CLI 테스트
evalvault generate docs/ -o dataset.json --method basic
evalvault generate docs/ -o dataset.json --method knowledge_graph
evalvault generate docs/ -o dataset.json --method knowledge_graph -n 20
```

#### 완료 조건

- [x] `--method` CLI 옵션 동작
- [x] `knowledge_graph` 선택 시 KnowledgeGraphGenerator 사용
- [x] Multi-hop 질문 생성 확인
- [x] 기존 테스트 통과

---

## 병렬 작업 계획

### 의존성 그래프

```
                ┌────────────────────────────────┐
                │      독립 작업 (병렬 가능)       │
                └────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
    [UNIT-01]             [UNIT-02]             [UNIT-03]
    Thinking              Insurance             KG Generator
    Integration           Metric                CLI Integration
        │                     │                     │
        └─────────────────────┴──────────┬──────────┘
                                         ▼
                                   [UNIT-04]
                                   통합 테스트 & 문서화
```

### 에이전트 할당

| 에이전트 | 할당 유닛 | 주요 변경 파일 | 충돌 가능성 |
|---------|----------|--------------|-----------|
| Agent-A | UNIT-01 (Thinking) | `ollama_adapter.py` | ✅ 없음 |
| Agent-B | UNIT-02 (Insurance) | `cli.py`, `evaluator.py` | ⚠️ 낮음 |
| Agent-C | UNIT-03 (KG Generator) | `cli.py` | ⚠️ 낮음 |

### 병렬 실행 명령

```bash
# 3개 에이전트 동시 실행 (Phase 1)
claude --parallel \
  "UNIT-01: Thinking Level 파이프라인 연결. ollama_adapter.py에서 get_think_level() 반환값을 LLM 호출 options에 적용" \
  "UNIT-02: InsuranceTermAccuracy 메트릭 등록. cli.py의 AVAILABLE_METRICS와 evaluator.py의 METRIC_MAP에 추가" \
  "UNIT-03: KnowledgeGraph 생성기 CLI 연결. generate 명령에 --method 옵션 추가"

# Phase 1 완료 후 (Phase 2)
claude "UNIT-04: 통합 테스트 실행 및 문서 업데이트"
```

### 충돌 방지

#### cli.py 분할 작업

UNIT-02와 UNIT-03 모두 `cli.py`를 수정하지만, 서로 다른 부분을 수정:
- UNIT-02: `AVAILABLE_METRICS` 상수, `run` 명령 관련
- UNIT-03: `generate` 명령 관련

충돌 최소화를 위해:
1. 각 에이전트는 자신의 영역만 수정
2. 필요시 브랜치 분리 후 병합

#### Branch 전략

```
main
├── feature/unit-01-thinking
├── feature/unit-02-insurance-metric
└── feature/unit-03-kg-generator
```

---

## Deferred (보류된 기능)

> 현재 사용 사례에서 필요하지 않아 보류. 어댑터는 구현 완료되어 테스트 통과 상태.
> 향후 필요 시 CLI 연결 작업만 수행하면 됨.

### 다국어 프롬프트

**보류 사유**: 현재 한국어/영어 기본 프롬프트로 충분

| 파일 | 설명 |
|-----|-----|
| `src/evalvault/utils/language.py` | LanguageDetector 클래스 |
| `src/evalvault/domain/prompts/*.py` | 4개 언어 프롬프트 |

**테스트**: 44개 Pass (`test_language_utils.py`, `test_prompts.py`)

**활성화 시 필요 작업**:
- CLI `--language` 옵션 추가
- evaluator에 프롬프트 선택 로직 추가

---

### LLM 어댑터 (Anthropic, Azure)

**보류 사유**: OpenAI, Ollama만 사용

| 파일 | 설명 |
|-----|-----|
| `src/evalvault/adapters/outbound/llm/anthropic_adapter.py` | Claude 어댑터 |
| `src/evalvault/adapters/outbound/llm/azure_adapter.py` | Azure OpenAI 어댑터 |

**테스트**: 37개 Pass (`test_anthropic_adapter.py`, `test_azure_adapter.py`)

**활성화 시 필요 작업**:
- CLI `--llm-provider` 옵션 추가
- 어댑터 팩토리 패턴 구현

---

### PostgreSQL Storage

**보류 사유**: SQLite로 충분

| 파일 | 설명 |
|-----|-----|
| `src/evalvault/adapters/outbound/storage/postgres_adapter.py` | PostgreSQL 어댑터 |
| `src/evalvault/adapters/outbound/storage/postgres_schema.sql` | 스키마 정의 |

**테스트**: 19개 Pass (`test_postgres_storage.py`)

**활성화 시 필요 작업**:
- CLI `--storage` 옵션 추가
- 스토리지 팩토리 구현

---

### MLflow Tracker

**보류 사유**: Langfuse만 사용

| 파일 | 설명 |
|-----|-----|
| `src/evalvault/adapters/outbound/tracker/mlflow_adapter.py` | MLflow 어댑터 |

**테스트**: 17개 Pass (`test_mlflow_tracker.py`)

**활성화 시 필요 작업**:
- CLI `--tracker` 옵션 추가
- 트래커 팩토리 구현

---

## 미사용 Settings 옵션 (참고용)

### Active 항목 관련

| 옵션 | 사용 여부 | 관련 UNIT |
|-----|----------|----------|
| `ollama_think_level` | ❌ 미사용 | UNIT-01 |

### Deferred 항목 관련

| 옵션 | 사용 여부 | 보류 사유 |
|-----|----------|---------|
| `azure_openai_*` | ❌ | OpenAI/Ollama만 사용 |
| `anthropic_*` | ❌ | OpenAI/Ollama만 사용 |
| `mlflow_*` | ❌ | Langfuse만 사용 |
| `postgres_*` | ❌ | SQLite로 충분 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|-----|------|---------|
| 2025-12-25 | 1.0.0 | 초기 문서 작성 |
| 2025-12-25 | 1.1.0 | 병렬 작업 계획 섹션 추가 |
| 2025-12-25 | 2.0.0 | Active/Deferred 구분, 3개 UNIT으로 축소 |
| 2025-12-25 | 3.0.0 | UNIT-01, 02, 03 모두 완료 (병렬 작업 성공) |
