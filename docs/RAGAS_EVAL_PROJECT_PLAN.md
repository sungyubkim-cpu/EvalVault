# Ragas Evaluation Project - 설계 문서

> GyeotAI 개발 정책(6대 핵심 원칙)을 기반으로 한 Ragas 평가 프로젝트 추상화

---

## 1. 프로젝트 개요

### 목적
RAG 파이프라인의 품질을 체계적으로 평가하고, 모든 테스트 결과를 이력으로 관리하여 품질 추이를 추적하는 시스템

### 핵심 기능
| 기능 | 설명 |
|------|------|
| **데이터셋 관리** | Ragas 평가용 데이터셋 로드 및 버전 관리 |
| **OpenAI API 호환** | OpenAI API 형식으로 LLM 호출 (로컬/클라우드 모두 지원) |
| **Ragas 테스트 실행** | Faithfulness, Answer Relevancy, Context Precision 등 평가 |
| **결과 저장 및 추적** | 모든 테스트 결과를 이력으로 저장 |
| **통계 대시보드** | 점수 추이, 토큰 사용량, 비용 분석 |

---

## 2. 6대 핵심 원칙 적용

```
┌─────────────────────────────────────────────────────────────────┐
│  설계 원칙 (Design)           │  실행 원칙 (Execution)          │
│  ───────────────────────────  │  ───────────────────────────    │
│  1. Spec-First (명세 우선)    │  4. SLA-Driven (목표 기반)     │
│  2. YAGNI (필요한 것만)       │  5. Experiment-Driven (실험)   │
│  3. Testable (테스트 가능)    │  6. Evidence-Based (근거 기반) │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Spec-First 적용

```
명세 → 인터페이스 → 테스트 → 구현
```

**이 프로젝트에서의 적용:**
1. 데이터셋 스키마 정의 먼저
2. 평가 메트릭 인터페이스 정의
3. 저장소 인터페이스 정의
4. 구현체 개발

### 2.2 YAGNI 적용

**포함할 것 (MVP):**
- 데이터셋 로드 (JSON/CSV)
- Ragas 핵심 메트릭 4개: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- 결과 저장 (SQLite 또는 셀프호스팅 도구)
- 기본 통계 (평균, 분포)

**포함하지 않을 것 (후순위):**
- 복잡한 시각화 대시보드
- 실시간 알림
- 멀티 모델 A/B 테스트
- 자동 프롬프트 최적화

### 2.3 SLA-Driven 적용

| SLA 유형 | 최소 | 목표 | 도전 |
|----------|------|------|------|
| **Faithfulness** | ≥ 0.70 | ≥ 0.85 | ≥ 0.95 |
| **Answer Relevancy** | ≥ 0.70 | ≥ 0.85 | ≥ 0.95 |
| **Context Precision** | ≥ 0.60 | ≥ 0.80 | ≥ 0.90 |
| **Context Recall** | ≥ 0.60 | ≥ 0.80 | ≥ 0.90 |
| **평가 실행 시간** | < 5min/100건 | < 2min/100건 | < 1min/100건 |
| **비용 (OpenAI)** | < $0.10/건 | < $0.05/건 | < $0.02/건 |

---

## 3. 아키텍처 설계

### 3.1 Hexagonal Architecture (Ports & Adapters)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Domain Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Evaluator  │  │  Dataset    │  │  EvaluationResult       │  │
│  │  (Core)     │  │  (Entity)   │  │  (Entity)               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Ports        │    │  Ports        │    │  Ports        │
│  (Inbound)    │    │  (Outbound)   │    │  (Outbound)   │
│               │    │               │    │               │
│  - CLI        │    │  - LLMPort    │    │  - StoragePort│
│  - HTTP API   │    │  - DatasetPort│    │  - TrackerPort│
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Adapters     │    │  Adapters     │    │  Adapters     │
│               │    │               │    │               │
│  - Typer CLI  │    │  - OpenAI     │    │  - MLflow     │
│  - FastAPI    │    │  - Ollama     │    │  - langfuse   │
│               │    │  - JSON/CSV   │    │  - SQLite     │
└───────────────┘    └───────────────┘    └───────────────┘
```

### 3.2 프로젝트 구조

```
ragas-eval/
├── src/ragas_eval/
│   ├── domain/                    # 비즈니스 로직 (Pure Python)
│   │   ├── entities/              # 데이터 모델
│   │   │   ├── dataset.py         # Dataset, TestCase
│   │   │   └── result.py          # EvaluationResult, RunMetrics
│   │   └── services/
│   │       └── evaluator.py       # 평가 로직
│   │
│   ├── ports/                     # 인터페이스 (Protocol)
│   │   ├── inbound/
│   │   │   └── evaluator_port.py  # 평가 실행 인터페이스
│   │   └── outbound/
│   │       ├── llm_port.py        # LLM 호출 인터페이스
│   │       ├── dataset_port.py    # 데이터셋 로드 인터페이스
│   │       ├── storage_port.py    # 결과 저장 인터페이스
│   │       └── tracker_port.py    # 실험 추적 인터페이스
│   │
│   ├── adapters/                  # 구현체
│   │   ├── inbound/
│   │   │   ├── cli.py             # Typer CLI
│   │   │   └── http.py            # FastAPI (선택)
│   │   └── outbound/
│   │       ├── llm/
│   │       │   ├── openai.py      # OpenAI API
│   │       │   └── ollama.py      # Ollama (로컬)
│   │       ├── dataset/
│   │       │   ├── json_loader.py
│   │       │   └── csv_loader.py
│   │       ├── storage/
│   │       │   └── sqlite.py      # SQLite 저장소
│   │       └── tracker/
│   │           ├── mlflow.py      # MLflow 어댑터
│   │           └── langfuse.py    # langfuse 어댑터
│   │
│   └── config/
│       └── settings.py            # Pydantic Settings
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # 테스트용 데이터셋
│
├── experiments/                   # 실험 레지스트리
│   ├── registry.yaml
│   └── YYYY-MM-DD-experiment-name/
│
├── data/                          # 데이터셋 저장
│   └── datasets/
│
└── docs/
    ├── specs/                     # PRD, 기능 명세
    └── decisions/                 # ADR
```

---

## 4. 핵심 인터페이스 정의 (Spec-First)

### 4.1 데이터셋 스키마

```python
# domain/entities/dataset.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestCase:
    """단일 평가 케이스"""
    id: str
    question: str                    # 사용자 질문
    answer: str                      # LLM 생성 답변
    contexts: list[str]              # 검색된 컨텍스트
    ground_truth: Optional[str]      # 정답 (있는 경우)

@dataclass
class Dataset:
    """평가용 데이터셋"""
    name: str
    version: str
    test_cases: list[TestCase]
    metadata: dict
```

### 4.2 평가 결과 스키마

```python
# domain/entities/result.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class MetricScore:
    """개별 메트릭 점수"""
    name: str                        # faithfulness, answer_relevancy, etc.
    score: float                     # 0.0 ~ 1.0
    threshold: float                 # SLA 임계값
    passed: bool                     # threshold 통과 여부

@dataclass
class TestCaseResult:
    """개별 테스트 케이스 결과"""
    test_case_id: str
    metrics: list[MetricScore]
    tokens_used: int                 # 총 토큰 사용량
    latency_ms: int                  # 응답 시간
    cost_usd: Optional[float]        # 비용 (계산 가능한 경우)

@dataclass
class EvaluationRun:
    """전체 평가 실행 결과"""
    run_id: str
    dataset_name: str
    dataset_version: str
    model_name: str
    started_at: datetime
    finished_at: datetime

    # 개별 결과
    results: list[TestCaseResult]

    # 집계 통계
    total_test_cases: int
    passed_test_cases: int

    # 메트릭별 평균
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float

    # 리소스 사용량
    total_tokens: int
    total_cost_usd: Optional[float]
    total_duration_seconds: float
```

### 4.3 Port 인터페이스

```python
# ports/outbound/llm_port.py
from typing import Protocol

class LLMPort(Protocol):
    """LLM 호출 인터페이스 (OpenAI API 호환)"""

    async def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """OpenAI 호환 chat completion"""
        ...

    def get_token_count(self, text: str) -> int:
        """토큰 수 계산"""
        ...


# ports/outbound/storage_port.py
class StoragePort(Protocol):
    """결과 저장 인터페이스"""

    async def save_run(self, run: EvaluationRun) -> str:
        """평가 실행 결과 저장"""
        ...

    async def get_run(self, run_id: str) -> EvaluationRun:
        """평가 실행 결과 조회"""
        ...

    async def list_runs(
        self,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[EvaluationRun]:
        """평가 실행 목록 조회"""
        ...


# ports/outbound/tracker_port.py
class TrackerPort(Protocol):
    """실험 추적 인터페이스 (MLflow/langfuse 호환)"""

    def start_run(self, run_name: str, tags: dict) -> str:
        """실험 실행 시작"""
        ...

    def log_params(self, params: dict) -> None:
        """파라미터 기록"""
        ...

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """메트릭 기록"""
        ...

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        """아티팩트 저장"""
        ...

    def end_run(self) -> None:
        """실험 실행 종료"""
        ...
```

---

## 5. 기술 스택 선정

### 5.1 실험 추적 도구 비교 (ADR 후보)

| 도구 | 셀프호스팅 | 라이선스 | 토큰 추적 | 비용 추적 | LLM 특화 |
|------|-----------|----------|----------|----------|----------|
| **MLflow** | ✅ Docker | Apache 2.0 | ❌ 수동 | ❌ 수동 | ❌ 범용 |
| **langfuse** | ✅ Docker | MIT | ✅ 자동 | ✅ 자동 | ✅ LLM 특화 |
| **Phoenix (Arize)** | ✅ pip install | ELv2 | ✅ 자동 | ❌ | ✅ LLM 특화 |
| **opik** | ✅ Docker | Apache 2.0 | ✅ 자동 | ✅ 자동 | ✅ LLM 특화 |

**권장: langfuse 또는 opik**
- LLM 평가에 특화된 기능 제공
- 토큰/비용 자동 추적
- 셀프호스팅 가능 (Docker Compose)
- MIT/Apache 2.0 라이선스

### 5.2 권장 기술 스택

| 영역 | 도구 | 근거 |
|------|------|------|
| **언어** | Python 3.12+ | Ragas 네이티브 지원 |
| **패키지 관리** | uv | 빠른 설치, lockfile 지원 |
| **CLI** | Typer | 타입 힌트 기반 CLI |
| **HTTP** | FastAPI | 비동기, OpenAPI 문서화 |
| **설정** | pydantic-settings | 환경 변수, .env 지원 |
| **평가** | Ragas | RAG 평가 표준 |
| **추적** | langfuse/opik | 토큰/비용 자동 추적 |
| **저장** | SQLite → PostgreSQL | 단순 시작, 확장 가능 |
| **테스트** | pytest | TDD |
| **린트** | ruff | 빠른 린터/포매터 |

---

## 6. 구현 계획 (Phase 기반)

### Phase 1: Core MVP

**목표:** 데이터셋 로드 → Ragas 평가 → 결과 저장

```
[ ] 프로젝트 초기화 (uv, pyproject.toml)
[ ] 도메인 엔티티 정의 (Dataset, EvaluationRun)
[ ] Port 인터페이스 정의
[ ] OpenAI 어댑터 구현 (LLMPort)
[ ] JSON 데이터셋 로더 구현 (DatasetPort)
[ ] SQLite 저장소 구현 (StoragePort)
[ ] Ragas 평가 서비스 구현 (Evaluator)
[ ] Typer CLI 구현
[ ] 단위 테스트 작성
```

**산출물:**
```bash
ragas-eval run --dataset data/test.json --model gpt-5-nano
ragas-eval history --limit 10
```

### Phase 2: 실험 추적 통합

**목표:** langfuse/opik 통합으로 자동 추적

```
[ ] langfuse 또는 opik 선정 (ADR 작성)
[ ] TrackerPort 구현
[ ] 토큰 사용량 자동 계산
[ ] 비용 자동 계산
[ ] 대시보드 설정 (Docker Compose)
```

**산출물:**
- 웹 대시보드에서 모든 실행 이력 조회
- 메트릭 추이 그래프
- 비용 분석

### Phase 3: 고급 기능 (선택)

```
[ ] CSV 데이터셋 지원
[ ] Ollama 로컬 LLM 지원
[ ] FastAPI HTTP 인터페이스
[ ] 배치 평가 (병렬 처리)
[ ] 자동 리포트 생성
```

---

## 7. 저장할 정보 상세

### 7.1 실행별 저장 정보

| 카테고리 | 필드 | 설명 |
|----------|------|------|
| **메타데이터** | run_id | 고유 식별자 (UUID) |
| | dataset_name | 데이터셋 이름 |
| | dataset_version | 데이터셋 버전 |
| | model_name | 사용한 모델 (gpt-4o, gpt-5-nano 등) |
| | started_at | 시작 시간 |
| | finished_at | 종료 시간 |
| **집계 점수** | avg_faithfulness | Faithfulness 평균 |
| | avg_answer_relevancy | Answer Relevancy 평균 |
| | avg_context_precision | Context Precision 평균 |
| | avg_context_recall | Context Recall 평균 |
| **리소스** | total_tokens | 총 토큰 사용량 |
| | prompt_tokens | 입력 토큰 |
| | completion_tokens | 출력 토큰 |
| | total_cost_usd | 총 비용 (USD) |
| | duration_seconds | 총 소요 시간 |
| **통계** | total_cases | 전체 테스트 케이스 수 |
| | passed_cases | 통과한 케이스 수 |
| | pass_rate | 통과율 |

### 7.2 개별 문항별 저장 정보

| 필드 | 설명 |
|------|------|
| test_case_id | 테스트 케이스 ID |
| question | 질문 원문 |
| answer | 생성된 답변 |
| contexts | 검색된 컨텍스트 목록 |
| ground_truth | 정답 (있는 경우) |
| faithfulness_score | Faithfulness 점수 |
| answer_relevancy_score | Answer Relevancy 점수 |
| context_precision_score | Context Precision 점수 |
| context_recall_score | Context Recall 점수 |
| tokens_used | 해당 케이스 토큰 사용량 |
| latency_ms | 응답 시간 (ms) |
| cost_usd | 해당 케이스 비용 |

---

## 8. 데이터셋 형식

### 8.1 입력 JSON 형식

```json
{
  "name": "my-rag-dataset",
  "version": "1.0.0",
  "test_cases": [
    {
      "id": "tc-001",
      "question": "What is the capital of France?",
      "answer": "The capital of France is Paris.",
      "contexts": [
        "Paris is the capital and largest city of France.",
        "France is a country in Western Europe."
      ],
      "ground_truth": "Paris"
    }
  ]
}
```

### 8.2 Ragas 호환 형식

내부적으로 Ragas Dataset 형식으로 변환:

```python
from ragas import EvaluationDataset

dataset = EvaluationDataset.from_list([
    {
        "user_input": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieved_contexts": ["Paris is the capital..."],
        "reference": "Paris"
    }
])
```

---

## 9. CLI 사용 예시

```bash
# 평가 실행
ragas-eval run \
  --dataset data/my-dataset.json \
  --model gpt-5-nano \
  --metrics faithfulness,answer_relevancy

# 특정 임계값으로 실행
ragas-eval run \
  --dataset data/my-dataset.json \
  --model gpt-4o \
  --threshold-faithfulness 0.85 \
  --threshold-relevancy 0.80

# 이력 조회
ragas-eval history --limit 20

# 특정 실행 상세 조회
ragas-eval show run-abc123

# 비교 분석
ragas-eval compare run-abc123 run-xyz789

# 리포트 생성
ragas-eval report run-abc123 --format html --output report.html
```

---

## 10. 환경 설정

### 10.1 환경 변수

```bash
# .env
# LLM 설정
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # 또는 로컬 프록시

# 추적 도구 (langfuse 예시)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=http://localhost:3000  # 셀프호스팅

# 저장소
DATABASE_URL=sqlite:///data/evaluations.db
```

### 10.2 Docker Compose (langfuse 셀프호스팅)

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/langfuse
      - NEXTAUTH_URL=http://localhost:3000
      - NEXTAUTH_SECRET=your-secret
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse-data:/var/lib/postgresql/data

volumes:
  langfuse-data:
```

---

## 11. 다음 단계

### 즉시 수행 (Spec-First)

1. **ADR 작성**: 추적 도구 선정 (langfuse vs opik vs MLflow)
2. **PRD 작성**: Phase 1 상세 요구사항
3. **인터페이스 확정**: Port 인터페이스 최종 검토

### 프로젝트 생성

```bash
# 새 프로젝트 생성
mkdir ragas-eval && cd ragas-eval
uv init
uv add ragas openai pydantic typer structlog
uv add --dev pytest ruff

# 기본 구조 생성
mkdir -p src/ragas_eval/{domain,ports,adapters,config}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p docs/{specs,decisions}
mkdir -p data/datasets
mkdir -p experiments
```

---

## 12. 체크리스트

### 설계 시 체크 (병렬 개발 가능성)

- [x] 다른 모듈과 독립적으로 개발 가능한가? → Hexagonal 아키텍처로 분리
- [x] 인터페이스가 명확히 정의되었는가? → Port 인터페이스 정의 완료
- [x] Mock으로 독립 테스트가 가능한가? → Port 기반으로 Mock 가능

### 6대 원칙 준수 확인

- [x] **Spec-First**: 인터페이스와 스키마 먼저 정의
- [x] **YAGNI**: MVP 범위 명확히 제한
- [x] **Testable**: TDD 가능한 구조
- [x] **SLA-Driven**: 메트릭별 임계값 정의
- [x] **Experiment-Driven**: experiments/ 폴더 구조 정의
- [x] **Evidence-Based**: 모든 결과 저장 및 통계 제공

---

## 라이선스

Apache License 2.0

---

*이 문서는 GyeotAI 개발 정책을 기반으로 작성되었습니다.*
