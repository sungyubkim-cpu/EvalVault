# EvalVault

> RAG(Retrieval-Augmented Generation) 시스템 품질 평가를 위한 올인원 솔루션

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ragas](https://img.shields.io/badge/Ragas-v1.0-green.svg)](https://docs.ragas.io/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

---

## 핵심 기능

- **6가지 평가 메트릭**: Ragas v1.0 기반 표준화된 RAG 평가
- **다중 데이터 포맷**: JSON, CSV, Excel 지원
- **자동 결과 저장**: SQLite (로컬) + Langfuse (클라우드/셀프호스팅)
- **CLI 인터페이스**: 간편한 명령줄 도구
- **Hexagonal Architecture**: 확장 가능한 포트/어댑터 구조

## 빠른 시작

```bash
# 1. 설치
git clone https://github.com/ntts9990/EvalVault.git
cd EvalVault
uv pip install -e ".[dev]"

# 2. 환경 설정
cp .env.example .env
echo "OPENAI_API_KEY=sk-your-key" >> .env

# 3. 평가 실행
evalvault run tests/fixtures/e2e/insurance_qa_korean.json --metrics faithfulness

# 4. 결과 확인
evalvault history
```

## 지원 메트릭

| 메트릭 | 설명 | Ground Truth |
|--------|------|--------------|
| `faithfulness` | 답변이 컨텍스트에 충실한지 (환각 감지) | ❌ |
| `answer_relevancy` | 답변이 질문과 관련있는지 | ❌ |
| `context_precision` | 검색된 컨텍스트의 정밀도 | ✅ |
| `context_recall` | 필요한 정보가 검색되었는지 | ✅ |
| `factual_correctness` | 답변이 정답과 사실적으로 일치하는지 | ✅ |
| `semantic_similarity` | 답변과 정답의 의미적 유사도 | ✅ |

## CLI 명령어

```bash
# 평가 실행
evalvault run data.json --metrics faithfulness,answer_relevancy

# 모든 메트릭으로 평가
evalvault run data.json --metrics faithfulness,answer_relevancy,context_precision,context_recall,factual_correctness,semantic_similarity

# Langfuse 연동
evalvault run data.json --metrics faithfulness --langfuse

# 히스토리 조회
evalvault history --limit 10

# 결과 비교
evalvault compare <run_id1> <run_id2>

# 결과 내보내기
evalvault export <run_id> -o result.json
```

## 데이터 형식

### JSON (권장)

```json
{
  "name": "my-dataset",
  "version": "1.0.0",
  "test_cases": [
    {
      "id": "tc-001",
      "question": "보험금은 얼마인가요?",
      "answer": "1억원입니다.",
      "contexts": ["사망보험금은 1억원입니다."],
      "ground_truth": "1억원"
    }
  ]
}
```

### CSV

```csv
id,question,answer,contexts,ground_truth
tc-001,"보험금은?","1억원입니다.","[""사망보험금은 1억원""]","1억원"
```

## 환경 설정

```bash
# .env 파일
OPENAI_API_KEY=sk-...                    # 필수
OPENAI_MODEL=gpt-4o-mini                 # 선택 (기본: gpt-4o-mini)

# Langfuse 연동 (선택)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 문서

**[상세 사용자 가이드](docs/README.md)** - 설치, 설정, 메트릭 설명, 문제 해결 등

## 아키텍처

```
src/evalvault/
├── domain/           # 비즈니스 로직
│   ├── entities/     # TestCase, Dataset, EvaluationRun
│   ├── services/     # RagasEvaluator
│   └── metrics/      # 커스텀 메트릭
├── ports/            # 인터페이스
│   ├── inbound/      # EvaluatorPort
│   └── outbound/     # LLMPort, StoragePort, TrackerPort
├── adapters/         # 구현체
│   ├── inbound/      # CLI (Typer)
│   └── outbound/     # OpenAI, SQLite, Langfuse, ...
└── config/           # Settings
```

## 개발

```bash
# 테스트 실행
pytest tests/ -v

# E2E 테스트 (실제 API 호출)
pytest tests/integration/test_e2e_scenarios.py -v

# 린트
ruff check src/
ruff format src/
```

## 버전

- **v0.3.0** (2025-12-24): Phase 6 완료, 6개 메트릭, Ragas v1.0 호환
- **v0.2.0** (2024-12-24): SQLite 저장, 히스토리 기능
- **v0.1.0** (2024-12-24): 초기 릴리스

## 라이선스

Apache 2.0 - See [LICENSE.md](LICENSE.md)

---

<div align="center">

**EvalVault** - RAG 평가의 새로운 기준

</div>
