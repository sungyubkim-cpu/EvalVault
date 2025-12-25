# Ollama 지원 구현 계획

> 폐쇄망 환경에서 로컬 LLM(Ollama)을 사용한 RAG 평가 지원

## 1. 개요

### 목표
- 외부 API 호출 없이 완전한 오프라인 RAG 평가 지원
- OpenAI ↔ Ollama 간 간편한 전환
- 기존 아키텍처(Hexagonal) 유지

### Ragas + Ollama 통합 방식
Ragas는 두 가지 방식으로 Ollama를 지원합니다:

```python
# 방식 1: OpenAI 호환 API (권장 - 안정적)
from openai import OpenAI
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
llm = llm_factory("mistral", provider="openai", client=client)

# 방식 2: 직접 Ollama 프로바이더
llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")
```

**권장: 방식 1 (OpenAI 호환 API)**
- 기존 OpenAIAdapter 코드 재사용 가능
- 더 안정적이고 테스트됨
- Ollama의 OpenAI 호환 API가 잘 동작함

---

## 2. 구현 범위

### 2.1 Settings 확장 (`src/evalvault/config/settings.py`)

```python
class Settings(BaseSettings):
    # LLM Provider Selection
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' or 'ollama'"
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model name for evaluation"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model"
    )
    ollama_timeout: int = Field(
        default=120,
        description="Ollama request timeout in seconds"
    )
```

### 2.2 OllamaAdapter 구현 (`src/evalvault/adapters/outbound/llm/ollama_adapter.py`)

```python
class OllamaAdapter(LLMPort):
    """Ollama LLM adapter for air-gapped environments.

    Uses OpenAI-compatible API for maximum compatibility with Ragas.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_name = settings.ollama_model
        self._embedding_model_name = settings.ollama_embedding_model

        # Token usage tracker
        self._token_usage = TokenUsage()

        # Create OpenAI-compatible client pointing to Ollama
        self._client = TokenTrackingAsyncOpenAI(
            usage_tracker=self._token_usage,
            api_key="ollama",  # Ollama doesn't require real key
            base_url=f"{settings.ollama_base_url}/v1",
            timeout=settings.ollama_timeout,
        )

        # Create Ragas LLM using OpenAI provider with Ollama backend
        self._ragas_llm = llm_factory(
            model=self._model_name,
            provider="openai",
            client=self._client,
        )

        # Create Ragas embeddings (Ollama OpenAI-compatible)
        self._ragas_embeddings = RagasOpenAIEmbeddings(
            model=self._embedding_model_name,
            client=self._client,
        )

    def get_model_name(self) -> str:
        return f"ollama/{self._model_name}"

    def as_ragas_llm(self):
        return self._ragas_llm

    def as_ragas_embeddings(self):
        return self._ragas_embeddings

    # Token tracking methods (same as OpenAIAdapter)
    ...
```

### 2.3 LLM Factory 함수 추가 (`src/evalvault/adapters/outbound/llm/__init__.py`)

```python
def get_llm_adapter(settings: Settings) -> LLMPort:
    """Factory function to create appropriate LLM adapter.

    Args:
        settings: Application settings

    Returns:
        LLMPort implementation based on settings.llm_provider
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from evalvault.adapters.outbound.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(settings)
    elif provider == "ollama":
        from evalvault.adapters.outbound.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
```

### 2.4 CLI 수정 (`src/evalvault/adapters/inbound/cli.py`)

```python
@app.command()
def run(
    dataset: Path = ...,
    metrics: str = ...,
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: 'openai' or 'ollama' (overrides settings).",
    ),
    model: str | None = typer.Option(...),
    ...
):
    # Load settings
    settings = Settings()

    # Override provider if specified
    if provider:
        settings.llm_provider = provider

    # Override model based on provider
    if model:
        if settings.llm_provider == "ollama":
            settings.ollama_model = model
        else:
            settings.openai_model = model

    # Validate provider-specific requirements
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set.")
        raise typer.Exit(1)

    # Use factory to get appropriate adapter
    from evalvault.adapters.outbound.llm import get_llm_adapter
    llm = get_llm_adapter(settings)

    console.print(f"Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(f"Model: [cyan]{llm.get_model_name()}[/cyan]\n")
    ...
```

### 2.5 Config 명령 업데이트

```python
@app.command()
def config():
    """Show current configuration."""
    settings = Settings()

    # LLM Provider section
    table.add_row("LLM Provider", settings.llm_provider)

    if settings.llm_provider == "openai":
        api_key_status = "[green]Set[/green]" if settings.openai_api_key else "[red]Not set[/red]"
        table.add_row("OpenAI API Key", api_key_status)
        table.add_row("OpenAI Model", settings.openai_model)
    elif settings.llm_provider == "ollama":
        table.add_row("Ollama URL", settings.ollama_base_url)
        table.add_row("Ollama Model", settings.ollama_model)
        table.add_row("Ollama Embedding", settings.ollama_embedding_model)
```

---

## 3. 환경 설정 예시

### `.env` 파일 (폐쇄망용)

```bash
# LLM Provider: 'openai' or 'ollama'
LLM_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120

# Langfuse (선택 - 폐쇄망에서는 self-hosted 필요)
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=http://internal-langfuse:3000
```

### CLI 사용 예시

```bash
# 기본 (설정 파일 기준)
evalvault run data.json --metrics faithfulness

# Provider 명시적 지정
evalvault run data.json --metrics faithfulness --provider ollama

# 모델 오버라이드
evalvault run data.json --metrics faithfulness --provider ollama --model mistral

# OpenAI로 전환
evalvault run data.json --metrics faithfulness --provider openai
```

---

## 4. 테스트 계획

### 4.1 단위 테스트

| 테스트 | 파일 | 설명 |
|--------|------|------|
| OllamaAdapter 생성 | `tests/unit/test_ollama_adapter.py` | Mock 기반 어댑터 초기화 |
| get_llm_adapter 팩토리 | `tests/unit/test_llm_factory.py` | Provider별 올바른 어댑터 반환 |
| Settings 검증 | `tests/unit/test_settings.py` | Ollama 설정 파싱 |

### 4.2 통합 테스트

| 테스트 | 마커 | 설명 |
|--------|------|------|
| Ollama 연결 테스트 | `@pytest.mark.requires_ollama` | 실제 Ollama 서버 연결 |
| Ollama 평가 테스트 | `@pytest.mark.requires_ollama` | faithfulness 메트릭 실행 |
| 임베딩 테스트 | `@pytest.mark.requires_ollama` | answer_relevancy 메트릭 |

### 4.3 E2E 테스트

```python
@pytest.mark.requires_ollama
async def test_real_evaluation_with_ollama(korean_json, e2e_results_db):
    """Ollama를 사용한 실제 평가 테스트."""
    settings = Settings()
    settings.llm_provider = "ollama"

    llm = get_llm_adapter(settings)
    evaluator = RagasEvaluator()

    dataset = get_loader(korean_json).load(korean_json)

    run = await evaluator.evaluate(
        dataset=dataset,
        metrics=["faithfulness"],
        llm=llm,
    )

    assert run is not None
    assert len(run.results) == len(dataset)
```

---

## 5. 권장 Ollama 모델

### LLM 모델 (평가용)

| 모델 | 크기 | VRAM | 특징 |
|------|------|------|------|
| `llama3.2` | 3B | 4GB | 경량, 빠른 추론 |
| `llama3.2:1b` | 1B | 2GB | 최소 사양 |
| `mistral` | 7B | 8GB | 균형잡힌 성능 |
| `qwen2.5` | 7B | 8GB | 다국어 지원 우수 |
| `gemma2` | 9B | 10GB | 고품질 출력 |

### 임베딩 모델

| 모델 | 차원 | 특징 |
|------|------|------|
| `nomic-embed-text` | 768 | 경량, 빠름 |
| `mxbai-embed-large` | 1024 | 고품질 |
| `bge-m3` | 1024 | 다국어 지원 |

### 설치 명령

```bash
# LLM 모델
ollama pull llama3.2
ollama pull mistral

# 임베딩 모델
ollama pull nomic-embed-text
```

---

## 6. 구현 순서

### Phase 1: 핵심 구현
1. [ ] Settings에 Ollama 설정 추가
2. [ ] OllamaAdapter 구현
3. [ ] get_llm_adapter 팩토리 함수 구현
4. [ ] 단위 테스트 작성

### Phase 2: CLI 통합
5. [ ] CLI run 명령에 --provider 옵션 추가
6. [ ] CLI config 명령 업데이트
7. [ ] CLI 도움말/문서 업데이트

### Phase 3: 테스트 & 문서
8. [ ] 통합 테스트 작성 (Ollama 마커)
9. [ ] E2E 테스트 작성
10. [ ] README/docs 업데이트
11. [ ] .env.example 업데이트

---

## 7. 주의사항

### 폐쇄망 고려사항
- Ollama 바이너리와 모델을 사전에 오프라인 설치 필요
- Python 패키지도 오프라인 설치 필요 (wheels)
- Langfuse 사용 시 self-hosted 버전 필요

### 성능 고려사항
- Ollama는 OpenAI 대비 느릴 수 있음
- GPU 없이 CPU만 사용 시 상당히 느림
- 대용량 데이터셋의 경우 타임아웃 조정 필요

### 호환성
- Ollama 0.1.0+ 필요 (OpenAI 호환 API 지원)
- 일부 메트릭은 특정 모델에서 품질이 낮을 수 있음
- 한국어 평가 시 다국어 모델 권장 (qwen2.5, gemma2)

---

## 8. 참고 자료

- [Ragas Quick Start](https://docs.ragas.io/en/stable/getstarted/quickstart/)
- [Ollama OpenAI Compatibility](https://ollama.com/blog/openai-compatibility)
- [Ragas + Ollama Issue #1857](https://github.com/explodinggradients/ragas/issues/1857)
- [LangChain + Ollama RAG Guide](https://blog.sudhirnakka.com/2025/build_rag_ollama)
