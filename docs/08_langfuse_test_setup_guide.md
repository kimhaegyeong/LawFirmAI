# Langfuse 테스트 설정 가이드

## 개요

`scripts/run_answer_quality_tests.py` 스크립트는 실제 워크플로우를 실행하여 답변 품질을 테스트하고, Langfuse를 통해 모니터링합니다.

## 변경 사항

### 1. Mock 제거
- 이전: Mock을 사용하여 가짜 응답 반환
- 현재: **실제 워크플로우를 실행**하여 실제 답변 생성

### 2. Langfuse 통합 개선
- 이전: Langfuse 로그만 남김
- 현재: **실제 Langfuse API를 호출**하여 trace와 score를 전송

### 3. .env 설정 존중
- 이전: 강제로 Langfuse 활성화/비활성화
- 현재: **.env 파일의 설정을 따름**

## 테스트 구성

이 스크립트는 두 가지 테스트를 실행합니다:

### 1. 기본 답변 품질 테스트
- `.env` 파일의 `LANGFUSE_ENABLED` 설정을 따름
- Langfuse 활성화 여부와 관계없이 실행
- 답변 품질 검증에 집중

### 2. Langfuse 추적 포함 테스트
- `.env`에서 `LANGFUSE_ENABLED=true`인 경우에만 실행
- Langfuse 인증 정보가 있어야 실행됨
- Langfuse 대시보드에 메트릭 전송

## 환경 변수 설정

### 1. .env 파일 생성

`.env` 파일이 없으면 `env.example`을 복사하여 생성하세요:

```bash
cp env.example .env
```

### 2. Langfuse 인증 정보 추가

`.env` 파일에 다음을 추가하거나 수정하세요:

```env
# Langfuse 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=sk-xxxxx  # Langfuse 대시보드에서 확인
LANGFUSE_PUBLIC_KEY=pk-xxxxx  # Langfuse 대시보드에서 확인
LANGFUSE_HOST=https://cloud.langfuse.com
```

#### Langfuse 키 확인 방법:
1. [Langfuse Cloud](https://cloud.langfuse.com)에 로그인
2. Settings > API Keys에서 Secret Key와 Public Key 확인

### 3. 필수 설정

다음 설정도 확인하세요:

```env
# LLM 설정
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here

# Ollama 설정 (백업용)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

# LangGraph 설정
LANGGRAPH_ENABLED=true
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints/langgraph.db
```

## 테스트 실행

### 전체 테스트 실행

```bash
python scripts/run_answer_quality_tests.py
```

이 스크립트는 다음 두 테스트를 실행합니다:

#### 1. 기본 답변 품질 테스트
- `.env` 파일의 설정을 따름
- Langfuse 활성화 여부와 관계없이 실행
- 답변의 품질, 신뢰도, 소스 개수 등 검증

#### 2. Langfuse 추적 포함 테스트
- `LANGFUSE_ENABLED=true`인 경우에만 실행
- Langfuse 인증 정보가 있어야 실행
- Langfuse 대시보드에 trace와 score 전송

### 시나리오별 동작

#### 시나리오 A: Langfuse 비활성화 (LANGFUSE_ENABLED=false)
```
✓ 기본 답변 품질 테스트 실행
⚠️  Langfuse가 비활성화되어 있어 두 번째 테스트 건너뜀
```

#### 시나리오 B: Langfuse 활성화, 인증 정보 없음
```
✓ 기본 답변 품질 테스트 실행
⚠️  Langfuse 인증 정보가 없어 두 번째 테스트 건너뜀
```

#### 시나리오 C: Langfuse 활성화, 인증 정보 있음
```
✓ 기본 답변 품질 테스트 실행
✓ Langfuse 추적 포함 테스트 실행
✓ Langfuse 대시보드에 데이터 전송
```

## 추적되는 메트릭

Langfuse에서 다음 메트릭들을 추적합니다:

### Trace 정보
- **name**: `answer_quality_tracking`
- **input**: 질문(query), 답변 길이(answer_length)
- **output**: 전체 품질 점수(overall_quality), 신뢰도(confidence)
- **metadata**: 소스 개수, 법률 참조 개수, 처리 시간, 에러 여부

### Score 메트릭
1. **answer_quality_score**: 종합 품질 점수 (0.0~1.0)
2. **answer_confidence**: 신뢰도 점수 (0.0~1.0)
3. **sources_count**: 소스 개수
4. **legal_references_count**: 법률 참조 개수
5. **processing_time**: 처리 시간 (초)
6. **has_errors**: 에러 발생 여부 (0.0 또는 1.0)

## Langfuse 대시보드에서 확인

1. [Langfuse Cloud](https://cloud.langfuse.com)에 로그인
2. Traces 메뉴에서 최근 테스트 결과 확인
3. "answer_quality_tracking" trace를 클릭하여 상세 정보 확인
4. Scores 탭에서 메트릭 점수 확인

## 문제 해결

### 1. "Langfuse 인증 정보가 설정되지 않았습니다"

**원인**: `.env` 파일에 Langfuse 키가 없음

**해결**: `.env` 파일에 다음 추가:
```env
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=sk-xxxxx
LANGFUSE_PUBLIC_KEY=pk-xxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 2. "Langfuse 클라이언트가 활성화되지 않았습니다"

**원인**: Langfuse 라이브러리가 설치되지 않았거나 키가 잘못됨

**해결**:
```bash
pip install langfuse
```

키가 올바른지 확인:
```bash
echo $LANGFUSE_SECRET_KEY
echo $LANGFUSE_PUBLIC_KEY
```

### 3. "Failed to track answer quality metrics"

**원인**: Langfuse 서버 연결 실패

**해결**:
- 인터넷 연결 확인
- Langfuse 키 유효성 확인
- Firewall 설정 확인
- `LANGFUSE_DEBUG=true`로 설정하여 상세 로그 확인

## 예상 출력

```
================================================================================
기본 답변 품질 테스트 (Langfuse 비활성화)
================================================================================
✓ 설정 확인
  - Langfuse: False
  - Checkpoint DB: ./data/checkpoints/langgraph.db
✓ 워크플로우 서비스 초기화 완료

실행 중: 계약서 작성 시 주의사항은?

✅ 답변 길이: 234자
✅ 신뢰도: 0.85
✅ 소스 개수: 3
✅ 법률 참조 개수: 2
✅ 에러 없음: True
✅ 처리 시간: 2.34초

✅ 모든 검증 통과!

================================================================================
Langfuse 추적 포함 답변 품질 테스트
================================================================================
✓ Langfuse 설정 확인
  - Enabled: True
  - Host: https://cloud.langfuse.com
✓ Langfuse 클라이언트 활성화됨

실행 중: 계약서 작성 시 주의사항은?
Trace created: abc123-def456-ghi789

✅ 답변 길이: 234자
✅ 신뢰도: 0.85
✅ 소스 개수: 3
✅ 법률 참조 개수: 2
✅ 에러 없음: True
✅ 처리 시간: 2.34초

✅ 모든 검증 통과!

================================================================================
✅ 모든 답변 품질 테스트 통과!
================================================================================
```

## 다음 단계

1. ✅ 환경 변수 설정
2. ✅ 테스트 실행
3. ✅ Langfuse 대시보드에서 결과 확인
4. ✅ 메트릭 분석 및 품질 개선
