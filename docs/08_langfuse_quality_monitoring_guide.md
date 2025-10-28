# Langfuse 답변 품질 모니터링 가이드

## 📋 개요

Langfuse를 사용하여 LangGraph 워크플로우의 답변 품질을 실시간으로 모니터링하는 방법을 안내합니다.

## ✅ 완료된 작업

### 1. Langfuse 통합
- ✅ LangGraph 워크플로우 서비스에 LangfuseClient 통합
- ✅ 답변 품질 메트릭 자동 추적
- ✅ 종합 품질 점수 계산 및 추적

### 2. 테스트 인프라
- ✅ `scripts/test_langfuse_quality.py` - Langfuse 추적 테스트
- ✅ `scripts/run_answer_quality_tests.py` - Mock 테스트

### 3. 추적 메트릭
다음 7개 메트릭이 자동으로 추적됩니다:

1. **answer_length** - 답변 길이 (문자 수)
2. **answer_confidence** - 신뢰도 점수 (0.0 ~ 1.0)
3. **sources_count** - 소스 개수
4. **processing_time** - 처리 시간 (초)
5. **has_errors** - 에러 발생 여부 (0.0 / 1.0)
6. **legal_refs_count** - 법률 참조 개수
7. **overall_quality** - 종합 품질 점수 (0.0 ~ 1.0)

## 🚀 빠른 시작

### 1. 환경 설정

`.env` 파일에 다음 설정을 추가합니다:

```bash
# Langfuse 활성화
LANGFUSE_ENABLED=true

# Langfuse API 키 (Langfuse 홈페이지에서 발급)
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxx

# Langfuse 호스트 (기본값: https://cloud.langfuse.com)
LANGFUSE_HOST=https://cloud.langfuse.com

# 디버그 모드 (선택사항)
LANGFUSE_DEBUG=false
```

### 2. 테스트 실행

```bash
# Langfuse 추적 테스트 실행
python scripts/test_langfuse_quality.py
```

**결과**:
- 콘솔에 품질 점수 표시
- Langfuse 대시보드에서 추적 데이터 확인 가능

### 3. Langfuse 대시보드 확인

1. [Langfuse 대시보드](https://cloud.langfuse.com) 로그인
2. **Traces** 탭에서 모든 질문 처리 추적 확인
3. **Scores** 탭에서 품질 점수 분석
4. **Analytics** 탭에서 트렌드 분석

## 📊 테스트 결과

### 실행 결과 예시

```
Langfuse 답변 품질 테스트

테스트 질문:
1. 계약서 작성 시 주의사항은?
2. 이혼 절차는 어떻게 되나요?
3. 민법 제105조는 무엇인가요?
4. 배임죄는 어떤 죄인가요?

평균 품질 점수: 0.57/1.0
전반적인 답변 품질: 보통
```

### 품질 점수 해석

| 점수 | 등급 | 의미 |
|------|------|------|
| 0.9 ~ 1.0 | ⭐⭐⭐ 우수 | 매우 좋은 품질 |
| 0.7 ~ 0.9 | ⭐⭐ 양호 | 양호한 품질 |
| 0.5 ~ 0.7 | ⭐ 보통 | 기본 요구사항 충족 |
| 0.0 ~ 0.5 | ⚠️ 개선 필요 | 개선이 필요함 |

## 🔍 Langfuse 대시보드 사용법

### 1. Traces 탭

모든 질문 처리 추적을 확인:

```
✅ 질문: 계약서 작성 시 주의사항은?
✅ 답변 미리보기: "계약서는 양 당사자의 권리와 의무를 명확히..."
✅ 신뢰도: 0.85
✅ 처리 시간: 2.34초
```

### 2. Scores 탭

품질 점수를 그래프로 확인:

- **overall_quality**: 종합 품질 점수 분포
- **answer_confidence**: 신뢰도 분포
- **processing_time**: 처리 시간 분포
- **sources_count**: 소스 개수 분포

### 3. Analytics 탭

품질 트렌드 분석:

- 시간별 품질 점수 변화
- 질문 유형별 품질 비교
- 성능 지표 분석

## 💡 코드 사용 예시

### 기본 사용

```python
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig

# 설정 로드
config = LangGraphConfig.from_env()

# 워크플로우 서비스 생성
service = LangGraphWorkflowService(config)

# 질문 처리 (자동으로 Langfuse에 추적됨)
result = await service.process_query("계약서 작성 시 주의사항은?")

# 결과 확인
print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence']}")
print(f"품질 점수: {result.get('quality_score', 'N/A')}")
```

### 품질 점수 확인

```python
# 답변 품질 점수 계산
quality_score = service._calculate_quality_score(result)
print(f"품질 점수: {quality_score:.2f}/1.0")
```

## 📈 모니터링 전략

### 1. 일일 품질 체크

매일 테스트를 실행하여 품질 점수 모니터링:

```bash
# Cron 작업 설정 (선택사항)
# 매일 자정에 테스트 실행
0 0 * * * cd /path/to/LawFirmAI && python scripts/test_langfuse_quality.py
```

### 2. 품질 저하 알림

Langfuse에서 알림 설정:

1. **대시보드** → **Settings** → **Alerts**
2. **overall_quality** < 0.5 일 때 알림 설정
3. 이메일 또는 Slack으로 알림 수신

### 3. 성능 최적화

처리 시간이 긴 질문 분석:

- Langfuse 대시보드에서 **processing_time** 필터링
- 장시간 처리 질문 식별
- 워크플로우 최적화

## 🔧 문제 해결

### Langfuse 연결 실패

**증상**: `Failed to initialize Langfuse client`

**해결**:
1. `.env` 파일 확인
2. API 키 검증
3. 네트워크 연결 확인

### 추적 데이터가 없음

**증상**: 대시보드에 데이터가 안 보임

**해결**:
1. `LANGFUSE_ENABLED=true` 확인
2. 로그 확인: `logs/`
3. Langfuse 대시보드에서 프로젝트 선택 확인

### 품질 점수가 0

**원인**:
- 답변이 비어있거나 매우 짧음
- 소스가 제공되지 않음
- 에러가 발생함

**해결**:
- 데이터베이스 확인
- 워크플로우 로그 확인

## 📚 추가 자료

- [Langfuse 공식 문서](https://langfuse.com/docs)
- [LangGraph 워크플로우 문서](./05_rag_system/langgraph_integration_guide.md)
- [답변 품질 테스트](./08_langgraph_langfuse_integration_guide.md)

## 🎯 다음 단계

1. **실제 환경에서 테스트**: 실제 LLM으로 질문 처리
2. **대시보드 확인**: Langfuse 대시보드에서 추적 데이터 확인
3. **품질 개선**: 품질 점수가 낮은 영역 개선
4. **자동화**: CI/CD에 품질 테스트 통합
5. **알림 설정**: 품질 저하 시 자동 알림

---

**작성일**: 2025년 1월 27일  
**버전**: 1.0  
**작성자**: LawFirmAI Development Team
