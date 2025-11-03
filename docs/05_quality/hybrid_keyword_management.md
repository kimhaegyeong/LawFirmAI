# 하이브리드 키워드 관리 시스템

## 개요

하이브리드 키워드 관리 시스템은 기존 데이터베이스와 AI 모델을 결합하여 법률 도메인별 키워드를 동적으로 관리하는 시스템입니다. 하드코딩된 키워드 대신 확장 가능하고 지능적인 키워드 관리가 가능합니다.

## 🎯 주요 특징

### 1. 다중 소스 통합
- **데이터베이스 우선**: 기존 법률 용어 사전에서 키워드 로드
- **AI 자동 확장**: 부족한 도메인에 대해 Gemini API로 키워드 확장
- **폴백 시스템**: 모든 방법 실패 시 기본 키워드 제공

### 2. 지능형 캐싱
- **메모리 캐시**: 빠른 접근을 위한 인메모리 캐시
- **파일 캐시**: 영구 저장을 위한 JSON 파일 캐시
- **TTL 관리**: 24시간 자동 만료로 최신성 보장

### 3. 확장 전략 선택
- **DATABASE_ONLY**: 데이터베이스만 사용
- **AI_ONLY**: AI 모델만 사용
- **HYBRID**: 데이터베이스 + AI 통합 (권장)
- **FALLBACK**: 기본 키워드로 폴백

## 🏗️ 시스템 아키텍처

```
HybridKeywordManager
├── KeywordDatabaseLoader     # 데이터베이스에서 키워드 로드
├── AIKeywordGenerator        # AI 모델을 사용한 키워드 확장
├── KeywordCache              # 메모리 + 파일 캐시 관리
└── DomainSpecificExtractor   # 도메인별 용어 추출기
```

## 📊 성능 지표

### 도메인별 키워드 현황 (2025-10-19 기준)

| 도메인 | 키워드 수 | 소스 | 신뢰도 | 확장 여부 |
|--------|-----------|------|--------|-----------|
| 민사법 | 101개 | 하이브리드 | 0.90 | - |
| 형사법 | 52개 | 하이브리드 | 0.90 | - |
| 가족법 | 58개 | 하이브리드 | 0.90 | - |
| 상사법 | 41개 | 하이브리드 | 0.90 | - |
| 노동법 | 75개 | 하이브리드 | 0.90 | - |
| 부동산법 | 36개 | 하이브리드 | 0.90 | - |
| 지적재산권법 | 25개 | AI 확장 | 0.80 | ✅ |
| 세법 | 29개 | AI 확장 | 0.80 | ✅ |
| 민사소송법 | 42개 | 하이브리드 | 0.90 | - |
| 형사소송법 | 22개 | AI 확장 | 0.80 | ✅ |
| 기타/일반 | 660개 | 하이브리드 | 0.90 | - |

### 캐시 성능
- **메모리 캐시**: 11개 도메인
- **파일 캐시**: 11개 파일
- **총 캐시 크기**: 28,289 bytes
- **평균 로드 시간**: 0.015초

## 🚀 사용법

### 기본 사용법

```python
from source.services.domain_specific_extractor import DomainSpecificExtractor, LegalDomain
from source.services.hybrid_keyword_manager import ExpansionStrategy

# 하이브리드 키워드 추출기 초기화
extractor = DomainSpecificExtractor(
    data_dir="data",
    cache_dir="data/cache",
    min_keyword_threshold=20,
    expansion_strategy=ExpansionStrategy.HYBRID
)

# 도메인 분류
primary_domain, confidence = await extractor.get_primary_domain(text)
print(f"주요 도메인: {primary_domain.value} (신뢰도: {confidence:.3f})")

# 키워드 확장 (필요시)
expanded = await extractor.expand_domain_keywords_if_needed(LegalDomain.CIVIL_LAW)
```

### 확장 전략 변경

```python
# 데이터베이스만 사용
extractor.set_expansion_strategy(ExpansionStrategy.DATABASE_ONLY)

# AI만 사용
extractor.set_expansion_strategy(ExpansionStrategy.AI_ONLY)

# 하이브리드 (권장)
extractor.set_expansion_strategy(ExpansionStrategy.HYBRID)
```

### 통계 조회

```python
# 도메인별 통계
stats = await extractor.get_domain_statistics()
for domain, stat in stats.items():
    print(f"{domain}: {stat['keyword_count']}개 키워드")

# 캐시 통계
cache_stats = extractor.get_cache_statistics()
print(f"메모리 캐시: {cache_stats['memory_cache_count']}개")
```

## 🔧 컴포넌트 상세

### 1. KeywordDatabaseLoader

기존 데이터베이스에서 키워드를 로드하는 컴포넌트입니다.

**지원 데이터베이스:**
- `comprehensive_legal_term_dictionary.json`
- `legal_term_dictionary.json`
- `legal_terms_database.json`

**주요 기능:**
- 자동 도메인 매핑
- 동의어 및 관련 용어 통합
- 중복 제거 및 정렬

### 2. AIKeywordGenerator

Gemini API를 사용하여 키워드를 확장하는 컴포넌트입니다.

**확장 방식:**
- 직접 관련 용어
- 하위/상위 개념
- 실무 용어
- 법령/판례 용어
- 동의어/유사어

**품질 관리:**
- 신뢰도 점수 계산
- 폴백 확장 규칙
- 오류 처리 및 복구

### 3. KeywordCache

메모리와 파일을 활용한 이중 캐시 시스템입니다.

**캐시 레벨:**
1. **메모리 캐시**: 가장 빠른 접근
2. **파일 캐시**: 영구 저장
3. **데이터베이스**: 최종 소스

**캐시 관리:**
- TTL 기반 자동 만료
- 수동 무효화
- 만료된 캐시 자동 정리

### 4. HybridKeywordManager

전체 시스템을 통합 관리하는 메인 컴포넌트입니다.

**주요 기능:**
- 다중 소스 통합
- 확장 전략 관리
- 성능 최적화
- 오류 처리

## 📈 성능 최적화

### 1. 캐싱 전략
- **L1 캐시**: 메모리 (가장 빠름)
- **L2 캐시**: 파일 (영구 저장)
- **L3 소스**: 데이터베이스 (최신 데이터)

### 2. 지연 로딩
- 필요할 때만 키워드 로드
- 도메인별 개별 로딩
- 병렬 처리 지원

### 3. 메모리 관리
- 불필요한 데이터 즉시 해제
- 캐시 크기 제한
- 정기적인 가비지 컬렉션

## 🛠️ 설정 옵션

### 환경 변수

```bash
# Gemini API 키 (AI 확장용)
GOOGLE_API_KEY=your_google_api_key

# 데이터 디렉토리
DATA_DIR=data

# 캐시 디렉토리
CACHE_DIR=data/cache

# 최소 키워드 임계값
MIN_KEYWORD_THRESHOLD=20
```

### 설정 파일

```python
# config/hybrid_keyword_config.yaml
hybrid_keyword_manager:
  data_dir: "data"
  cache_dir: "data/cache"
  min_keyword_threshold: 20
  expansion_strategy: "HYBRID"
  cache_ttl_hours: 24
  ai_expansion:
    max_keywords_per_domain: 50
    confidence_threshold: 0.7
```

## 🔍 문제 해결

### 자주 발생하는 문제

1. **AI 확장 실패**
   - `GOOGLE_API_KEY` 확인
   - 네트워크 연결 상태 확인
   - API 할당량 확인

2. **캐시 오류**
   - 캐시 디렉토리 권한 확인
   - 디스크 공간 확인
   - 캐시 파일 손상 시 삭제

3. **성능 저하**
   - 캐시 통계 확인
   - 메모리 사용량 모니터링
   - 불필요한 캐시 정리

### 로그 확인

```python
import logging
logging.basicConfig(level=logging.INFO)

# 상세 로그 확인
logger = logging.getLogger('source.services.hybrid_keyword_manager')
```

## 🚀 향후 개선 계획

### 1. 웹 스크래핑 통합
- 법률 사이트에서 최신 용어 수집
- 정기적인 키워드 업데이트
- 실시간 용어 확장

### 2. 사용자 피드백 학습
- 사용자 행동 기반 키워드 가중치 조정
- 인기 키워드 우선순위 설정
- 개인화된 키워드 추천

### 3. 성능 모니터링
- 실시간 성능 메트릭
- 자동 성능 최적화
- 알림 시스템 구축

## 📚 참고 자료

- [LangChain 문서](https://python.langchain.com/)
- [Gemini API 문서](https://ai.google.dev/docs)
- [법률 용어 사전](https://www.law.go.kr/)
- [프로젝트 개발 규칙](../10_technical_reference/development_rules.md)

---

**마지막 업데이트**: 2025-10-19  
**버전**: 1.0.0  
**상태**: 프로덕션 준비 완료
