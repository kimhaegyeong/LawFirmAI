# LawFirmAI 데이터 품질 개선 시스템

## 개요

LawFirmAI 데이터 품질 개선 시스템은 법률 데이터의 품질을 자동으로 검증, 개선, 모니터링하는 종합적인 솔루션입니다. 이 시스템은 4단계로 구성되어 있으며, 각 단계는 데이터 품질 향상을 위한 특정 기능을 담당합니다.

## 최신 업데이트 (2025-01-XX)

### 🎯 키워드 매핑 시스템 정교화
- **가중치 기반 키워드 시스템**: 핵심/중요/보조 키워드로 분류하여 정확도 향상
- **컨텍스트 인식 매핑**: 질문 유형별 맞춤형 키워드 제공
- **동적 키워드 학습**: 사용자 피드백 기반 지속적 개선
- **의미적 유사도 매핑**: 법률 용어 간 의미적 관계 활용

### 📈 품질 향상 효과
- **키워드 포함도**: 0.390 → 0.7+ 목표 달성 가능
- **답변 구조화**: 컨텍스트별 맞춤형 구조 제공
- **법적 정확성**: 의미적 관계를 통한 전문 용어 활용
- **지속적 학습**: 사용자 피드백을 통한 자동 개선

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    데이터 품질 개선 시스템                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: 데이터 품질 검증 및 개선                          │
│  ├── DataQualityValidator                                   │
│  ├── MLEnhancedArticleParser (개선됨)                       │
│  ├── HybridArticleParser                                    │
│  └── Preprocessing Pipeline (업데이트됨)                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: 고급 중복 감지 및 해결                            │
│  ├── AdvancedDuplicateDetector                             │
│  ├── IntelligentDuplicateResolver                           │
│  └── Duplicate Detection Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: 데이터베이스 스키마 개선 및 마이그레이션           │
│  ├── Schema Migration Scripts                              │
│  ├── Data Migration Scripts                                │
│  ├── DatabaseManager (업데이트됨)                           │
│  └── Import Scripts (업데이트됨)                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: 자동화된 품질 관리 및 모니터링                    │
│  ├── AutomatedDataCleaner                                  │
│  ├── RealTimeQualityMonitor                                │
│  ├── ScheduledTaskManager                                   │
│  ├── QualityReportingDashboard                             │
│  └── AutoPipelineOrchestrator (통합됨)                     │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: 향상된 키워드 매핑 시스템 (NEW!)                  │
│  ├── LegalKeywordMapper (가중치 기반)                       │
│  ├── ContextAwareKeywordMapper (컨텍스트 인식)              │
│  ├── AdaptiveKeywordMapper (동적 학습)                      │
│  ├── SemanticKeywordMapper (의미적 유사도)                  │
│  └── EnhancedKeywordMapper (통합 시스템)                    │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: 데이터 품질 검증 및 개선

### 1.1 DataQualityValidator

**위치**: `scripts/data_processing/quality/data_quality_validator.py`

**기능**:
- 법률 데이터의 파싱 품질 검증
- 품질 점수 계산 (0.0 ~ 1.0)
- 품질 개선 제안 생성
- 구조적 완성도 검사

**주요 메서드**:
```python
def validate_parsing_quality(self, law_data: Dict[str, Any]) -> QualityReport
def calculate_quality_score(self, law_data: Dict[str, Any]) -> float
def suggest_improvements(self, law_data: Dict[str, Any]) -> List[str]
```

**사용 예시**:
```python
from scripts.data_processing.quality.data_quality_validator import DataQualityValidator

validator = DataQualityValidator()
quality_report = validator.validate_parsing_quality(law_data)
quality_score = validator.calculate_quality_score(law_data)
```

### 1.2 MLEnhancedArticleParser (개선됨)

**위치**: `scripts/ml_training/model_training/ml_enhanced_parser.py`

**개선사항**:
- 품질 검증 레이어 통합
- 자동 수정 기능 추가
- 누락된 조문 복구
- 중복 조문 제거

**새로운 메서드**:
```python
def validate_parsed_result(self, parsed_result: Dict[str, Any]) -> QualityReport
def fix_missing_articles(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]
def remove_duplicate_articles(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]
def parse_law_with_validation(self, law_content: str) -> Dict[str, Any]
```

### 1.3 HybridArticleParser

**위치**: `scripts/data_processing/quality/hybrid_parser.py`

**기능**:
- 규칙 기반 파싱과 ML 기반 파싱 결합
- 최적 결과 선택
- 후처리 수정 적용
- 수동 검토 플래그 설정

**주요 메서드**:
```python
def parse_law_content(self, law_content: str, law_name: str = "Unknown") -> Dict[str, Any]
def _select_best_result(self, rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, Any]
def _apply_post_processing_corrections(self, result: Dict[str, Any]) -> Dict[str, Any]
```

### 1.4 Preprocessing Pipeline (업데이트됨)

**위치**: `scripts/data_processing/preprocessing/preprocess_laws.py`

**개선사항**:
- HybridArticleParser 통합
- 품질 관련 메타데이터 저장
- 자동 수정 및 수동 검토 플래그

## Phase 2: 고급 중복 감지 및 해결

### 2.1 AdvancedDuplicateDetector

**위치**: `scripts/data_processing/quality/duplicate_detector.py`

**기능**:
- 파일 레벨 중복 감지 (해시, 크기, 이름 유사성)
- 콘텐츠 레벨 중복 감지 (TF-IDF, 코사인 유사성)
- 의미적 중복 감지 (벡터 임베딩 기반)
- 다층 감지 알고리즘

**주요 메서드**:
```python
def detect_file_level_duplicates(self, files: List[Dict[str, Any]]) -> List[DuplicateGroup]
def detect_content_level_duplicates(self, content_items: List[Dict[str, Any]]) -> List[DuplicateGroup]
def detect_semantic_duplicates(self, content_items: List[Dict[str, Any]]) -> List[DuplicateGroup]
```

### 2.2 IntelligentDuplicateResolver

**위치**: `scripts/data_processing/quality/duplicate_resolver.py`

**기능**:
- 지능적 중복 해결 전략
- 품질 기반 우선순위 결정
- 메타데이터 병합
- 버전 히스토리 생성

**해결 전략**:
- `quality_based`: 품질 점수 기반
- `completeness_based`: 완성도 기반
- `recency_based`: 최신성 기반
- `conservative`: 보수적 접근

**주요 메서드**:
```python
def resolve_duplicates(self, duplicate_groups: List[DuplicateGroup], strategy: str = "quality_based") -> List[ResolutionResult]
def add_resolution_strategy(self, strategy: Dict[str, Any])
def export_resolution_report(self, results: List[ResolutionResult]) -> Dict[str, Any]
```

### 2.3 Duplicate Detection Pipeline

**위치**: `scripts/data_processing/run_duplicate_detection.py`

**기능**:
- 독립 실행 가능한 중복 감지 파이프라인
- 자동 해결 옵션
- 상세 보고서 생성
- 통계 및 권장사항 제공

**사용법**:
```bash
python scripts/data_processing/run_duplicate_detection.py \
    --input data/processed \
    --output reports/duplicate_detection \
    --auto-resolve \
    --strategy quality_based
```

## Phase 3: 데이터베이스 스키마 개선 및 마이그레이션

### 3.1 Schema Migration Scripts

**위치**: `scripts/database/migrate_schema_v2.py`

**기능**:
- 데이터베이스 스키마 버전 2로 마이그레이션
- 새로운 컬럼 추가 (품질 관련)
- 새로운 테이블 생성 (중복 그룹, 품질 보고서)
- 인덱스 생성 및 최적화

**새로운 컬럼**:
```sql
-- assembly_laws 테이블에 추가
law_name_hash TEXT UNIQUE,
content_hash TEXT UNIQUE,
quality_score REAL DEFAULT 0.0,
duplicate_group_id TEXT,
is_primary_version BOOLEAN DEFAULT TRUE,
version_number INTEGER DEFAULT 1,
parsing_method TEXT DEFAULT "legacy",
auto_corrected BOOLEAN DEFAULT FALSE,
manual_review_required BOOLEAN DEFAULT FALSE,
migration_timestamp TEXT
```

**새로운 테이블**:
- `duplicate_groups`: 중복 그룹 정보
- `quality_reports`: 품질 보고서 상세 정보
- `migration_history`: 마이그레이션 히스토리
- `schema_version`: 스키마 버전 관리

### 3.2 Data Migration Scripts

**위치**: `scripts/database/migrate_existing_data.py`

**기능**:
- 기존 데이터를 새 스키마에 맞게 마이그레이션
- 해시 계산 및 저장
- 품질 점수 계산
- 중복 감지 및 해결
- 배치 처리 지원

**사용법**:
```bash
python scripts/database/migrate_existing_data.py \
    --db-path data/lawfirm.db \
    --batch-size 1000 \
    --verbose
```

### 3.3 DatabaseManager (업데이트됨)

**위치**: `source/data/database.py`

**개선사항**:
- 새로운 스키마 지원
- 품질 통계 메서드 추가
- 중복 통계 메서드 추가
- 품질 기반 쿼리 메서드 추가

**새로운 메서드**:
```python
def get_quality_statistics(self) -> Dict[str, Any]
def get_duplicate_statistics(self) -> Dict[str, Any]
def get_laws_by_quality_score(self, min_score: float, max_score: float) -> List[Dict[str, Any]]
def get_laws_requiring_review(self) -> List[Dict[str, Any]]
def update_law_quality_score(self, law_id: str, quality_score: float)
```

### 3.4 Import Scripts (업데이트됨)

**위치**: `scripts/data_processing/utilities/import_laws_to_db.py`

**개선사항**:
- 중복 검사 로직 통합
- 품질 점수 계산 및 저장
- 품질 관련 메타데이터 처리
- 향상된 에러 처리

**새로운 기능**:
```python
def _check_for_duplicates(self, new_law_record: Tuple, new_law_data: Dict[str, Any]) -> Dict[str, Any]
def _update_existing_law(self, new_law_record: Tuple, new_law_data: Dict[str, Any], existing_law_id: str) -> bool
def _enhance_law_record_with_quality(self, law_record: Tuple, law_data: Dict[str, Any], quality_score: float) -> Tuple
```

## Phase 4: 자동화된 품질 관리 및 모니터링

### 4.1 AutomatedDataCleaner

**위치**: `scripts/data_processing/quality/automated_data_cleaner.py`

**기능**:
- 일일 데이터 정리 루틴
- 주간 종합 정리 루틴
- 월간 감사 루틴
- 실시간 품질 모니터링

**정리 작업**:
- 중복 감지 및 해결
- 품질 평가 및 개선
- 고아 데이터 정리
- 인덱스 최적화

**사용법**:
```bash
python scripts/data_processing/quality/automated_data_cleaner.py \
    --operation daily \
    --db-path data/lawfirm.db \
    --output reports/daily_cleaning.json
```

### 4.2 RealTimeQualityMonitor

**위치**: `scripts/data_processing/quality/real_time_quality_monitor.py`

**기능**:
- 연속적인 품질 모니터링
- 임계값 초과 시 알림 생성
- 품질 트렌드 분석
- 성능 메트릭 추적
- 자동 권장사항 생성

**모니터링 메트릭**:
- 전체 품질 점수
- 낮은 품질 레코드 비율
- 중복 비율
- 최근 추가된 데이터 품질
- 품질 트렌드 (개선/안정/악화)

**사용법**:
```bash
python scripts/data_processing/quality/real_time_quality_monitor.py \
    --action start \
    --db-path data/lawfirm.db
```

### 4.3 ScheduledTaskManager

**위치**: `scripts/data_processing/quality/scheduled_task_manager.py`

**기능**:
- 예약된 데이터 품질 작업 관리
- 일일/주간/월간 작업 스케줄링
- 작업 실행 로깅
- 에러 처리 및 복구
- 알림 시스템

**스케줄된 작업**:
- 일일 정리: 매일 오전 2시
- 주간 정리: 매주 일요일 오전 3시
- 월간 감사: 매월 1일 오전 4시
- 실시간 모니터링: 5분 간격

**사용법**:
```bash
python scripts/data_processing/quality/scheduled_task_manager.py \
    --action start \
    --db-path data/lawfirm.db
```

### 4.4 QualityReportingDashboard

**위치**: `scripts/data_processing/quality/quality_reporting_dashboard.py`

**기능**:
- 실시간 품질 메트릭 대시보드
- 품질 보고서 자동 생성
- 다양한 형식으로 내보내기 (JSON, CSV, HTML)
- 성능 분석 및 시각화
- 품질 권장사항 생성

**보고서 유형**:
- `summary`: 요약 보고서
- `detailed`: 상세 보고서
- `trends`: 트렌드 분석 보고서
- `performance`: 성능 분석 보고서

**사용법**:
```bash
python scripts/data_processing/quality/quality_reporting_dashboard.py \
    --action report \
    --report-type summary \
    --period-days 7 \
    --format html \
    --output reports/quality_report.html
```

### 4.5 AutoPipelineOrchestrator (통합됨)

**위치**: `scripts/data_processing/auto_pipeline_orchestrator.py`

**개선사항**:
- 품질 검증 단계 통합 (Step 3)
- 품질 메트릭을 PipelineResult에 포함
- 품질 관련 설정 추가
- 품질 보고서 생성

**새로운 파이프라인 단계**:
1. 데이터 감지
2. 전처리
3. **품질 검증 및 개선** (새로 추가)
4. 벡터 임베딩
5. 데이터베이스 임포트
6. 최종 통계 생성

**품질 관련 설정**:
```python
'quality': {
    'enabled': True,
    'validation_threshold': 0.7,
    'enable_duplicate_detection': True,
    'enable_quality_improvement': True,
    'quality_thresholds': {
        'excellent': 0.9,
        'good': 0.8,
        'fair': 0.6,
        'poor': 0.4
    }
}
```

## Phase 5: 향상된 키워드 매핑 시스템 (NEW!)

### 5.1 LegalKeywordMapper (가중치 기반)

**위치**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 가중치 기반 키워드 분류 (핵심/중요/보조)
- 질문 유형별 필수 키워드 매핑
- 키워드 포함도 계산 및 분석
- 법률 용어 사전 관리

**가중치 시스템**:
- **핵심 키워드 (Core)**: 가중치 1.0 - 필수 포함 키워드
- **중요 키워드 (Important)**: 가중치 0.8 - 중요도가 높은 키워드  
- **보조 키워드 (Supporting)**: 가중치 0.6 - 보완적 키워드

**주요 메서드**:
```python
def get_weighted_keywords_for_question(self, question: str, query_type: str) -> Dict[str, List[str]]
def calculate_weighted_keyword_coverage(self, answer: str, query_type: str, question: str = "") -> Dict[str, float]
def get_keyword_analysis_report(self, answer: str, query_type: str, question: str = "") -> Dict[str, any]
```

**사용 예시**:
```python
from source.services.langgraph.keyword_mapper import LegalKeywordMapper

mapper = LegalKeywordMapper()
keywords = mapper.get_keywords_for_question("계약서 검토 시 주의사항", "contract_review")
coverage = mapper.calculate_weighted_keyword_coverage(answer, "contract_review", question)
```

### 5.2 ContextAwareKeywordMapper (컨텍스트 인식)

**위치**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 질문의 컨텍스트 자동 식별
- 컨텍스트별 맞춤형 키워드 제공
- 질문 의도 분석 및 복잡도 평가
- 컨텍스트 기반 권장사항 생성

**컨텍스트 패턴**:
- **질문형**: "어떻게", "무엇인가", "언제", "어디서", "왜"
- **절차형**: "절차", "방법", "과정", "단계", "순서"
- **비교형**: "차이점", "비교", "구분", "다른점"
- **문제해결형**: "문제", "해결", "방법", "대처", "대응"
- **법적효력형**: "효력", "무효", "취소", "해제", "해지"

**주요 메서드**:
```python
def identify_context(self, question: str) -> str
def get_contextual_keywords(self, question: str, query_type: str) -> Dict[str, List[str]]
def analyze_question_intent(self, question: str) -> Dict[str, any]
def get_enhanced_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]
```

### 5.3 AdaptiveKeywordMapper (동적 학습)

**위치**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 사용자 피드백 기반 키워드 효과성 학습
- 질문 패턴 분석 및 키워드 추천
- 지속적인 키워드 효과성 업데이트
- 학습 인사이트 및 개선 권장사항 제공

**학습 메커니즘**:
- 사용자 평점 (40%) + 답변 품질 (60%) 기반 효과성 계산
- 질문 패턴별 키워드 효과성 추적
- 최근 피드백 기반 동적 조정
- 효과성이 낮은 키워드 자동 식별

**주요 메서드**:
```python
def update_keyword_effectiveness(self, question: str, keywords: List[str], user_rating: float, answer_quality: float, query_type: str = "")
def get_effective_keywords(self, query_type: str, limit: int = 10) -> List[str]
def get_pattern_based_keywords(self, question: str, query_type: str) -> List[str]
def get_learning_insights(self) -> Dict[str, any]
```

### 5.4 SemanticKeywordMapper (의미적 유사도)

**위치**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 법률 용어 간 의미적 관계 정의
- 키워드 의미적 클러스터링
- 의미적 키워드 확장 및 추천
- 의미적 다양성 분석

**의미적 관계**:
- **동의어**: 거리 0.1 - 완전히 같은 의미
- **관련어**: 거리 0.3 - 밀접한 관련성
- **컨텍스트어**: 거리 0.5 - 같은 맥락에서 사용

**주요 메서드**:
```python
def calculate_semantic_similarity(self, keyword1: str, keyword2: str) -> float
def find_semantic_related_keywords(self, target_keyword: str, threshold: float = 0.5) -> List[Tuple[str, float]]
def expand_keywords_semantically(self, keywords: List[str], expansion_factor: float = 0.7) -> List[str]
def get_semantic_keyword_clusters(self, keywords: List[str]) -> Dict[str, List[str]]
```

### 5.5 EnhancedKeywordMapper (통합 시스템)

**위치**: `source/services/langgraph/keyword_mapper.py`

**기능**:
- 모든 키워드 매핑 시스템 통합
- 종합적인 키워드 우선순위 계산
- 상세한 분석 보고서 생성
- 실시간 피드백 업데이트

**통합 방식**:
- 가중치 기반 점수 (30%)
- 컨텍스트 기반 점수 (25%)
- 적응형 점수 (25%)
- 의미적 점수 (20%)

**주요 메서드**:
```python
def get_comprehensive_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]
def update_feedback(self, question: str, keywords: List[str], user_rating: float, answer_quality: float, query_type: str = "")
def get_keyword_effectiveness_report(self) -> Dict[str, any]
```

### 5.6 키워드 매핑 시스템 사용법

**기본 사용법**:
```python
from source.services.langgraph.keyword_mapper import EnhancedKeywordMapper

# 통합 키워드 매퍼 초기화
mapper = EnhancedKeywordMapper()

# 종합적인 키워드 매핑 실행
result = mapper.get_comprehensive_keyword_mapping(
    question="계약서 검토 시 주의해야 할 사항은 무엇인가요?",
    query_type="contract_review"
)

# 결과 분석
print(f"기본 키워드: {result['base_keywords']}")
print(f"가중치별 키워드: {result['weighted_keywords']}")
print(f"컨텍스트: {result['contextual_data']['identified_context']}")
print(f"상위 우선순위 키워드: {result['comprehensive_analysis']['top_keywords']}")
```

**피드백 업데이트**:
```python
# 사용자 피드백 업데이트
mapper.update_feedback(
    question="계약서 검토 시 주의해야 할 사항은 무엇인가요?",
    keywords=["계약서", "당사자", "조건", "기간"],
    user_rating=0.8,
    answer_quality=0.9,
    query_type="contract_review"
)
```

**효과성 보고서**:
```python
# 키워드 효과성 보고서 생성
report = mapper.get_keyword_effectiveness_report()
print(f"적응형 인사이트: {report['adaptive_insights']}")
print(f"의미적 분석: {report['semantic_analysis']}")
```

### 5.7 성능 및 효과

**테스트 결과**:
- **키워드 확장**: 기본 24개 → 확장 38개 (1.58배 증가)
- **의미적 클러스터링**: 27개 클러스터로 체계적 분류
- **컨텍스트 인식**: 질문 유형별 맞춤형 키워드 제공
- **가중치 포함도**: 핵심 키워드 75% 포함도 달성

**기대 효과**:
- 키워드 포함도: 0.390 → 0.7+ 목표 달성 가능
- 답변 구조화 개선: 컨텍스트별 맞춤형 구조 제공
- 법적 정확성 증대: 의미적 관계를 통한 전문 용어 활용
- 지속적 학습: 사용자 피드백을 통한 자동 개선

## 설정 및 구성

### 환경 변수

```bash
# 데이터베이스 설정
DATABASE_URL=sqlite:///data/lawfirm.db

# 품질 모니터링 설정
QUALITY_MONITORING_ENABLED=true
QUALITY_CHECK_INTERVAL=300
QUALITY_THRESHOLD=0.8

# 중복 감지 설정
DUPLICATE_DETECTION_ENABLED=true
DUPLICATE_THRESHOLD=0.95
DUPLICATE_MAX_PERCENTAGE=5.0

# 알림 설정
ENABLE_EMAIL_NOTIFICATIONS=false
ENABLE_WEBHOOK_NOTIFICATIONS=false
NOTIFICATION_WEBHOOK_URL=

# 키워드 매핑 시스템 설정
KEYWORD_MAPPING_ENABLED=true
KEYWORD_EFFECTIVENESS_FILE=data/keyword_effectiveness.json
KEYWORD_LEARNING_ENABLED=true
SEMANTIC_SIMILARITY_THRESHOLD=0.6
CONTEXT_AWARE_MAPPING=true
```

### 설정 파일 예시

```json
{
  "quality": {
    "enabled": true,
    "validation_threshold": 0.7,
    "enable_duplicate_detection": true,
    "enable_quality_improvement": true,
    "quality_thresholds": {
      "excellent": 0.9,
      "good": 0.8,
      "fair": 0.6,
      "poor": 0.4
    }
  },
  "quality_monitor": {
    "enabled": true,
    "check_interval_seconds": 300,
    "alert_thresholds": {
      "overall_quality_min": 0.8,
      "duplicate_max_percentage": 5.0
    }
  },
  "scheduling": {
    "daily_cleaning_time": "02:00",
    "weekly_cleaning_day": "sunday",
    "weekly_cleaning_time": "03:00",
    "monthly_audit_day": 1,
    "monthly_audit_time": "04:00",
    "real_time_monitoring": true
  },
  "keyword_mapping": {
    "enabled": true,
    "effectiveness_file": "data/keyword_effectiveness.json",
    "learning_enabled": true,
    "semantic_similarity_threshold": 0.6,
    "context_aware_mapping": true,
    "weighted_keywords": {
      "core_weight": 1.0,
      "important_weight": 0.8,
      "supporting_weight": 0.6
    },
    "adaptive_learning": {
      "user_rating_weight": 0.4,
      "answer_quality_weight": 0.6,
      "min_feedback_count": 5,
      "learning_rate": 0.1
    }
  }
}
```

## 사용 가이드

### 1. 초기 설정

```bash
# 1. 스키마 마이그레이션
python scripts/database/migrate_schema_v2.py --db-path data/lawfirm.db

# 2. 기존 데이터 마이그레이션
python scripts/database/migrate_existing_data.py --db-path data/lawfirm.db

# 3. 중복 감지 및 해결
python scripts/data_processing/run_duplicate_detection.py \
    --input data/processed \
    --output reports/duplicate_detection \
    --auto-resolve
```

### 2. 일상적인 운영

```bash
# 1. 자동화된 파이프라인 실행 (품질 검증 포함)
python scripts/data_processing/auto_pipeline_orchestrator.py \
    --data-source law_only \
    --auto-detect

# 2. 품질 보고서 생성
python scripts/data_processing/quality/quality_reporting_dashboard.py \
    --action report \
    --report-type summary \
    --period-days 7

# 3. 실시간 모니터링 시작
python scripts/data_processing/quality/real_time_quality_monitor.py \
    --action start
```

### 3. 예약된 작업 설정

```bash
# 스케줄된 작업 관리자 시작
python scripts/data_processing/quality/scheduled_task_manager.py \
    --action start \
    --db-path data/lawfirm.db
```

### 4. 수동 품질 관리

```bash
# 일일 정리 작업 실행
python scripts/data_processing/quality/automated_data_cleaner.py \
    --operation daily \
    --db-path data/lawfirm.db

# 주간 정리 작업 실행
python scripts/data_processing/quality/automated_data_cleaner.py \
    --operation weekly \
    --db-path data/lawfirm.db

# 월간 감사 실행
python scripts/data_processing/quality/automated_data_cleaner.py \
    --operation monthly \
    --db-path data/lawfirm.db
```

### 5. 키워드 매핑 시스템 사용

```python
# 키워드 매핑 시스템 테스트
from source.services.langgraph.keyword_mapper import EnhancedKeywordMapper

# 통합 키워드 매퍼 초기화
mapper = EnhancedKeywordMapper()

# 종합적인 키워드 매핑 실행
result = mapper.get_comprehensive_keyword_mapping(
    question="계약서 검토 시 주의해야 할 사항은 무엇인가요?",
    query_type="contract_review"
)

# 결과 분석
print(f"기본 키워드: {result['base_keywords']}")
print(f"가중치별 키워드: {result['weighted_keywords']}")
print(f"컨텍스트: {result['contextual_data']['identified_context']}")
print(f"상위 우선순위 키워드: {result['comprehensive_analysis']['top_keywords']}")

# 사용자 피드백 업데이트
mapper.update_feedback(
    question="계약서 검토 시 주의해야 할 사항은 무엇인가요?",
    keywords=["계약서", "당사자", "조건", "기간"],
    user_rating=0.8,
    answer_quality=0.9,
    query_type="contract_review"
)

# 키워드 효과성 보고서 생성
report = mapper.get_keyword_effectiveness_report()
print(f"적응형 인사이트: {report['adaptive_insights']}")
```

## 모니터링 및 알림

### 품질 메트릭

- **전체 품질 점수**: 모든 법률 데이터의 평균 품질 점수
- **품질 분포**: 우수/양호/보통/불량 비율
- **중복 비율**: 중복된 데이터의 비율
- **품질 트렌드**: 시간에 따른 품질 변화 추이

### 알림 조건

- 전체 품질 점수가 임계값 이하로 떨어질 때
- 중복 비율이 임계값을 초과할 때
- 최근 추가된 데이터의 품질이 낮을 때
- 품질이 지속적으로 악화될 때

### 알림 심각도

- **Critical**: 즉시 조치 필요
- **High**: 우선 조치 필요
- **Medium**: 계획된 조치 필요
- **Low**: 모니터링 필요

## 성능 최적화

### 메모리 사용량 최적화

- 모델 양자화 (Float16)
- 지연 로딩
- 메모리 압축 및 정리
- 캐싱 전략

### 처리 속도 최적화

- 배치 처리
- 병렬 처리
- 인덱스 최적화
- 쿼리 최적화

### 스토리지 최적화

- 데이터 압축
- 아카이빙 전략
- 중복 제거
- 정리 작업 자동화

## 문제 해결

### 일반적인 문제

1. **품질 점수가 낮은 경우**
   - 파싱 품질 검사 실행
   - 수동 검토 플래그된 데이터 확인
   - 파싱 규칙 개선

2. **중복이 많이 감지되는 경우**
   - 중복 감지 임계값 조정
   - 해결 전략 변경
   - 데이터 소스 검토

3. **성능이 느린 경우**
   - 배치 크기 조정
   - 인덱스 최적화
   - 메모리 사용량 확인

### 로그 및 디버깅

```bash
# 로그 레벨 설정
export LOG_LEVEL=DEBUG

# 상세 로그 확인
tail -f logs/quality_improvement.log
tail -f logs/automated_data_cleaner.log
tail -f logs/real_time_quality_monitor.log
```

## API 참조

### DataQualityValidator

```python
class DataQualityValidator:
    def validate_parsing_quality(self, law_data: Dict[str, Any]) -> QualityReport
    def calculate_quality_score(self, law_data: Dict[str, Any]) -> float
    def suggest_improvements(self, law_data: Dict[str, Any]) -> List[str]
```

### AdvancedDuplicateDetector

```python
class AdvancedDuplicateDetector:
    def detect_file_level_duplicates(self, files: List[Dict[str, Any]]) -> List[DuplicateGroup]
    def detect_content_level_duplicates(self, content_items: List[Dict[str, Any]]) -> List[DuplicateGroup]
    def detect_semantic_duplicates(self, content_items: List[Dict[str, Any]]) -> List[DuplicateGroup]
```

### IntelligentDuplicateResolver

```python
class IntelligentDuplicateResolver:
    def resolve_duplicates(self, duplicate_groups: List[DuplicateGroup], strategy: str = "quality_based") -> List[ResolutionResult]
    def add_resolution_strategy(self, strategy: Dict[str, Any])
    def export_resolution_report(self, results: List[ResolutionResult]) -> Dict[str, Any]
```

### AutomatedDataCleaner

```python
class AutomatedDataCleaner:
    def run_daily_cleaning(self) -> CleaningReport
    def run_weekly_cleaning(self) -> CleaningReport
    def run_monthly_audit(self) -> CleaningReport
    def monitor_real_time_quality(self) -> Dict[str, Any]
```

### RealTimeQualityMonitor

```python
class RealTimeQualityMonitor:
    def start_monitoring(self) -> bool
    def stop_monitoring(self) -> bool
    def get_current_metrics(self) -> Optional[QualityMetrics]
    def get_active_alerts(self, severity: Optional[str] = None) -> List[QualityAlert]
```

### QualityReportingDashboard

```python
class QualityReportingDashboard:
    def get_dashboard_data(self, force_refresh: bool = False) -> QualityDashboardData
    def generate_quality_report(self, report_type: str = 'summary', period_days: int = 7) -> QualityReportData
    def export_report(self, report_data: QualityReportData, format: str = 'json', output_path: Optional[str] = None) -> str
```

### 키워드 매핑 시스템 API

#### LegalKeywordMapper

```python
class LegalKeywordMapper:
    @classmethod
    def get_keywords_for_question(cls, question: str, query_type: str) -> List[str]
    @classmethod
    def get_weighted_keywords_for_question(cls, question: str, query_type: str) -> Dict[str, List[str]]
    @classmethod
    def calculate_weighted_keyword_coverage(cls, answer: str, query_type: str, question: str = "") -> Dict[str, float]
    @classmethod
    def get_keyword_analysis_report(cls, answer: str, query_type: str, question: str = "") -> Dict[str, any]
```

#### ContextAwareKeywordMapper

```python
class ContextAwareKeywordMapper:
    def identify_context(self, question: str) -> str
    def get_contextual_keywords(self, question: str, query_type: str) -> Dict[str, List[str]]
    def analyze_question_intent(self, question: str) -> Dict[str, any]
    def get_enhanced_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]
```

#### AdaptiveKeywordMapper

```python
class AdaptiveKeywordMapper:
    def update_keyword_effectiveness(self, question: str, keywords: List[str], user_rating: float, answer_quality: float, query_type: str = "")
    def get_effective_keywords(self, query_type: str, limit: int = 10) -> List[str]
    def get_pattern_based_keywords(self, question: str, query_type: str) -> List[str]
    def get_learning_insights(self) -> Dict[str, any]
    def recommend_keyword_improvements(self, query_type: str) -> List[str]
```

#### SemanticKeywordMapper

```python
class SemanticKeywordMapper:
    def calculate_semantic_similarity(self, keyword1: str, keyword2: str) -> float
    def find_semantic_related_keywords(self, target_keyword: str, threshold: float = 0.5) -> List[Tuple[str, float]]
    def expand_keywords_semantically(self, keywords: List[str], expansion_factor: float = 0.7) -> List[str]
    def get_semantic_keyword_clusters(self, keywords: List[str]) -> Dict[str, List[str]]
    def analyze_keyword_semantic_coverage(self, answer: str, keywords: List[str]) -> Dict[str, any]
```

#### EnhancedKeywordMapper

```python
class EnhancedKeywordMapper:
    def get_comprehensive_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]
    def update_feedback(self, question: str, keywords: List[str], user_rating: float, answer_quality: float, query_type: str = "")
    def get_keyword_effectiveness_report(self) -> Dict[str, any]
```

## 테스트

### 단위 테스트 실행

```bash
# 전체 테스트 실행
python tests/test_quality_improvement_workflow.py --test-class all

# 특정 컴포넌트 테스트
python tests/test_quality_improvement_workflow.py --test-class validator
python tests/test_quality_improvement_workflow.py --test-class detector
python tests/test_quality_improvement_workflow.py --test-class resolver
python tests/test_quality_improvement_workflow.py --test-class cleaner
python tests/test_quality_improvement_workflow.py --test-class monitor
python tests/test_quality_improvement_workflow.py --test-class dashboard
python tests/test_quality_improvement_workflow.py --test-class orchestrator
python tests/test_quality_improvement_workflow.py --test-class workflow

# 키워드 매핑 시스템 테스트
python tests/test_keyword_mapping_system.py --test-class legal_mapper
python tests/test_keyword_mapping_system.py --test-class context_mapper
python tests/test_keyword_mapping_system.py --test-class adaptive_mapper
python tests/test_keyword_mapping_system.py --test-class semantic_mapper
python tests/test_keyword_mapping_system.py --test-class enhanced_mapper
```

### 통합 테스트

```bash
# 품질 개선 워크플로우 전체 테스트
python tests/test_quality_improvement_workflow.py --test-class workflow --verbose

# 키워드 매핑 시스템 통합 테스트
python tests/test_keyword_mapping_system.py --test-class integration --verbose
```

## 확장성 및 커스터마이징

### 새로운 품질 검증 규칙 추가

```python
class CustomQualityValidator(DataQualityValidator):
    def _custom_validation_rule(self, law_data: Dict[str, Any]) -> float:
        # 사용자 정의 검증 로직
        return score
```

### 새로운 중복 해결 전략 추가

```python
custom_strategy = {
    'name': 'custom_strategy',
    'description': '사용자 정의 해결 전략',
    'scoring_function': lambda item: custom_score(item)
}

resolver.add_resolution_strategy(custom_strategy)
```

### 새로운 알림 채널 추가

```python
def custom_alert_callback(alert: QualityAlert):
    # 사용자 정의 알림 처리
    pass

monitor.add_alert_callback(custom_alert_callback)
```

### 키워드 매핑 시스템 커스터마이징

#### 새로운 법률 용어 관계 추가

```python
# SemanticKeywordMapper 확장
class CustomSemanticKeywordMapper(SemanticKeywordMapper):
    def __init__(self):
        super().__init__()
        # 새로운 법률 용어 관계 추가
        self.semantic_relations["새로운_법률_분야"] = {
            "synonyms": ["동의어1", "동의어2"],
            "related": ["관련어1", "관련어2"],
            "context": ["컨텍스트어1", "컨텍스트어2"]
        }
```

#### 새로운 컨텍스트 패턴 추가

```python
# ContextAwareKeywordMapper 확장
class CustomContextAwareKeywordMapper(ContextAwareKeywordMapper):
    def __init__(self):
        super().__init__()
        # 새로운 컨텍스트 패턴 추가
        self.context_patterns["새로운_컨텍스트"] = ["패턴1", "패턴2", "패턴3"]
        self.context_keywords["새로운_컨텍스트"] = ["키워드1", "키워드2", "키워드3"]
```

#### 새로운 키워드 효과성 메트릭 추가

```python
# AdaptiveKeywordMapper 확장
class CustomAdaptiveKeywordMapper(AdaptiveKeywordMapper):
    def _calculate_custom_effectiveness(self, keyword_data: Dict[str, Any]) -> float:
        # 사용자 정의 효과성 계산 로직
        base_score = keyword_data['effectiveness_score']
        usage_frequency = keyword_data['total_usage']
        recency_bonus = self._calculate_recency_bonus(keyword_data['last_updated'])
        
        return base_score * (1 + usage_frequency * 0.1) * recency_bonus
    
    def _calculate_recency_bonus(self, last_updated: str) -> float:
        # 최근성 보너스 계산
        from datetime import datetime, timedelta
        last_update = datetime.fromisoformat(last_updated)
        days_ago = (datetime.now() - last_update).days
        
        if days_ago <= 7:
            return 1.2  # 최근 사용된 키워드에 보너스
        elif days_ago <= 30:
            return 1.1
        else:
            return 1.0
```

## 보안 고려사항

- 데이터베이스 접근 권한 관리
- 민감한 데이터 암호화
- 로그 파일 보안
- API 접근 제어
- 백업 및 복구 전략

## 라이선스 및 기여

이 시스템은 LawFirmAI 프로젝트의 일부이며, 프로젝트의 라이선스 정책을 따릅니다.

기여를 원하시는 경우, 다음 사항을 확인해주세요:
- 코드 스타일 가이드 준수
- 테스트 케이스 작성
- 문서 업데이트
- 품질 검증 통과

## 지원 및 문의

기술적 문제나 질문이 있으시면 다음 채널을 통해 문의해주세요:
- 이슈 트래커: GitHub Issues
- 문서: 프로젝트 Wiki
- 이메일: support@lawfirmai.com

---

**최종 업데이트**: 2025-01-XX  
**버전**: 2.0.0  
**작성자**: LawFirmAI Development Team
