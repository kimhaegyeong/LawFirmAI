# 재임베딩 모니터링 스크립트 재사용성 개선 방안

## 개요

`scripts/monitoring` 폴더의 재임베딩 모니터링 스크립트들의 재사용성을 개선하기 위한 방안을 제시합니다.

## 현재 문제점

### 1. 코드 중복
- 모든 스크립트에서 동일한 패턴 반복:
  - 데이터베이스 연결
  - 버전 정보 조회
  - 문서 수 계산
  - 진행률 계산
  - 진행 바 출력

### 2. 일관성 부족
- 각 스크립트마다 다른 에러 처리 방식
- 다른 출력 형식
- 다른 로깅 방식

### 3. 테스트 어려움
- 함수가 독립적으로 테스트하기 어려운 구조
- 의존성이 명확하지 않음

## 개선 방안

### 1. 공통 유틸리티 모듈 생성 ✅

**파일**: `scripts/monitoring/re_embedding_monitor_utils.py`

**제공 기능**:
- `ReEmbeddingMonitor` 클래스: 재임베딩 모니터링 공통 기능 제공
- 데이터베이스 연결 관리
- 버전 정보 조회
- 문서 수 계산
- 진행률 계산
- 성능 메트릭 계산
- 진행 바 포맷팅

**사용 예시**:
```python
from scripts.monitoring.re_embedding_monitor_utils import ReEmbeddingMonitor

monitor = ReEmbeddingMonitor(db_path="data/lawfirm_v2.db", version_id=5)
is_valid, error = monitor.validate()
if not is_valid:
    print(f"오류: {error}")
    return

progress, processed, total = monitor.get_progress()
print(f"진행률: {progress*100:.2f}% ({processed}/{total})")
print(monitor.format_progress_bar(progress))
```

### 2. 스크립트 리팩토링 (권장)

각 스크립트를 공통 유틸리티를 사용하도록 리팩토링:

#### `monitor_re_embedding_progress.py` 개선 예시

**개선 전**:
```python
def monitor_progress(db_path: str, version_id: int):
    conn = sqlite3.connect(db_path)
    # ... 중복 코드 ...
    version_manager = EmbeddingVersionManager(db_path)
    version_info = version_manager.get_version_statistics(version_id)
    # ... 중복 코드 ...
```

**개선 후**:
```python
from scripts.monitoring.re_embedding_monitor_utils import ReEmbeddingMonitor

def monitor_progress(db_path: str, version_id: int):
    monitor = ReEmbeddingMonitor(db_path, version_id)
    is_valid, error = monitor.validate()
    if not is_valid:
        print(f"✗ {error}")
        return
    
    progress, processed, total = monitor.get_progress()
    total_by_type, processed_by_type = monitor.get_documents_by_type()
    
    # 출력 로직...
```

### 3. 공통 출력 포맷터 추가

**제안**: `ReEmbeddingMonitor` 클래스에 출력 메서드 추가

```python
def print_summary(self, include_details: bool = True):
    """요약 정보 출력"""
    progress, processed, total = self.get_progress()
    
    print("=" * 80)
    print(f"재임베딩 진행 상황: {self.get_version_name()}")
    print("=" * 80)
    print(f"전체 문서 수: {total:,}")
    print(f"재임베딩 완료: {processed:,} ({progress*100:.1f}%)")
    print(self.format_progress_bar(progress))
    
    if include_details:
        self.print_details_by_type()
```

### 4. 에러 처리 통일

**제안**: 공통 예외 클래스 정의

```python
class ReEmbeddingMonitorError(Exception):
    """재임베딩 모니터링 관련 예외"""
    pass

class VersionNotFoundError(ReEmbeddingMonitorError):
    """버전을 찾을 수 없을 때"""
    pass

class DatabaseError(ReEmbeddingMonitorError):
    """데이터베이스 오류"""
    pass
```

### 5. 로깅 통일

**제안**: 공통 로깅 설정

```python
import logging

def setup_monitoring_logger(name: str) -> logging.Logger:
    """모니터링 스크립트용 로거 설정"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### 6. 설정 파일 통합

**제안**: 공통 설정 파일 사용

```python
# monitoring_config.py
DEFAULT_DB_PATH = "data/lawfirm_v2.db"
DEFAULT_VERSION_ID = 5
DEFAULT_PROGRESS_THRESHOLD = 0.99
DEFAULT_MONITORING_INTERVAL = 60
```

### 7. 테스트 가능한 구조

**제안**: 함수를 순수 함수로 분리

```python
def calculate_progress(processed: int, total: int) -> float:
    """진행률 계산 (순수 함수)"""
    return processed / total if total > 0 else 0.0

def format_progress_bar(progress: float, length: int = 50) -> str:
    """진행 바 포맷팅 (순수 함수)"""
    filled = int(length * progress)
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {progress*100:.2f}%"
```

## 개선 우선순위

### 높음 (즉시 적용 권장)
1. ✅ 공통 유틸리티 모듈 생성 (`re_embedding_monitor_utils.py`)
2. ✅ Import 경로 수정 (`wait_and_build_faiss_index.py`)
3. 에러 처리 통일
4. 로깅 통일

### 중간 (점진적 적용)
1. 각 스크립트를 공통 유틸리티 사용하도록 리팩토링
2. 공통 출력 포맷터 추가
3. 설정 파일 통합

### 낮음 (선택적 적용)
1. 테스트 코드 작성
2. 문서화 개선
3. 타입 힌트 추가

## 마이그레이션 가이드

### 단계 1: 공통 유틸리티 사용 시작
```python
# 기존 코드
conn = sqlite3.connect(db_path)
cursor = conn.execute("SELECT ...")
# ...

# 개선된 코드
monitor = ReEmbeddingMonitor(db_path, version_id)
progress, processed, total = monitor.get_progress()
```

### 단계 2: 점진적 리팩토링
- 한 번에 하나씩 스크립트 개선
- 기존 기능 유지하면서 공통 유틸리티 사용
- 테스트 후 다음 스크립트로 진행

### 단계 3: 완전한 통합
- 모든 스크립트가 공통 유틸리티 사용
- 중복 코드 제거
- 일관된 인터페이스 제공

## 예상 효과

### 코드 품질
- 중복 코드 감소: 약 40-50%
- 유지보수성 향상
- 테스트 용이성 증가

### 개발 생산성
- 새로운 모니터링 스크립트 작성 시간 단축
- 버그 수정 시간 단축
- 일관된 사용자 경험

### 확장성
- 새로운 모니터링 기능 추가 용이
- 다른 프로젝트에서 재사용 가능
- 플러그인 형태로 확장 가능

## 참고 사항

- 기존 스크립트의 동작은 변경하지 않음 (하위 호환성 유지)
- 점진적 마이그레이션 권장
- 각 스크립트는 독립적으로 실행 가능해야 함

