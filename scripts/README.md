# Scripts Directory - 통합 관리 시스템

LawFirmAI 프로젝트의 스크립트들이 목적과 용도에 따라 체계적으로 분류되어 관리됩니다.

## 📁 새로운 폴더 구조

### 🔧 **core/** - 핵심 기능
통합된 매니저 클래스들과 공통 모듈

- `unified_rebuild_manager.py` - 통합 데이터베이스 재구축 매니저
- `unified_vector_manager.py` - 통합 벡터 임베딩 매니저  
- `base_manager.py` - 모든 매니저의 기본 클래스 및 공통 유틸리티

### 🧪 **testing/** - 테스트 통합
모든 테스트 관련 스크립트들

- `unified_test_suite.py` - 통합 테스트 스위트 매니저
- `simple_multi_stage_test.py` - 간단한 다단계 테스트

### 📊 **analysis/** - 분석 도구
데이터 분석, 품질 검증, 모델 최적화 분석

- `simple_test_analysis.py` - 간단한 테스트 분석
- 기존 analysis/ 폴더의 모든 분석 스크립트들

### 🛠️ **utilities/** - 유틸리티
공통 유틸리티 및 도구들

- `setup_console_encoding.py` - 콘솔 인코딩 설정
- `check_db_schema.py` - 데이터베이스 스키마 확인
- `add_case_name_column.py` - 케이스명 컬럼 추가

### 📥 **data_collection/** - 데이터 수집 (기존 유지)
다양한 법률 데이터 소스에서 데이터를 수집하는 스크립트들

### 🔧 **data_processing/** - 데이터 처리 (기존 유지)
법률 데이터의 전처리, 정제, 최적화를 담당하는 스크립트들

### 🧠 **ml_training/** - ML 및 벡터 임베딩 (기존 유지)
AI 모델의 훈련, 평가, 벡터 임베딩 생성을 담당하는 스크립트들

### 🗄️ **database/** - 데이터베이스 (기존 유지)
데이터베이스 스키마, 백업, 분석을 담당하는 스크립트들

### ⚡ **benchmarking/** - 성능 벤치마킹 (기존 유지)
모델과 벡터 저장소의 성능을 측정하는 스크립트들

### 📈 **monitoring/** - 모니터링 (기존 유지)
시스템 모니터링, 로그 분석을 담당하는 스크립트들

### 📈 **performance/** - 성능 최적화 (기존 유지)
검색 성능 최적화 및 벡터 인덱스 최적화

### 🗂️ **deprecated/** - 사용 중단 예정
기존의 중복되거나 개선된 버전으로 대체된 스크립트들

## 🚀 새로운 통합 시스템 사용법

### 1. 데이터베이스 재구축
```bash
# 전체 재구축 (조문 처리 포함)
python scripts/core/unified_rebuild_manager.py --mode full

# 실제 데이터 재구축
python scripts/core/unified_rebuild_manager.py --mode real

# 간단한 재구축
python scripts/core/unified_rebuild_manager.py --mode simple

# 증분 재구축
python scripts/core/unified_rebuild_manager.py --mode incremental

# 품질 개선 전용 (assembly_articles 테이블)
python scripts/core/unified_rebuild_manager.py --mode quality_fix
```

### 2. 벡터 임베딩 구축
```bash
# 전체 벡터 인덱스 구축
python scripts/core/unified_vector_manager.py --mode full --model ko-sroberta

# 증분 벡터 구축
python scripts/core/unified_vector_manager.py --mode incremental

# 재시작 가능한 구축
python scripts/core/unified_vector_manager.py --mode resumable

# CPU 최적화 구축
python scripts/core/unified_vector_manager.py --mode cpu_optimized
```

### 3. 테스트 실행
```bash
# 검증 테스트
python scripts/testing/unified_test_suite.py --test-type validation --execution-mode sequential

# 성능 테스트
python scripts/testing/unified_test_suite.py --test-type performance --execution-mode parallel

# 통합 테스트
python scripts/testing/unified_test_suite.py --test-type integration --execution-mode async

# 대규모 테스트
python scripts/testing/unified_test_suite.py --test-type massive --execution-mode multiprocess --max-workers 8

# 벡터 임베딩 테스트
python scripts/testing/unified_test_suite.py --test-type vector_embedding --execution-mode sequential

# 시맨틱 검색 테스트
python scripts/testing/unified_test_suite.py --test-type semantic_search --execution-mode sequential
```

## 🔄 마이그레이션 가이드

### 기존 스크립트에서 새 시스템으로 전환

#### 데이터 재구축
```bash
# 기존: python scripts/full_raw_data_rebuild.py
# 신규: python scripts/core/unified_rebuild_manager.py --mode full

# 기존: python scripts/real_data_rebuild.py  
# 신규: python scripts/core/unified_rebuild_manager.py --mode real

# 기존: python scripts/simple_database_rebuild.py
# 신규: python scripts/core/unified_rebuild_manager.py --mode simple

# 기존: python fix_assembly_articles_quality.py
# 신규: python scripts/core/unified_rebuild_manager.py --mode quality_fix

# 기존: python fix_assembly_articles_quality_v2.py
# 신규: python scripts/core/unified_rebuild_manager.py --mode quality_fix
```

#### 벡터 임베딩
```bash
# 기존: python scripts/efficient_vector_builder.py
# 신규: python scripts/core/unified_vector_manager.py --mode full

# 기존: python scripts/ml_training/vector_embedding/build_ml_enhanced_vector_db.py
# 신규: python scripts/core/unified_vector_manager.py --mode full --model ko-sroberta
```

#### 테스트
```bash
# 기존: python scripts/massive_test_runner.py
# 신규: python scripts/testing/unified_test_suite.py --test-type massive --execution-mode multiprocess

# 기존: python scripts/test_performance_optimization.py
# 신규: python scripts/testing/unified_test_suite.py --test-type performance

# 기존: python simple_vector_test.py
# 신규: python scripts/testing/unified_test_suite.py --test-type vector_embedding

# 기존: python test_vector_embeddings.py
# 신규: python scripts/testing/unified_test_suite.py --test-type semantic_search
```

## 📋 주요 개선사항

### 1. 중복 제거
- **데이터 재구축**: 7개 스크립트 → 1개 통합 매니저 (품질 개선 기능 포함)
- **벡터 빌딩**: 8개 스크립트 → 1개 통합 매니저  
- **테스트**: 42개+ 스크립트 → 1개 통합 스위트 (벡터/시맨틱 테스트 포함)

### 2. 통합 설정 관리
```python
from scripts.core.base_manager import ScriptConfigManager

config_manager = ScriptConfigManager('config/scripts_config.json')
db_config = config_manager.get_database_config()
vector_config = config_manager.get_vector_config()
```

### 3. 표준화된 로깅
```python
from scripts.core.base_manager import BaseManager, BaseConfig

class MyManager(BaseManager):
    def execute(self):
        self.logger.info("Standardized logging")
        # 자동으로 파일과 콘솔에 로그 출력
```

### 4. 에러 처리 표준화
```python
from scripts.core.base_manager import ErrorHandler

error_handler = ErrorHandler(self.logger)
try:
    # 작업 수행
    pass
except Exception as e:
    error_handler.handle_error(e, "operation_context")
```

### 5. 성능 모니터링
```python
from scripts.core.base_manager import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("operation")
# 작업 수행
duration = monitor.end_timer("operation")
```

### 6. 품질 개선 기능 (신규)
```python
# Assembly Articles 품질 개선
from scripts.core.unified_rebuild_manager import UnifiedRebuildManager, RebuildConfig, RebuildMode

config = RebuildConfig(
    mode=RebuildMode.QUALITY_FIX,
    quality_fix_enabled=True
)
manager = UnifiedRebuildManager(config)
results = manager.rebuild_database()
```

### 7. 벡터 임베딩 테스트 (신규)
```python
# 벡터 임베딩 테스트
from scripts.testing.unified_test_suite import UnifiedTestSuite, TestConfig, TestType

config = TestConfig(
    test_type=TestType.VECTOR_EMBEDDING,
    execution_mode=ExecutionMode.SEQUENTIAL
)
test_suite = UnifiedTestSuite(config)
results = test_suite.run_tests(["계약서 작성", "법률 상담"])
```

### 8. 시맨틱 검색 테스트 (신규)
```python
# 시맨틱 검색 테스트
config = TestConfig(
    test_type=TestType.SEMANTIC_SEARCH,
    execution_mode=ExecutionMode.SEQUENTIAL
)
test_suite = UnifiedTestSuite(config)
results = test_suite.run_tests(["부동산 매매 계약", "노동법 관련 조항"])
```

## ⚙️ 설정 파일

### scripts_config.json 예시
```json
{
  "database": {
    "path": "data/lawfirm.db",
    "backup_enabled": true,
    "backup_dir": "data/backups"
  },
  "vector": {
    "embeddings_dir": "data/embeddings",
    "model": "jhgan/ko-sroberta-multitask",
    "batch_size": 32,
    "chunk_size": 1000
  },
  "testing": {
    "results_dir": "results",
    "max_workers": 4,
    "batch_size": 100,
    "timeout_seconds": 300
  },
  "logging": {
    "level": "INFO",
    "dir": "logs"
  }
}
```

## 📊 성능 개선

### 정량적 효과
- **파일 수 감소**: 244개 → 150개 (38% 감소)
- **중복 코드 제거**: 약 30% 감소
- **유지보수 시간**: 50% 단축 예상

### 정성적 효과
- **가독성 향상**: 명확한 구조와 네이밍
- **유지보수성 향상**: 중복 제거 및 표준화
- **확장성 향상**: 모듈화된 구조
- **문서화 개선**: 체계적인 가이드

## 🧪 통합 기능 검증

### 통합 시스템 테스트
```bash
# 모든 통합 기능 검증
python scripts/test_integrated_features.py

# 개별 기능 테스트
python scripts/test_integrated_features.py --test-file-structure
python scripts/test_integrated_features.py --test-base-manager
python scripts/test_integrated_features.py --test-rebuild-manager
python scripts/test_integrated_features.py --test-vector-manager
python scripts/test_integrated_features.py --test-test-suite
```

### 검증 결과 확인
```bash
# 테스트 결과 파일 확인
ls results/integration_test_results_*.json

# 최신 결과 확인
cat results/integration_test_results_$(date +%Y%m%d)*.json | jq '.'
```

## 🔧 문제 해결

### 일반적인 오류

1. **모듈 import 오류**
   ```bash
   # 프로젝트 루트에서 실행
   cd /path/to/LawFirmAI
   python scripts/core/unified_rebuild_manager.py --mode simple
   ```

2. **의존성 오류**
```bash
   # 필요한 패키지 설치
   pip install sentence-transformers faiss-cpu torch
   ```

3. **권한 오류**
   ```bash
   # 로그 디렉토리 권한 확인
   chmod 755 logs/
   ```

### 로그 확인
```bash
# 통합 로그 확인
tail -f logs/unified_rebuild_*.log
tail -f logs/unified_vector_*.log
tail -f logs/unified_test_*.log
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 로그 파일 확인
2. GitHub Issues에 문제 보고
3. 프로젝트 문서 참조

---

**마지막 업데이트**: 2025-10-22  
**관리자**: LawFirmAI 개발팀
**버전**: 2.0 (통합 시스템)