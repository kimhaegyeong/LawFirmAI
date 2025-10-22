# Deprecated Scripts

이 폴더에는 새로운 통합 시스템으로 대체된 기존 스크립트들이 보관되어 있습니다.

## 📋 대체된 스크립트 목록

### 데이터 재구축 스크립트
- `full_raw_data_rebuild.py` → `core/unified_rebuild_manager.py --mode full`
- `real_data_rebuild.py` → `core/unified_rebuild_manager.py --mode real`
- `real_raw_data_rebuild.py` → `core/unified_rebuild_manager.py --mode real`
- `simple_database_rebuild.py` → `core/unified_rebuild_manager.py --mode simple`
- `rebuild_database_from_raw.py` → `core/unified_rebuild_manager.py --mode full`

### 벡터 빌딩 스크립트
- `efficient_vector_builder.py` → `core/unified_vector_manager.py --mode full`
- `build_ml_enhanced_vector_db.py` → `core/unified_vector_manager.py --mode full`
- `build_ml_enhanced_vector_db_optimized.py` → `core/unified_vector_manager.py --mode full`
- `build_ml_enhanced_vector_db_cpu_optimized.py` → `core/unified_vector_manager.py --mode cpu_optimized`
- `build_resumable_vector_db.py` → `core/unified_vector_manager.py --mode resumable`
- `incremental_vector_builder.py` → `core/unified_vector_manager.py --mode incremental`
- `incremental_precedent_vector_builder.py` → `core/unified_vector_manager.py --mode incremental`
- `rebuild_improved_vector_db.py` → `core/unified_vector_manager.py --mode full`

### 테스트 스크립트
- `massive_test_runner.py` → `testing/unified_test_suite.py --test-type massive`
- `massive_test_query_generator.py` → `testing/unified_test_suite.py` (쿼리 파일 사용)
- `integrated_massive_test_system.py` → `testing/unified_test_suite.py --test-type integration`
- `test_performance_optimization.py` → `testing/unified_test_suite.py --test-type performance`
- `test_*` 접두사 파일들 → `testing/unified_test_suite.py` (적절한 test-type 사용)

### 분석 스크립트
- `analyze_*` 접두사 파일들 → `analysis/` 폴더로 이동
- `comprehensive_*` 접두사 파일들 → `analysis/` 폴더로 이동
- `create_*` 접두사 파일들 → `analysis/` 폴더로 이동

## ⚠️ 주의사항

1. **이 스크립트들은 더 이상 사용되지 않습니다**
2. **새로운 통합 시스템을 사용하세요**
3. **필요한 경우 참고용으로만 사용하세요**
4. **향후 버전에서 완전히 제거될 예정입니다**

## 🔄 마이그레이션 가이드

기존 스크립트에서 새 시스템으로 전환하는 방법은 메인 README.md를 참조하세요.

## 📅 제거 예정일

- **2025년 12월**: 완전 제거 예정
- **2025년 11월**: 경고 메시지 추가 예정

---

**마지막 업데이트**: 2025-10-22  
**상태**: 사용 중단 (Deprecated)
