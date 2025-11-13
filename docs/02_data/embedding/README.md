# 벡터 임베딩 (Vector Embedding)

이 섹션은 LawFirmAI 프로젝트의 벡터 임베딩 생성, 관리 및 관련 기술에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[벡터 임베딩 가이드](embedding_guide.md)**: 벡터 임베딩 생성 및 관리 방법
- **[외부 인덱스 설정 가이드](external_index_config_guide.md)**: 외부 FAISS 인덱스 사용 설정 방법
- **[버전 관리 사용법](version_management_guide.md)**: 벡터스토어 버전 관리 방법
- **[BGE-M3 Korean 사용 가이드](../archive/bge_m3_korean_usage_guide.md)**: BGE-M3 Korean 모델 사용법 (보관됨)

## 🎯 현재 상태 (2025-10-19)

### ✅ 활성 모델
- **ml_enhanced_ko_sroberta**: 법령 데이터 벡터화 (4,321개 조문)
- **ml_enhanced_ko_sroberta_precedents**: 판례 데이터 벡터화 (33,598개 문서) - Phase 4 업데이트

### ❌ 개발보류 모델
- **ml_enhanced_bge_m3**: 메모리 사용량 과다로 개발보류

### 📊 성능 지표 (Phase 4 최적화 후)
- **벡터 검색 성공률**: 100% (모든 테스트 쿼리 성공)
- **메모리 최적화**: PQ 양자화로 대폭 절약
- **검색 속도**: 평균 0.043초 (99.8% 향상)
- **전체 벡터화 데이터**: 33,598개 문서 (행정 798개, 특허 998개 추가)
- **인덱스 최적화**: IVF + PQ 양자화 적용

### 🆕 Phase 4 신규 기능
- **IVF 인덱스**: 대용량 데이터를 위한 Inverted File Index
- **PQ 양자화**: Product Quantization으로 메모리 사용량 최적화
- **성능 모니터링**: 실시간 벡터 검색 성능 추적
- **캐싱 시스템**: 자주 검색되는 쿼리 결과 캐싱

## 🔗 관련 섹션

- [02_Data_Processing](../02_data/processing/README.md): 데이터 전처리
- [03_RAG_System](../03_rag_system/README.md): RAG 시스템 아키텍처
- [04_Models_Performance](../04_models/performance/README.md): 모델 성능 분석
