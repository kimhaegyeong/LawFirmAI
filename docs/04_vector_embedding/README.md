# 벡터 임베딩 (Vector Embedding)

이 섹션은 LawFirmAI 프로젝트의 벡터 임베딩 생성, 관리 및 관련 기술에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[벡터 임베딩 가이드](embedding_guide.md)**: 벡터 임베딩 생성 및 관리 방법
- **[BGE-M3 Korean 사용 가이드](../archive/bge_m3_korean_usage_guide.md)**: BGE-M3 Korean 모델 사용법 (보관됨)

## 🎯 현재 상태 (2025-10-17)

### ✅ 활성 모델
- **ml_enhanced_ko_sroberta**: 법령 데이터 벡터화 (4,321개 조문)
- **ml_enhanced_ko_sroberta_precedents**: 판례 데이터 벡터화 (6,285개 청크)

### ❌ 개발보류 모델
- **ml_enhanced_bge_m3**: 메모리 사용량 과다로 개발보류

### 📊 성능 지표
- **벡터 검색 성공률**: 100% (5/5 테스트 쿼리 성공)
- **메모리 최적화**: 34.3% 절약 (Float16 양자화)
- **검색 속도**: 평균 0.124초
- **전체 벡터화 데이터**: 10,606개 문서

## 🔗 관련 섹션

- [03_Data_Processing](../03_data_processing/README.md): 데이터 전처리
- [05_RAG_System](../05_rag_system/README.md): RAG 시스템 아키텍처
- [06_Models_Performance](../06_models_performance/README.md): 모델 성능 분석

---

**LawFirmAI 개발팀**
