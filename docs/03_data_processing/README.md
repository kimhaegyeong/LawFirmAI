# 데이터 전처리 (Data Processing)

이 섹션은 LawFirmAI 프로젝트의 데이터 전처리 파이프라인과 관련 기술에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[데이터 전처리 가이드](preprocessing_guide.md)**: Assembly 법률 데이터 전처리 파이프라인 v4.1의 상세 가이드
- **[증분 전처리 파이프라인 가이드](incremental_pipeline_guide.md)**: 자동화된 증분 전처리 시스템 사용법

## 🚀 주요 기능

### 증분 전처리 시스템
- **자동 데이터 감지**: 새로운 파일만 자동으로 감지하고 처리
- **체크포인트 시스템**: 중단 시 이어서 처리 가능
- **메모리 최적화**: 대용량 파일도 효율적으로 처리

### 통합 파이프라인 오케스트레이터
- **원스톱 처리**: 데이터 감지 → 전처리 → 벡터 임베딩 → DB 저장
- **자동화된 워크플로우**: 수동 개입 없이 전체 파이프라인 실행
- **오류 복구**: 실패한 파일은 별도 추적하여 재처리 가능

### ML-Enhanced Parsing (선택적)
- **Machine Learning Model**: RandomForest-based article boundary classification
- **Hybrid Scoring**: ML model (50%) + Rule-based (50%) combination
- **Fallback System**: ML 모델이 없으면 규칙 기반 파서로 안정적 동작

## 🔗 관련 섹션

- [02_Data_Collection](../02_data_collection/README.md): 데이터 수집 방법
- [04_Vector_Embedding](../04_vector_embedding/README.md): 벡터 임베딩 생성
- [10_Technical_Reference](../10_technical_reference/README.md): 데이터베이스 스키마
