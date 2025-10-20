# 데이터 수집 (Data Collection)

이 섹션은 LawFirmAI 프로젝트의 법률 데이터 수집 방법과 전략에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[데이터 수집 가이드](data_collection_guide.md)**: 법률 데이터 수집의 전체적인 방법론과 전략을 설명합니다.
- **[Assembly 데이터 수집 가이드](assembly_data_collection_guide.md)**: 국회 법률정보시스템을 통한 웹 스크래핑 기반 데이터 수집 방법을 설명합니다.

## 🔗 관련 섹션

- [03_Data_Processing](../03_data_processing/README.md): 수집된 데이터의 전처리 방법
- [10_Technical_Reference](../10_technical_reference/README.md): 데이터베이스 스키마 및 기술 참조

## 📁 주요 스크립트 위치

- **통합 파이프라인**: `scripts/data_processing/run_data_pipeline.py`
- **데이터 수집 전용**: `scripts/data_collection/qa_generation/collect_data_only.py`
- **Assembly 법령 수집**: `scripts/data_collection/assembly/collect_laws.py`
- **Assembly 판례 수집**: `scripts/data_collection/assembly/collect_precedents_by_category.py`
- **날짜별 판례 수집**: `scripts/data_collection/precedent/collect_by_date.py`

---

**LawFirmAI 개발팀**
