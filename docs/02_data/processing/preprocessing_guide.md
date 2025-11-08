# 데이터 전처리 가이드

수집된 raw 데이터를 벡터 DB에 적합한 형태로 전처리하는 방법에 대한 가이드입니다.

## 전처리 실행

```bash
# 전체 전처리 실행 (모든 데이터 유형)
python scripts/data_processing/preprocess_raw_data.py

# 특정 데이터 유형만 전처리
python scripts/data_processing/batch_preprocess.py --data-type laws
python scripts/data_processing/batch_preprocess.py --data-type precedents
python scripts/data_processing/batch_preprocess.py --data-type constitutional
python scripts/data_processing/batch_preprocess.py --data-type interpretations
python scripts/data_processing/batch_preprocess.py --data-type terms

# 드라이런 모드 (계획만 확인)
python scripts/data_processing/batch_preprocess.py --data-type all --dry-run

# 전처리된 데이터 검증
python scripts/analysis/validate_processed_data.py

# 특정 데이터 유형만 검증
python scripts/analysis/validate_processed_data.py --data-type laws
```

## 전처리 기능

- ✅ **텍스트 정리**: HTML 태그 제거, 공백 정규화, 특수문자 처리
- ✅ **법률 용어 정규화**: 국가법령정보센터 API 기반 용어 표준화
- ✅ **텍스트 청킹**: 벡터 검색에 최적화된 크기로 분할 (200-3000자)
- ✅ **법률 엔티티 추출**: 법률명, 조문, 사건번호, 법원명 등 자동 추출
- ✅ **품질 검증**: 완성도, 정확도, 일관성 자동 검증
- ✅ **중복 제거**: 해시 기반 중복 데이터 자동 제거

## 상세 문서

- [데이터 전처리 계획서](docs/development/raw_data_preprocessing_plan.md)
- [법률 용어 정규화 전략](docs/development/legal_term_normalization_strategy.md)
