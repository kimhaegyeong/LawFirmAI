# 성능 최적화 스크립트

이 디렉토리는 LawFirmAI 시스템의 검색 성능 최적화와 관련된 스크립트들을 포함합니다.

## 파일 목록

### 1. `optimize_search_performance.py`
- **목적**: 검색 성능 분석 및 최적화 제안
- **기능**: 
  - 다양한 검색 시나리오 테스트 (단일 키워드, 복합 키워드, 긴 문장)
  - 벡터 인덱스 성능 분석
  - 메모리 사용량 확인
  - 최적화 제안 생성
- **사용법**: `python optimize_search_performance.py`

### 2. `create_optimized_vector_index.py`
- **목적**: 최적화된 벡터 인덱스 생성
- **기능**:
  - IVF (Inverted File Index) 인덱스 생성
  - PQ (Product Quantization) 양자화 인덱스 생성
  - 성능 비교 테스트
  - 메모리 사용량 비교
- **사용법**: `python create_optimized_vector_index.py`

### 3. `search_optimization_results.json`
- **목적**: 검색 성능 최적화 결과 데이터
- **내용**: 
  - 단일 키워드, 복합 키워드, 긴 문장별 검색 성능 데이터
  - 평균 검색 시간 및 점수 통계
- **생성**: `optimize_search_performance.py` 실행 시 자동 생성

### 4. `optimized_search_config.json`
- **목적**: 최적화된 검색 설정 파일
- **내용**:
  - 벡터 검색 설정
  - 하이브리드 검색 설정
  - 성능 최적화 설정
  - 인덱스 최적화 설정
- **생성**: `optimize_search_performance.py` 실행 시 자동 생성

## 사용 순서

1. **성능 분석**: `optimize_search_performance.py` 실행
2. **인덱스 최적화**: `create_optimized_vector_index.py` 실행
3. **결과 확인**: 생성된 JSON 파일들로 성능 개선 확인

## 주의사항

- 대용량 데이터 처리 시 충분한 메모리 확보 필요
- 인덱스 생성은 시간이 오래 걸릴 수 있음
- 기존 인덱스 백업 후 최적화 작업 수행 권장
