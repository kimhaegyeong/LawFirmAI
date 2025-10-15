# Data Processing Scripts

법률 데이터의 전처리, 정제, 최적화를 담당하는 스크립트들입니다.

## 📁 파일 목록

### 핵심 전처리 스크립트
- **`preprocess_raw_data.py`** (24.9 KB) - 원본 법률 데이터 전처리
- **`quality_improved_preprocess.py`** (17.6 KB) - 품질 개선된 전처리 파이프라인
- **`optimize_law_data.py`** (6.8 KB) - 법률 데이터 최적화

### 배치 처리 스크립트
- **`batch_preprocess.py`** (2.0 KB) - 배치 전처리 실행
- **`batch_update_law_content.py`** (12.5 KB) - 배치 법률 내용 업데이트
- **`update_law_content.py`** (7.5 KB) - 개별 법률 내용 업데이트

### 데이터 정제 스크립트
- **`refine_law_data_from_html.py`** (10.5 KB) - HTML에서 법률 데이터 정제
- **`add_missing_data_types.py`** (10.5 KB) - 누락된 데이터 타입 추가

### 파이프라인 관리
- **`run_data_pipeline.py`** (14.3 KB) - 전체 데이터 처리 파이프라인 실행
- **`setup_env.py`** (2.5 KB) - 데이터 처리 환경 설정

## 🚀 사용법

### 전체 데이터 처리 파이프라인
```bash
# 전체 파이프라인 실행
python scripts/data_processing/run_data_pipeline.py

# 환경 설정
python scripts/data_processing/setup_env.py
```

### 원본 데이터 전처리
```bash
# 기본 전처리
python scripts/data_processing/preprocess_raw_data.py

# 품질 개선된 전처리
python scripts/data_processing/quality_improved_preprocess.py
```

### 배치 처리
```bash
# 배치 전처리
python scripts/data_processing/batch_preprocess.py

# 배치 업데이트
python scripts/data_processing/batch_update_law_content.py
```

### 데이터 정제
```bash
# HTML 데이터 정제
python scripts/data_processing/refine_law_data_from_html.py

# 데이터 최적화
python scripts/data_processing/optimize_law_data.py
```

## 📊 처리 단계

### 1. 원본 데이터 수집
- 국회 법률 데이터 수집
- HTML 형태의 원본 데이터

### 2. 전처리
- HTML 태그 제거
- 특수 문자 정리
- 텍스트 정규화

### 3. 품질 개선
- 제어 문자 제거
- 텍스트 정제
- 구조화된 데이터 변환

### 4. 최적화
- 데이터 압축
- 인덱싱 최적화
- 메타데이터 추가

## 🔧 설정

### 환경 변수
```bash
# 데이터 디렉토리 설정
export DATA_DIR="data/raw"
export PROCESSED_DIR="data/processed"

# 로그 레벨 설정
export LOG_LEVEL="INFO"
```

### 입력 데이터 형식
- **원본 데이터**: HTML 형태의 법률 문서
- **메타데이터**: JSON 형태의 법률 정보
- **구조화 데이터**: 파싱된 법률 조문

### 출력 데이터 형식
- **정제된 텍스트**: 깨끗한 법률 텍스트
- **구조화된 JSON**: 파싱된 법률 데이터
- **메타데이터**: 처리 정보 및 품질 지표

## 📈 성능 지표

### 처리 속도
- **원본 전처리**: ~100 파일/분
- **품질 개선**: ~50 파일/분
- **배치 처리**: ~200 파일/분

### 품질 지표
- **텍스트 정제율**: 99.9%
- **구조화 정확도**: 95%+
- **메타데이터 완성도**: 98%+

## 🛠️ 트러블슈팅

### 일반적인 문제
1. **메모리 부족**: 배치 크기 조정
2. **인코딩 오류**: UTF-8 인코딩 확인
3. **파일 권한**: 읽기/쓰기 권한 확인

### 로그 확인
```bash
# 처리 로그 확인
tail -f logs/data_processing.log

# 에러 로그 확인
grep "ERROR" logs/data_processing.log
```

---

**마지막 업데이트**: 2025-10-15  
**관리자**: LawFirmAI 개발팀
