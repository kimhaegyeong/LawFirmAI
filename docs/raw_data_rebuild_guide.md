# Raw 데이터 재적재 시스템 사용 가이드

## 개요

이 시스템은 기존 데이터베이스의 품질 문제를 해결하기 위해 Raw 데이터를 사용하여 데이터베이스를 완전히 재구축하는 종합적인 솔루션입니다.

## 🚀 빠른 시작

### 1. 전체 프로세스 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
python scripts/rebuild_database_from_raw.py
```

이 명령어는 다음 4단계를 자동으로 실행합니다:
1. 기존 데이터 백업 및 정리
2. Raw 데이터 전처리
3. 데이터베이스 적재
4. 품질 검증 및 최적화

### 2. 개별 단계 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# Phase 1: 데이터 백업 및 정리
python scripts/database/clear_existing_data.py

# Phase 2: Raw 데이터 전처리
python scripts/data_processing/enhanced_raw_preprocessor.py

# Phase 3: 데이터베이스 적재
python scripts/data_processing/enhanced_import_manager.py

# Phase 4: 품질 검증 및 최적화
python scripts/data_processing/quality_validation_system.py
```

## 📋 시스템 구성

### Phase 1: 데이터 백업 및 정리
- **파일**: `scripts/database/clear_existing_data.py`
- **기능**: 
  - 기존 데이터베이스 백업
  - 기존 데이터 완전 삭제
  - 데이터베이스 구조 확인

### Phase 2: 향상된 Raw 데이터 전처리
- **파일**: `scripts/data_processing/enhanced_raw_preprocessor.py`
- **기능**:
  - 하이브리드 파싱 (규칙 기반 + ML)
  - 품질 검증 및 점수 계산
  - 법률 및 판례 데이터 전처리

### Phase 3: 향상된 데이터베이스 적재
- **파일**: `scripts/data_processing/enhanced_import_manager.py`
- **기능**:
  - 전처리된 데이터 데이터베이스 적재
  - FTS 인덱스 생성
  - 데이터베이스 최적화

### Phase 4: 품질 검증 및 최적화
- **파일**: `scripts/data_processing/quality_validation_system.py`
- **기능**:
  - 데이터 품질 검증
  - 데이터 무결성 검사
  - 성능 최적화

## 📊 예상 결과

### 데이터 규모
- **법률**: 9,000+ 개
- **조문**: 290,000+ 개
- **판례**: 16,000+ 개

### 품질 향상
- **평균 품질 점수**: 0.8+ (기존 0.3에서 향상)
- **파싱 정확도**: 90%+ (기존 60%에서 향상)
- **구조적 완성도**: 95%+

### 처리 시간
- **전체 프로세스**: 4-6시간
- **Phase 1**: 5-10분
- **Phase 2**: 2-3시간
- **Phase 3**: 1-2시간
- **Phase 4**: 30분-1시간

## 📁 생성되는 파일

### 리포트 파일
- `data/rebuild_database_report.json` - 전체 프로세스 상세 리포트
- `data/rebuild_summary.json` - 요약 리포트
- `data/quality_validation_report.json` - 품질 검증 리포트
- `data/import_report.json` - 데이터 임포트 리포트
- `data/preprocessing_report.json` - 전처리 리포트

### 로그 파일
- `logs/rebuild_database.log` - 전체 프로세스 로그
- `logs/data_backup_clear.log` - 백업 및 정리 로그
- `logs/enhanced_preprocessing.log` - 전처리 로그
- `logs/enhanced_import.log` - 임포트 로그
- `logs/quality_validation.log` - 품질 검증 로그

### 백업 파일
- `data/lawfirm.db.backup_YYYYMMDD_HHMMSS` - 기존 데이터베이스 백업

## ⚠️ 주의사항

### 실행 전 확인사항
1. **디스크 공간**: 최소 10GB 이상의 여유 공간 필요
2. **메모리**: 최소 8GB RAM 권장
3. **Raw 데이터**: `data/raw/` 디렉토리에 Raw 데이터 존재 확인
4. **백업**: 중요한 데이터는 별도 백업 권장

### 실행 중 주의사항
1. **중단 금지**: 프로세스 중단 시 데이터 손상 가능
2. **동시 실행 금지**: 여러 인스턴스 동시 실행 금지
3. **로그 모니터링**: 오류 발생 시 로그 확인

## 🔧 문제 해결

### 일반적인 문제

#### 1. 메모리 부족 오류
```bash
# 해결방법: 배치 크기 조정
export BATCH_SIZE=50  # 기본값: 100
```

#### 2. 디스크 공간 부족
```bash
# 해결방법: 불필요한 파일 정리
rm -rf data/cache/*
rm -rf logs/*.log.old
```

#### 3. 데이터베이스 잠금 오류
```bash
# 해결방법: 프로세스 확인 및 종료
ps aux | grep python
kill -9 <PID>
```

### 로그 확인 방법

```bash
# 실시간 로그 모니터링
tail -f logs/rebuild_database.log

# 특정 단계 로그 확인
grep "PHASE 1" logs/rebuild_database.log
grep "ERROR" logs/rebuild_database.log
```

## 📈 성능 최적화

### 시스템 최적화
1. **SSD 사용**: 데이터베이스 파일을 SSD에 저장
2. **메모리 증설**: 16GB 이상 권장
3. **CPU 코어**: 멀티코어 활용

### 설정 최적화
```python
# 배치 크기 조정
BATCH_SIZE = 100  # 기본값
BATCH_SIZE = 200  # 고성능 시스템

# 병렬 처리 설정
MAX_WORKERS = 4  # CPU 코어 수에 맞게 조정
```

## 🎯 품질 개선 효과

### Before (기존 시스템)
- 평균 품질 점수: 0.3
- 파싱 정확도: 60%
- 중복 데이터: 15-20%
- 구조적 완성도: 70%

### After (개선된 시스템)
- 평균 품질 점수: 0.8+
- 파싱 정확도: 90%+
- 중복 데이터: 5% 미만
- 구조적 완성도: 95%+

## 📞 지원

문제가 발생하거나 질문이 있는 경우:
1. 로그 파일 확인
2. 리포트 파일 검토
3. GitHub Issues 등록

---

**주의**: 이 프로세스는 기존 데이터를 완전히 대체합니다. 실행 전 반드시 백업을 확인하세요.
