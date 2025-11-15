# 자동 완료 스크립트 사용 가이드

## 개요

`auto_complete_dynamic_chunking.py` 스크립트는 재임베딩 완료를 자동으로 감지하고, 완료되면 FAISS 인덱스를 빌드한 후 다음 단계를 안내합니다.

## 실행 방법

### 방법 1: 포그라운드 실행 (권장)

진행 상황을 실시간으로 확인할 수 있습니다:

```bash
python scripts/auto_complete_dynamic_chunking.py \
    --db data/lawfirm_v2.db \
    --version-id 5 \
    --check-interval 300 \
    --timeout 14400
```

### 방법 2: 백그라운드 실행 (Windows)

```powershell
# PowerShell에서 실행
Start-Process python -ArgumentList "scripts/auto_complete_dynamic_chunking.py --db data/lawfirm_v2.db --version-id 5 --check-interval 300 --timeout 14400" -WindowStyle Hidden
```

### 방법 3: 로그 파일로 리다이렉트

```bash
python scripts/auto_complete_dynamic_chunking.py \
    --db data/lawfirm_v2.db \
    --version-id 5 \
    --check-interval 300 \
    --timeout 14400 \
    > logs/auto_complete.log 2>&1
```

## 파라미터 설명

- `--db`: 데이터베이스 경로 (기본값: `data/lawfirm_v2.db`)
- `--version-id`: 임베딩 버전 ID (필수)
- `--check-interval`: 재임베딩 완료 확인 간격 (초, 기본값: 300 = 5분)
- `--timeout`: 최대 대기 시간 (초, 기본값: 14400 = 4시간)
- `--threshold`: 완료 임계값 (기본값: 0.99 = 99%)
- `--skip-wait`: 대기 건너뛰고 바로 FAISS 인덱스 빌드

## 동작 방식

1. **재임베딩 완료 확인**: `check_interval` 간격으로 재임베딩 진행률 확인
2. **완료 감지**: 진행률이 `threshold` 이상이면 완료로 간주
3. **FAISS 인덱스 빌드**: 완료되면 자동으로 FAISS 인덱스 빌드
4. **다음 단계 안내**: 성능 비교 및 버전 활성화 방법 안내

## 모니터링

### 재임베딩 진행 상황 확인

```bash
python scripts/monitor_re_embedding_progress.py \
    --db data/lawfirm_v2.db \
    --version-id 5
```

### 완료 여부 확인

```bash
python scripts/check_re_embedding_complete.py \
    --db data/lawfirm_v2.db \
    --version-id 5
```

### 로그 확인 (백그라운드 실행 시)

```bash
# Windows PowerShell
Get-Content logs/auto_complete_dynamic_chunking.log -Tail 20

# Linux/Mac
tail -20 logs/auto_complete_dynamic_chunking.log
```

### 프로세스 확인

```powershell
# Windows PowerShell
Get-Process python | Where-Object {$_.CommandLine -like "*auto_complete*"}
```

## 예상 동작 시간

- **재임베딩**: 약 2-4시간 (CPU 기준)
- **확인 간격**: 5분마다 확인 (기본값)
- **FAISS 인덱스 빌드**: 약 1-2분
- **전체 소요 시간**: 재임베딩 완료 시간 + 약 2분

## 완료 후 자동 실행되는 작업

1. ✅ 재임베딩 완료 확인
2. ✅ FAISS 인덱스 빌드
3. ✅ FAISS 버전 활성화
4. ✅ 다음 단계 안내 (성능 비교, 버전 활성화)

## 수동으로 다음 단계 진행

자동 완료 스크립트가 실행되지 않거나 문제가 있는 경우:

### 1. FAISS 인덱스 빌드

```bash
python scripts/build_faiss_index_for_dynamic_chunking.py \
    --db data/lawfirm_v2.db \
    --version-id 5
```

### 2. 성능 비교

```bash
python scripts/test_performance_monitoring.py \
    --db data/lawfirm_v2.db \
    --version1 v1.0.0-standard-standard \
    --version2 v2.0.0-dynamic-dynamic
```

### 3. 버전 활성화

```bash
python scripts/utils/embedding_version_switcher.py \
    --action activate \
    --version-id 5 \
    --db data/lawfirm_v2.db
```

## 문제 해결

### 스크립트가 실행되지 않는 경우

1. Python 경로 확인
2. 필요한 모듈 설치 확인
3. 데이터베이스 경로 확인

### 재임베딩이 완료되지 않는 경우

1. 재임베딩 프로세스 확인
2. 타임아웃 시간 증가 (`--timeout` 파라미터)
3. 수동으로 재임베딩 재시작

### FAISS 인덱스 빌드 실패

1. 충분한 디스크 공간 확인
2. 메모리 확인
3. 로그 파일 확인 (`logs/auto_complete_dynamic_chunking_error.log`)

## 권장 사항

1. **포그라운드 실행**: 진행 상황을 실시간으로 확인
2. **충분한 타임아웃**: 재임베딩 완료 시간을 고려하여 타임아웃 설정
3. **정기적인 모니터링**: 재임베딩 진행 상황을 주기적으로 확인
4. **로그 확인**: 문제 발생 시 로그 파일 확인

