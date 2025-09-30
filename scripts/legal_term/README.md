# 법률 용어 수집 시스템

국가법령정보센터 OpenAPI를 활용한 법률 용어 수집 및 관리 시스템입니다.

## 📁 구조

```
scripts/legal_term/
├── __init__.py              # 모듈 초기화
├── term_collector.py        # 용어 수집기
├── synonym_manager.py       # 동의어 관리자
├── term_validator.py        # 용어 검증기
└── README.md               # 이 파일
```

## 🚀 사용법

### 기본 사용법

```bash
# 환경변수 설정
export LAW_OPEN_API_OC="your_email@example.com"

# 전체 용어 수집 (기본)
python scripts/collect_legal_terms.py

# 최대 1000개 용어 수집
python scripts/collect_legal_terms.py --max-terms 1000

# 카테고리별 수집
python scripts/collect_legal_terms.py --collection-type categories

# 키워드별 수집
python scripts/collect_legal_terms.py --collection-type keywords

# 기존 사전만 검증
python scripts/collect_legal_terms.py --validate-only

# 수집 후 JSON으로 내보내기
python scripts/collect_legal_terms.py --export json

# 수집 후 CSV로 내보내기
python scripts/collect_legal_terms.py --export csv
```

## 📁 출력 파일

### 수집 결과
- `data/legal_terms/legal_term_dictionary.json`: 용어 사전
- `logs/legal_term_collection.log`: 수집 로그
- `logs/validation_report.json`: 검증 보고서

### 내보내기 파일
- `data/legal_terms/exported_terms.json`: JSON 형식 내보내기
- `data/legal_terms/exported_terms.csv`: CSV 형식 내보내기

## ⚙️ 설정

### 환경 변수
```bash
export LAW_OPEN_API_OC="your_email@example.com"  # 필수
```

### API 설정
```python
config = TermCollectionConfig()
config.batch_size = 100                    # 배치 크기
config.delay_between_requests = 0.05       # 요청 간 지연 시간
config.max_retries = 3                     # 최대 재시도 횟수
config.timeout = 30                        # 타임아웃 시간
```

## 🚨 문제 해결

### 일반적인 문제

1. **API 연결 실패**
   ```bash
   # 환경변수 확인
   echo $LAW_OPEN_API_OC
   
   # 재설정
   export LAW_OPEN_API_OC="your_email@example.com"
   ```

2. **메모리 부족**
   ```bash
   # 용어 수 제한
   python scripts/collect_legal_terms.py --max-terms 1000
   ```

3. **수집 속도 저하**
   ```python
   # 설정 조정
   config.delay_between_requests = 0.01  # 지연 시간 단축
   config.batch_size = 200               # 배치 크기 증가
   ```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f logs/legal_term_collection.log

# 검증 보고서 확인
cat logs/validation_report.json
```
