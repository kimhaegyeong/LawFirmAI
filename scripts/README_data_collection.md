# 법률 데이터 수집 가이드

이 문서는 LawFirmAI 프로젝트의 법률 데이터 수집 스크립트들의 사용법을 설명합니다.

## 📋 수집 목표

### 법령 데이터 (현행법령 기준)
- **기본법**: 민법, 상법, 형법, 민사소송법, 형사소송법 (5개)
- **특별법**: 노동법, 부동산법, 금융법, 지적재산권법, 개인정보보호법 (5개)
- **행정법**: 행정소송법, 국세기본법, 건축법, 행정절차법, 정보공개법 (5개)
- **사회법**: 사회보장법, 의료법, 교육법, 환경법, 소비자보호법 (5개)
- **총 20개 주요 법령의 모든 조문 및 개정이력**

### 판례 데이터 (국가법령정보센터 기준)
- **판례**: 5,000건 (최근 5년간)
- **헌재결정례**: 1,000건 (최근 5년간)
- **법령해석례**: 2,000건 (최근 3년간)
- **행정심판례**: 1,000건 (최근 3년간)
- **총 9,000건의 법률 관련 판결문**

### 행정규칙 및 자치법규
- **행정규칙**: 1,000건 (주요 부처별)
- **자치법규**: 500건 (주요 지자체별)
- **위원회결정문**: 500건 (주요 위원회별)
- **총 2,000건의 하위법령**

### 기타 법률 데이터
- **조약**: 100건 (주요 조약)

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경변수 설정
export LAW_OPEN_API_OC="your_email_id"

# 또는 .env 파일 생성
echo "LAW_OPEN_API_OC=your_email_id" > .env
```

### 2. 전체 데이터 수집 (권장)

```bash
# 모든 데이터 타입 수집
python scripts/master_data_collector.py --oc your_email_id --mode all

# 우선순위 데이터만 수집 (핵심 데이터)
python scripts/master_data_collector.py --oc your_email_id --mode priority

# 추가 데이터만 수집 (보조 데이터)
python scripts/master_data_collector.py --oc your_email_id --mode additional
```

### 3. 개별 데이터 타입 수집

```bash
# 특정 데이터 타입만 수집
python scripts/master_data_collector.py --oc your_email_id --mode single --types laws precedents

# 개별 스크립트 실행
python scripts/collect_laws.py
python scripts/collect_precedents.py
python scripts/collect_constitutional_decisions.py
python scripts/legal_interpretation/collect_legal_interpretations.py
python scripts/collect_administrative_appeals.py
python scripts/collect_administrative_rules.py
python scripts/collect_local_ordinances.py
python scripts/collect_committee_decisions.py
python scripts/collect_treaties.py
```

## 📁 스크립트 목록

### 마스터 스크립트
- **`master_data_collector.py`**: 모든 데이터 타입을 통합 관리하는 마스터 스크립트

### 개별 수집 스크립트
- **`collect_laws.py`**: 법령 데이터 수집 (20개 주요 법령)
- **`collect_precedents.py`**: 판례 데이터 수집 (5,000건)
- **`collect_constitutional_decisions.py`**: 헌재결정례 수집 (1,000건)
- **`legal_interpretation/collect_legal_interpretations.py`**: 법령해석례 수집 (2,000건)
- **`collect_administrative_appeals.py`**: 행정심판례 수집 (1,000건)
- **`collect_administrative_rules.py`**: 행정규칙 수집 (1,000건)
- **`collect_local_ordinances.py`**: 자치법규 수집 (500건)
- **`collect_committee_decisions.py`**: 위원회결정문 수집 (500건)
- **`collect_treaties.py`**: 조약 수집 (100건)

### 유틸리티 스크립트
- **`collect_data_only.py`**: 간단한 데이터 수집 (JSON 저장만)
- **`simple_data_collector.py`**: 복잡한 의존성 없이 API만으로 수집

## 📊 수집 결과

### 데이터 저장 위치
```
data/
├── raw/                    # 원본 데이터
│   ├── laws/              # 법령 데이터
│   ├── precedents/        # 판례 데이터
│   ├── constitutional_decisions/  # 헌재결정례
│   ├── legal_interpretations/     # 법령해석례
│   ├── administrative_appeals/    # 행정심판례
│   ├── administrative_rules/      # 행정규칙
│   ├── local_ordinances/          # 자치법규
│   ├── committee_decisions/       # 위원회결정문
│   └── treaties/                  # 조약
└── master_collection_report.json  # 수집 보고서
```

### 수집 보고서
각 수집 완료 후 다음 정보가 포함된 보고서가 생성됩니다:
- 수집 일시 및 소요 시간
- 데이터 타입별 수집 결과
- API 요청 수 및 사용량
- 오류 목록 (있는 경우)

## ⚠️ 주의사항

### API 제한
- **일일 요청 제한**: 1,000회
- **요청 간격**: 1초 이상 권장
- **OC 파라미터**: 사용자 이메일 ID 필수

### 수집 시간
- **전체 수집**: 약 2-4시간 (API 제한에 따라)
- **우선순위 데이터**: 약 1-2시간
- **개별 데이터 타입**: 10-30분

### 메모리 사용량
- **권장 메모리**: 8GB 이상
- **디스크 공간**: 10GB 이상 (압축 전)

## 🔧 문제 해결

### 일반적인 오류

1. **API 요청 한도 초과**
   ```bash
   # 다음 날 다시 시도하거나 OC 파라미터 변경
   export LAW_OPEN_API_OC="another_email_id"
   ```

2. **스크립트 실행 실패**
   ```bash
   # 로그 확인
   tail -f logs/master_data_collector.log
   ```

3. **메모리 부족**
   ```bash
   # 개별 스크립트로 분할 실행
   python scripts/collect_laws.py
   python scripts/collect_precedents.py
   ```

### 로그 확인
```bash
# 마스터 수집기 로그
tail -f logs/master_data_collector.log

# 개별 스크립트 로그
tail -f logs/collect_laws.log
tail -f logs/collect_precedents.log
```

## 📈 성능 최적화

### 병렬 수집
```bash
# 여러 터미널에서 동시 실행 (API 제한 고려)
python scripts/collect_laws.py &
python scripts/collect_precedents.py &
python scripts/collect_constitutional_decisions.py &
```

### 배치 수집
```bash
# 특정 시간대에 자동 실행
crontab -e
# 매일 새벽 2시에 우선순위 데이터 수집
0 2 * * * cd /path/to/LawFirmAI && python scripts/master_data_collector.py --oc your_email_id --mode priority
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 로그 파일 확인
2. GitHub Issues에 문제 보고
3. 프로젝트 문서 참조

---

**참고**: 이 가이드는 LawFirmAI 프로젝트의 데이터 수집 시스템을 위한 것입니다. 
국가법령정보센터 OpenAPI의 이용약관을 준수하여 사용하시기 바랍니다.
