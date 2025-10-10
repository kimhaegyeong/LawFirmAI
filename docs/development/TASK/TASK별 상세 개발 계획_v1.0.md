# LawFirmAI TASK별 상세 개발 계획 v1.0

## 📋 TASK 개요

본 문서는 LawFirmAI 프로젝트의 개발계획_v1.0.md를 기반으로 각 주차별 TASK를 세분화하고 구체적인 실행 계획을 제시합니다.

## 🎯 프로젝트 진행 현황 (2025-10-10 업데이트)

### ✅ 완료된 TASK
- **TASK 1.1**: 시스템 아키텍처 설계 ✅
- **TASK 1.2**: 데이터베이스 스키마 설계 ✅
- **TASK 1.3**: 개발 환경 구성 ✅
- **TASK 2.1**: 법률 데이터 수집 시스템 구축 ✅
- **TASK 2.2**: 데이터 전처리 및 구조화 ✅
- **TASK 2.3**: 벡터DB 구축 파이프라인 ✅
- **TASK 2.4**: Q&A 데이터셋 생성 (법령/판례 기반) ✅

### 🔄 진행 중인 TASK
- 없음

### 📊 전체 진행률
- **완료**: 7개 TASK
- **진행률**: 87.5% (7/8)
- **다음 마일스톤**: TASK 2.5 통합 데이터 파이프라인 구축
- **최신 성과**: LLM 기반 Q&A 생성 시스템 구축 완료 (2025-10-10)

---

## 🗓️ Week 1-2: 프로젝트 설계 및 환경 구성

### TASK 1.1: 시스템 아키텍처 설계
**담당자**: 시스템 아키텍트  
**예상 소요시간**: 3일  
**우선순위**: Critical

#### 세부 작업
- [X] 마이크로서비스 아키텍처 다이어그램 작성
- [X] 모듈 간 인터페이스 정의
- [X] 데이터 플로우 설계
- [X] API 스펙 초안 작성

#### 산출물
- `docs/architecture/system_architecture.md` ✅
- `docs/architecture/module_interfaces.md` ✅
- `docs/api/api_specification.md` ✅

#### 완료 기준
- [X] 아키텍처 다이어그램 검토 완료
- [X] 모듈 인터페이스 명세서 완성
- [X] 기술 스택 최종 확정

---

### TASK 1.2: 데이터베이스 스키마 설계
**담당자**: 데이터베이스 설계자  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [X] ERD 다이어그램 작성
- [X] 테이블 스키마 정의
- [X] 인덱스 전략 수립
- [X] 데이터 마이그레이션 계획

#### 산출물
- `docs/database_schema.md` ✅
- `source/data/database.py` ✅
- `scripts/migration/` ✅

#### 완료 기준
- [X] 모든 테이블 스키마 정의 완료
- [X] 인덱스 최적화 방안 수립
- [X] 데이터베이스 초기화 스크립트 작성

---

### TASK 1.3: 개발 환경 구성
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [X] Python 가상환경 설정
- [X] 의존성 관리 (requirements.txt)
- [X] Docker 환경 구성
- [ ] CI/CD 파이프라인 구축

#### 산출물
- `requirements.txt` ✅
- `Dockerfile` ✅
- `docker-compose.yml` ✅
- `.github/workflows/` ⏳

#### 완료 기준
- [X] 로컬 개발 환경 구축 완료
- [X] Docker 컨테이너 정상 작동 확인
- [ ] CI/CD 파이프라인 구축 완료

---

### TASK 1.4: 모델 성능 벤치마크
**담당자**: ML 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [X] KoBART vs KoGPT-2 성능 비교
- [X] FAISS vs ChromaDB 벤치마크
- [X] 메모리 사용량 측정
- [X] 추론 속도 테스트

#### 산출물
- `scripts/benchmark_models.py` ✅
- `scripts/benchmark_vector_stores.py` ✅
- `docs/benchmark_analysis.md` ✅

#### 완료 기준
- [X] 모델 성능 비교 결과 확보
- [X] 벡터 스토어 성능 분석 완료
- [X] 최적 모델 조합 선정

---

## 🗓️ Week 3-4: 법률 데이터 수집 및 전처리

### TASK 2.1: 데이터 수집 파이프라인 구축 (법령/판례 데이터, JSON 저장) ✅
**담당자**: 데이터 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: Critical  
**상태**: 완료 (2025-09-25)

#### 📋 상세 작업 계획

##### Day 1: 국가법령정보센터 OpenAPI 연동 기반 구축
- [X] **API 키 발급 및 인증 설정**
  - [X] 국가법령정보센터 OpenAPI 신청 및 키 발급
  - [X] OC 파라미터 기반 인증 시스템 구현 (사용자 이메일 ID)
  - [X] 요청 제한 및 쿼터 관리 로직 구현 (일일 1000회 제한)

- [X] **핵심 API 연동 구현**
  - [X] **법령 관련 API** (6개):
    - [X] 현행법령(시행일) 목록 조회 API 연동 (`target=lsEfYd`)
    - [X] 현행법령(시행일) 본문 조회 API 연동 (`target=eflaw`)
    - [X] 현행법령(공포일) 목록 조회 API 연동 (`target=lsPrYd`)
    - [X] 현행법령(공포일) 본문 조회 API 연동 (`target=prlaw`)
    - [X] 법령 연혁 목록 조회 API 연동 (`target=lsHst`)
    - [X] 법령 연혁 본문 조회 API 연동 (`target=hstlaw`)
  - [X] **판례 관련 API** (2개):
    - [X] 판례 목록 조회 API 연동 (`target=prec`)
    - [X] 판례 본문 조회 API 연동 (`target=prec`)
  - [X] **헌재결정례 관련 API** (2개):
    - [X] 헌재결정례 목록 조회 API 연동 (`target=ccon`)
    - [X] 헌재결정례 본문 조회 API 연동 (`target=ccon`)
  - [X] **법령해석례 관련 API** (2개):
    - [X] 법령해석례 목록 조회 API 연동 (`target=lex`)
    - [X] 법령해석례 본문 조회 API 연동 (`target=lex`)

##### Day 2: 핵심 데이터 수집 스크립트 개발 (법령/판례)
- [X] **법령 수집 스크립트 (`scripts/collect_laws.py`)**
  - [X] 주요 법령 20개 목록 정의 및 수집 로직
  - [X] 현행법령 조문별 수집 (시행일 기준, `eflaw` API 사용)
  - [X] 법령 개정 이력 추적 및 수집 (`hstlaw` API 사용)
  - [X] 법령 체계도 수집 (부가서비스, `target=lsSys`)
  - [X] 신구법 비교 데이터 수집 (`target=lsNewOld`)
  - [X] 영문법령 데이터 수집 (`target=lsEng`)

- [X] **판례 수집 스크립트 (`scripts/collect_precedents.py`)**
  - [X] 판례 목록 조회 및 메타데이터 추출 (`target=prec`)
  - [X] 판례 본문 수집 및 구조화 (`target=prec`)
  - [X] 사건번호, 법원, 판결일 등 메타데이터 정제
  - [X] 판례 카테고리 분류 (민사, 형사, 행정, 상사 등)
  - [X] 검색 옵션 활용 (판례명 검색, 본문 검색, 날짜 범위 검색)

##### Day 3: 에러 핸들링 및 품질 관리
- [X] **에러 핸들링 및 재시도 로직**
  - [X] API 호출 실패 시 재시도 메커니즘 (최대 3회)
  - [X] 네트워크 타임아웃 처리 (30초)
  - [X] API 쿼터 초과 시 백오프 전략 (일일 1000회 제한)
  - [X] 데이터 검증 및 품질 체크 ✅
  - [X] 로깅 및 모니터링 시스템

- [ ] **향후 확장을 위한 API 연동 준비**
  - [X] **행정규칙 수집** (2개) - 향후 확장용:
    - [X] 행정규칙 목록 조회 API 연동 (`target=lsAdRul`)
    - [X] 행정규칙 본문 조회 API 연동 (`target=adrul`)
    - [X] 행정규칙 신구법 비교 데이터 수집 (`target=lsAdRulNewOld`)
  - [X] **자치법규 수집** (2개) - 향후 확장용:
    - [X] 자치법규 목록 조회 API 연동 (`target=lsOrdinance`)
    - [X] 자치법규 본문 조회 API 연동 (`target=ordinance`)
    - [X] 법령-자치법규 연계 데이터 수집 (`target=lsOrdinanceLink`)
  - [X] **위원회결정문 수집** (22개) - 향후 확장용:
    - [X] 국정감사위원회 결정문 수집 (`target=lsAudit`)
    - [X] 예산결산특별위원회 결정문 수집 (`target=lsBudget`)
    - [X] 법제사법위원회 결정문 수집 (`target=lsLegis`)
    - [X] 기획재정위원회 결정문 수집 (`target=lsPlan`)
    - [X] 과학기술정보통신위원회 결정문 수집 (`target=lsSciTech`)
    - [X] 행정안전위원회 결정문 수집 (`target=lsAdmin`)
    - [X] 문화체육관광위원회 결정문 수집 (`target=lsCulture`)
    - [X] 농림축산식품해양수산위원회 결정문 수집 (`target=lsAgri`)
    - [X] 산업통상자원중소벤처기업위원회 결정문 수집 (`target=lsIndustry`)
    - [X] 보건복지위원회 결정문 수집 (`target=lsWelfare`)
    - [X] 환경노동위원회 결정문 수집 (`target=lsEnvLabor`)
  - [X] **기타 데이터 수집** - 향후 확장용:
    - [X] 행정심판례 수집 (`target=lsAppeal`)
    - [X] 조약 수집 (`target=lsTreaty`)

##### Day 3: 데이터 수집 파이프라인 통합 및 최적화 (법령/판례 데이터)
- [X] **데이터 수집 파이프라인 통합**
  - [X] 배치 처리 시스템 구현 (API별 순차 처리)
  - [X] 병렬 처리 최적화 (동일 API 내에서만)
  - [X] 메모리 사용량 최적화 (스트리밍 처리)
  - [X] 진행률 모니터링 및 상태 추적

- [X] **JSON 파일 저장 시스템**
  - [X] 원본 데이터 JSON 파일 저장 (법령/판례 데이터)
  - [X] 데이터 타입별 디렉토리 구조 생성
  - [X] 메타데이터 및 수집 통계 저장
  - [X] 데이터 검증 및 품질 체크 ✅

#### 🎯 데이터 수집 목표 (OpenAPI 기반) - TASK 2 범위

##### 법령 데이터 (현행법령 기준) - TASK 2에서 수집
- **기본법**: 민법, 상법, 형법, 민사소송법, 형사소송법 (5개)
- **특별법**: 노동법, 부동산법, 금융법, 지적재산권법, 개인정보보호법 (5개)
- **행정법**: 행정소송법, 국세기본법, 건축법, 행정절차법, 정보공개법 (5개)
- **사회법**: 사회보장법, 의료법, 교육법, 환경법, 소비자보호법 (5개)
- **총 20개 주요 법령의 모든 조문 및 개정이력**

##### 판례 데이터 (국가법령정보센터 기준) - TASK 2에서 수집
- **판례**: 7,699건 (최근 5년간, target=prec) ✅
- **총 7,699건의 판례 데이터** (목표 5,000건 초과 달성)

##### 날짜 기반 판례 수집 전략 (NEW) - TASK 2.5에서 추가
- **연도별 수집**: 최근 5년, 연간 2,000건, 총 10,000건
- **분기별 수집**: 최근 2년, 분기당 500건, 총 4,000건  
- **월별 수집**: 최근 1년, 월간 200건, 총 2,400건
- **주별 수집**: 최근 3개월, 주간 100건, 총 1,200건
- **총 예상 수집량**: 17,600건 (기존 대비 2.3배 증가)
- **폴더별 저장**: 각 수집 실행마다 별도 폴더로 raw 데이터 구분

##### 추가 데이터 (프로젝트 개발 이후 확장 예정)
- **헌재결정례**: 1,000건 (최근 5년간, target=detc) - 향후 추가
- **법령해석례**: 2,000건 (최근 3년간, target=expc) - 향후 추가
- **행정심판례**: 1,000건 (최근 3년간, target=decc) - 향후 추가
- **행정규칙**: 1,000건 (주요 부처별, target=lsAdRul) - 향후 추가
- **자치법규**: 500건 (주요 지자체별, target=lsOrdinance) - 향후 추가
- **위원회결정문**: 500건 (주요 위원회별, target=lsAudit~lsEnvLabor) - 향후 추가
- **조약**: 100건 (주요 조약, target=lsTreaty) - 향후 추가
- **법령 체계도**: 주요 법령별 체계도 (target=lsSys) - 향후 추가
- **신구법 비교**: 개정된 법령의 신구법 비교 (target=lsNewOld) - 향후 추가
- **영문법령**: 주요 법령의 영문 버전 (target=lsEng) - 향후 추가
- **관련법령 연계**: 법령 간 연관관계 데이터 (target=lsOrdinanceLink) - 향후 추가

#### 🛠️ 기술 구현 세부사항

##### 국가법령정보센터 OpenAPI 연동 설정 (120개 API 지원)
```python
# API 설정 클래스
class LawOpenAPIConfig:
    # 국가법령정보센터 OpenAPI 기본 설정
    BASE_URL = "http://www.law.go.kr/DRF"
    OC = os.getenv("LAW_OPEN_API_OC")  # 사용자 이메일 ID
    
    # API 엔드포인트 설정 (120개 API)
    ENDPOINTS = {
        # === 법령 관련 API (18개) ===
        # 본문 관련 (6개)
        "law_list_effective": "/lawSearch.do?target=lsEfYd",  # 현행법령(시행일) 목록
        "law_detail_effective": "/lawService.do?target=eflaw",  # 현행법령(시행일) 본문
        "law_list_promulgated": "/lawSearch.do?target=lsNw",  # 현행법령(공포일) 목록
        "law_detail_promulgated": "/lawService.do?target=prlaw",  # 현행법령(공포일) 본문
        "law_history_list": "/lawSearch.do?target=lsHst",  # 법령 연혁 목록
        "law_history_detail": "/lawService.do?target=hstlaw",  # 법령 연혁 본문
        
        # 조항호목 관련 (2개)
        "law_jo_effective": "/lawService.do?target=eflaw",  # 현행법령(시행일) 조항호목
        "law_jo_promulgated": "/lawService.do?target=prlaw",  # 현행법령(공포일) 조항호목
        
        # 영문법령 관련 (2개)
        "law_eng_list": "/lawSearch.do?target=lsEng",  # 영문 법령 목록
        "law_eng_detail": "/lawService.do?target=englaw",  # 영문 법령 본문
        
        # 이력 관련 (3개)
        "law_change_list": "/lawSearch.do?target=lsChg",  # 법령 변경이력 목록
        "law_day_jo_revise_list": "/lawSearch.do?target=lsDayJoRvs",  # 일자별 조문 개정 이력
        "law_jo_change_list": "/lawSearch.do?target=lsJoChg",  # 조문별 변경 이력
        
        # 연계 관련 (3개)
        "law_ordinance_link_list": "/lawSearch.do?target=lsOrdinCon",  # 법령-자치법규 연계 목록
        "law_ordinance_link_status": "/lawService.do?target=lsOrdinCon",  # 법령-자치법규 연계현황
        "law_delegated": "/lawService.do?target=lsDelegated",  # 위임법령 조회
        
        # 부가서비스 관련 (10개)
        "law_system_list": "/lawSearch.do?target=lsStmd",  # 법령 체계도 목록
        "law_system_detail": "/lawService.do?target=lsStmd",  # 법령 체계도 본문
        "law_old_new_list": "/lawSearch.do?target=lsNewOld",  # 신구법 목록
        "law_old_new_detail": "/lawService.do?target=lsNewOld",  # 신구법 본문
        "law_three_compare_list": "/lawSearch.do?target=lsThdCmp",  # 3단 비교 목록
        "law_three_compare_detail": "/lawService.do?target=lsThdCmp",  # 3단 비교 본문
        "law_abbreviation_list": "/lawSearch.do?target=lsAbrv",  # 법률명 약칭 조회
        "law_deleted_data_list": "/lawSearch.do?target=datDelHst",  # 삭제 데이터 목록
        "law_oneview_list": "/lawSearch.do?target=oneView",  # 한눈보기 목록
        "law_oneview_detail": "/lawService.do?target=oneView",  # 한눈보기 본문
        
        # === 행정규칙 관련 API (4개) ===
        "admin_rule_list": "/lawSearch.do?target=lsAdRul",  # 행정규칙 목록
        "admin_rule_detail": "/lawService.do?target=adRul",  # 행정규칙 본문
        "admin_rule_old_new_list": "/lawSearch.do?target=adRulNewOld",  # 행정규칙 신구법 비교 목록
        "admin_rule_old_new_detail": "/lawService.do?target=adRulNewOld",  # 행정규칙 신구법 비교 본문
        
        # === 자치법규 관련 API (3개) ===
        "local_ordinance_list": "/lawSearch.do?target=lsOrdinance",  # 자치법규 목록
        "local_ordinance_detail": "/lawService.do?target=ordinance",  # 자치법규 본문
        "local_ordinance_law_link_list": "/lawSearch.do?target=ordinLsCon",  # 자치법규-법령 연계 목록
        
        # === 판례 관련 API (2개) ===
        "precedent_list": "/lawSearch.do?target=prec",  # 판례 목록
        "precedent_detail": "/lawService.do?target=prec",  # 판례 본문
        
        # === 헌재결정례 관련 API (2개) ===
        "constitutional_list": "/lawSearch.do?target=detc",  # 헌재결정례 목록
        "constitutional_detail": "/lawService.do?target=detc",  # 헌재결정례 본문
        
        # === 법령해석례 관련 API (2개) ===
        "interpretation_list": "/lawSearch.do?target=expc",  # 법령해석례 목록
        "interpretation_detail": "/lawService.do?target=expc",  # 법령해석례 본문
        
        # === 행정심판례 관련 API (2개) ===
        "appeal_list": "/lawSearch.do?target=decc",  # 행정심판례 목록
        "appeal_detail": "/lawService.do?target=decc",  # 행정심판례 본문
        
        # === 위원회결정문 관련 API (24개) ===
        # 개인정보보호위원회 (2개)
        "ppc_list": "/lawSearch.do?target=lsPpc",  # 개인정보보호위원회 목록
        "ppc_detail": "/lawService.do?target=ppc",  # 개인정보보호위원회 본문
        
        # 고용보험심사위원회 (2개)
        "eiac_list": "/lawSearch.do?target=lsEiac",  # 고용보험심사위원회 목록
        "eiac_detail": "/lawService.do?target=eiac",  # 고용보험심사위원회 본문
        
        # 공정거래위원회 (2개)
        "ftc_list": "/lawSearch.do?target=lsFtc",  # 공정거래위원회 목록
        "ftc_detail": "/lawService.do?target=ftc",  # 공정거래위원회 본문
        
        # 국민권익위원회 (2개)
        "acr_list": "/lawSearch.do?target=lsAcr",  # 국민권익위원회 목록
        "acr_detail": "/lawService.do?target=acr",  # 국민권익위원회 본문
        
        # 금융위원회 (2개)
        "fsc_list": "/lawSearch.do?target=lsFsc",  # 금융위원회 목록
        "fsc_detail": "/lawService.do?target=fsc",  # 금융위원회 본문
        
        # 노동위원회 (2개)
        "nlrc_list": "/lawSearch.do?target=lsNlrc",  # 노동위원회 목록
        "nlrc_detail": "/lawService.do?target=nlrc",  # 노동위원회 본문
        
        # 방송통신위원회 (2개)
        "kcc_list": "/lawSearch.do?target=lsKcc",  # 방송통신위원회 목록
        "kcc_detail": "/lawService.do?target=kcc",  # 방송통신위원회 본문
        
        # 산업재해보상보험재심사위원회 (2개)
        "iaciac_list": "/lawSearch.do?target=lsIaciac",  # 산업재해보상보험재심사위원회 목록
        "iaciac_detail": "/lawService.do?target=iaciac",  # 산업재해보상보험재심사위원회 본문
        
        # 중앙토지수용위원회 (2개)
        "oclt_list": "/lawSearch.do?target=lsOclt",  # 중앙토지수용위원회 목록
        "oclt_detail": "/lawService.do?target=oclt",  # 중앙토지수용위원회 본문
        
        # 중앙환경분쟁조정위원회 (2개)
        "ecc_list": "/lawSearch.do?target=lsEcc",  # 중앙환경분쟁조정위원회 목록
        "ecc_detail": "/lawService.do?target=ecc",  # 중앙환경분쟁조정위원회 본문
        
        # 증권선물위원회 (2개)
        "sfc_list": "/lawSearch.do?target=lsSfc",  # 증권선물위원회 목록
        "sfc_detail": "/lawService.do?target=sfc",  # 증권선물위원회 본문
        
        # 국가인권위원회 (2개)
        "nhrck_list": "/lawSearch.do?target=lsNhrck",  # 국가인권위원회 목록
        "nhrck_detail": "/lawService.do?target=nhrck",  # 국가인권위원회 본문
        
        # === 조약 관련 API (2개) ===
        "treaty_list": "/lawSearch.do?target=lsTreaty",  # 조약 목록
        "treaty_detail": "/lawService.do?target=treaty",  # 조약 본문
        
        # === 별표ㆍ서식 관련 API (3개) ===
        "law_form_list": "/lawSearch.do?target=lsByl",  # 법령 별표ㆍ서식 목록
        "admin_rule_form_list": "/lawSearch.do?target=adRulByl",  # 행정규칙 별표ㆍ서식 목록
        "local_ordinance_form_list": "/lawSearch.do?target=ordinByl",  # 자치법규 별표ㆍ서식 목록
        
        # === 학칙ㆍ공단ㆍ공공기관 관련 API (2개) ===
        "school_public_rule_list": "/lawSearch.do?target=lsSchlPubRul",  # 학칙ㆍ공단ㆍ공공기관 목록
        "school_public_rule_detail": "/lawService.do?target=schlPubRul",  # 학칙ㆍ공단ㆍ공공기관 본문
        
        # === 법령용어 관련 API (2개) ===
        "legal_term_list": "/lawSearch.do?target=lsTrm",  # 법령 용어 목록
        "legal_term_detail": "/lawService.do?target=lsTrm",  # 법령 용어 본문
        
        # === 모바일 관련 API (12개) ===
        "mobile_law_list": "/lawSearch.do?target=mobLs",  # 모바일 법령 목록
        "mobile_law_detail": "/lawService.do?target=mobLs",  # 모바일 법령 본문
        "mobile_admin_rule_list": "/lawSearch.do?target=mobAdRul",  # 모바일 행정규칙 목록
        "mobile_admin_rule_detail": "/lawService.do?target=mobAdRul",  # 모바일 행정규칙 본문
        "mobile_local_ordinance_list": "/lawSearch.do?target=mobOrdin",  # 모바일 자치법규 목록
        "mobile_local_ordinance_detail": "/lawService.do?target=mobOrdin",  # 모바일 자치법규 본문
        "mobile_precedent_list": "/lawSearch.do?target=mobPrec",  # 모바일 판례 목록
        "mobile_precedent_detail": "/lawService.do?target=mobPrec",  # 모바일 판례 본문
        "mobile_constitutional_list": "/lawSearch.do?target=mobDetc",  # 모바일 헌재결정례 목록
        "mobile_constitutional_detail": "/lawService.do?target=mobDetc",  # 모바일 헌재결정례 본문
        "mobile_interpretation_list": "/lawSearch.do?target=mobExpc",  # 모바일 법령해석례 목록
        "mobile_interpretation_detail": "/lawService.do?target=mobExpc",  # 모바일 법령해석례 본문
        "mobile_appeal_list": "/lawSearch.do?target=mobDecc",  # 모바일 행정심판례 목록
        "mobile_appeal_detail": "/lawService.do?target=mobDecc",  # 모바일 행정심판례 본문
        "mobile_treaty_list": "/lawSearch.do?target=mobTrty",  # 모바일 조약 목록
        "mobile_treaty_detail": "/lawService.do?target=mobTrty",  # 모바일 조약 본문
        "mobile_law_form_list": "/lawSearch.do?target=mobLsByl",  # 모바일 법령 별표ㆍ서식 목록
        "mobile_admin_rule_form_list": "/lawSearch.do?target=mobAdRulByl",  # 모바일 행정규칙 별표ㆍ서식 목록
        "mobile_local_ordinance_form_list": "/lawSearch.do?target=mobOrdinByl",  # 모바일 자치법규 별표ㆍ서식 목록
        "mobile_legal_term_list": "/lawSearch.do?target=mobLsTrm",  # 모바일 법령 용어 목록
        
        # === 맞춤형 관련 API (6개) ===
        "custom_law_list": "/lawSearch.do?target=custLs",  # 맞춤형 법령 목록
        "custom_law_jo_list": "/lawSearch.do?target=custLsJo",  # 맞춤형 법령 조문 목록
        "custom_admin_rule_list": "/lawSearch.do?target=custAdRul",  # 맞춤형 행정규칙 목록
        "custom_admin_rule_jo_list": "/lawSearch.do?target=custAdRulJo",  # 맞춤형 행정규칙 조문 목록
        "custom_local_ordinance_list": "/lawSearch.do?target=custOrdin",  # 맞춤형 자치법규 목록
        "custom_local_ordinance_jo_list": "/lawSearch.do?target=custOrdinJo",  # 맞춤형 자치법규 조문 목록
        
        # === 법령정보지식베이스 관련 API (7개) ===
        "legal_term_ai": "/lawService.do?target=lsTrmAI",  # 법령용어 AI 조회
        "daily_term": "/lawService.do?target=dlyTrm",  # 일상용어 조회
        "legal_daily_term_relation": "/lawService.do?target=lsTrmRlt",  # 법령용어-일상용어 연계
        "daily_legal_term_relation": "/lawService.do?target=dlyTrmRlt",  # 일상용어-법령용어 연계
        "legal_term_jo_relation": "/lawService.do?target=lsTrmRltJo",  # 법령용어-조문 연계
        "jo_legal_term_relation": "/lawService.do?target=joRltLsTrm",  # 조문-법령용어 연계
        "related_law": "/lawService.do?target=lsRlt",  # 관련법령 조회
        
        # === 중앙부처 1차 해석 관련 API (15개) ===
        "moel_interpretation_list": "/lawSearch.do?target=cgmExpcMoel",  # 고용노동부 법령해석 목록
        "moel_interpretation_detail": "/lawService.do?target=cgmExpcMoel",  # 고용노동부 법령해석 본문
        "molit_interpretation_list": "/lawSearch.do?target=cgmExpcMolit",  # 국토교통부 법령해석 목록
        "molit_interpretation_detail": "/lawService.do?target=cgmExpcMolit",  # 국토교통부 법령해석 본문
        "moef_interpretation_list": "/lawSearch.do?target=cgmExpcMoef",  # 기획재정부 법령해석 목록
        "mof_interpretation_list": "/lawSearch.do?target=cgmExpcMof",  # 해양수산부 법령해석 목록
        "mof_interpretation_detail": "/lawService.do?target=cgmExpcMof",  # 해양수산부 법령해석 본문
        "mois_interpretation_list": "/lawSearch.do?target=cgmExpcMois",  # 행정안전부 법령해석 목록
        "mois_interpretation_detail": "/lawService.do?target=cgmExpcMois",  # 행정안전부 법령해석 본문
        "me_interpretation_list": "/lawSearch.do?target=cgmExpcMe",  # 환경부 법령해석 목록
        "me_interpretation_detail": "/lawService.do?target=cgmExpcMe",  # 환경부 법령해석 본문
        "kcs_interpretation_list": "/lawSearch.do?target=cgmExpcKcs",  # 관세청 법령해석 목록
        "kcs_interpretation_detail": "/lawService.do?target=cgmExpcKcs",  # 관세청 법령해석 본문
        "nts_interpretation_list": "/lawSearch.do?target=cgmExpcNts",  # 국세청 법령해석 목록
        
        # === 특별행정심판 관련 API (4개) ===
        "tt_appeal_list": "/lawSearch.do?target=specialDeccTt",  # 조세심판원 특별행정심판례 목록
        "tt_appeal_detail": "/lawService.do?target=specialDeccTt",  # 조세심판원 특별행정심판례 본문
        "kmst_appeal_list": "/lawSearch.do?target=specialDeccKmst",  # 해양안전심판원 특별행정심판례 목록
        "kmst_appeal_detail": "/lawService.do?target=specialDeccKmst",  # 해양안전심판원 특별행정심판례 본문
    }
    
    # 요청 제한 설정
    RATE_LIMIT = 1000  # requests per day (OpenAPI 제한)
    MAX_RETRIES = 3
    TIMEOUT = 30  # seconds
    BATCH_SIZE = 50  # 한 번에 처리할 항목 수
    MAX_DISPLAY = 100  # 한 번에 조회할 최대 결과 수
    
    # API 우선순위 설정
    API_PRIORITY = {
        "high": [  # 핵심 API (20개)
            "law_list_effective", "law_detail_effective", "law_list_promulgated", "law_detail_promulgated",
            "law_history_list", "law_history_detail", "law_eng_list", "law_eng_detail",
            "law_jo_effective", "law_jo_promulgated", "law_change_list", "law_day_jo_revise_list",
            "law_jo_change_list", "law_ordinance_link_list", "law_ordinance_link_status", "law_delegated",
            "law_system_list", "law_system_detail", "law_old_new_list", "law_old_new_detail"
        ],
        "medium": [  # 확장 API (30개)
            "admin_rule_list", "admin_rule_detail", "admin_rule_old_new_list", "admin_rule_old_new_detail",
            "local_ordinance_list", "local_ordinance_detail", "local_ordinance_law_link_list",
            "precedent_list", "precedent_detail", "constitutional_list", "constitutional_detail",
            "interpretation_list", "interpretation_detail", "appeal_list", "appeal_detail",
            "treaty_list", "treaty_detail", "law_form_list", "admin_rule_form_list", "local_ordinance_form_list",
            "school_public_rule_list", "school_public_rule_detail", "legal_term_list", "legal_term_detail",
            "mobile_law_list", "mobile_law_detail", "mobile_admin_rule_list", "mobile_admin_rule_detail",
            "mobile_local_ordinance_list", "mobile_local_ordinance_detail", "mobile_precedent_list", "mobile_precedent_detail"
        ],
        "low": [  # 전문 API (70개)
            # 위원회결정문, 모바일 나머지, 맞춤형, 법령정보지식베이스, 중앙부처 1차 해석, 특별행정심판
        ]
    }
```

##### 데이터 수집 전략 (OpenAPI 기반)
```python
# 수집 전략 클래스
class LawDataCollectionStrategy:
    def __init__(self, api_config: LawOpenAPIConfig):
        self.api_config = api_config
        self.collection_priority = {
            "laws": 1,  # 법령 (최우선)
            "precedents": 2,  # 판례
            "constitutional": 3,  # 헌재결정례
            "interpretations": 4,  # 법령해석례
            "administrative_rules": 5,  # 행정규칙
            "local_ordinances": 6,  # 자치법규
            "committee_decisions": 7,  # 위원회결정문
            "appeals": 8,  # 행정심판례
            "treaties": 9,  # 조약
        }
        
    def collect_laws(self):
        """법령 데이터 수집 (현행법령 기준)"""
        # 1. 현행법령(시행일) 목록 조회 (target=lsEfYd)
        # 2. 각 법령의 상세 조문 수집 (target=eflaw)
        # 3. 법령 연혁 데이터 수집 (target=lsHst, hstlaw)
        # 4. 법령 체계도 수집 (부가서비스, target=lsSys)
        # 5. 신구법 비교 데이터 수집 (target=lsNewOld)
        # 6. 영문법령 데이터 수집 (target=lsEng)
        pass
        
    def collect_precedents(self):
        """판례 데이터 수집"""
        # 1. 판례 목록 조회 (target=prec, 최근 5년간)
        # 2. 각 판례의 상세 내용 수집 (target=prec)
        # 3. 판례 메타데이터 추출 및 분류
        # 4. 검색 옵션 활용 (판례명 검색, 본문 검색, 날짜 범위 검색)
        pass
        
    def collect_committee_decisions(self):
        """위원회결정문 데이터 수집"""
        # 1. 각 위원회별 결정문 목록 조회
        # 2. 각 결정문의 상세 내용 수집
        # 3. 위원회별 분류 및 메타데이터 추출
        pass
        
    def collect_administrative_rules(self):
        """행정규칙 데이터 수집"""
        # 1. 행정규칙 목록 조회 (target=lsAdRul)
        # 2. 행정규칙 본문 수집 (target=adrul)
        # 3. 신구법 비교 데이터 수집 (target=lsAdRulNewOld)
        pass
        
    def collect_local_ordinances(self):
        """자치법규 데이터 수집"""
        # 1. 자치법규 목록 조회 (target=lsOrdinance)
        # 2. 자치법규 본문 수집 (target=ordinance)
        # 3. 법령-자치법규 연계 데이터 수집 (target=lsOrdinanceLink)
        pass
```

#### 📊 품질 관리 기준

##### 데이터 품질 지표
- **완성도**: 95% 이상 (필수 필드 누락 없음)
- **정확도**: 98% 이상 (API 응답과 일치)
- **일관성**: 90% 이상 (데이터 형식 통일)
- **신선도**: 7일 이내 업데이트

##### 검증 체크리스트
- [ ] 사건번호 형식 검증
- [ ] 판결일 유효성 검증
- [ ] 법원명 표준화 검증
- [ ] 텍스트 인코딩 검증
- [ ] 메타데이터 완성도 검증

#### 🚨 위험 요소 및 대응 방안

##### 기술적 위험
- **API 제한**: 요청 제한 초과 시 백오프 전략
- **네트워크 장애**: 재시도 로직 및 대체 데이터소스
- **메모리 부족**: 배치 처리 및 스트리밍 방식
- **데이터 손실**: 실시간 백업 및 체크포인트

##### 법적 위험
- **저작권 문제**: 공개 API만 사용, 출처 명시
- **개인정보**: 개인정보 포함 데이터 제외
- **데이터 사용권**: API 이용약관 준수

#### 📈 성능 최적화 전략

##### 수집 성능 최적화
- **병렬 처리**: 멀티프로세싱으로 동시 수집
- **배치 처리**: 대량 데이터 효율적 처리
- **캐싱**: 중복 요청 방지
- **압축**: 데이터 저장 공간 최적화

##### 메모리 최적화
- **스트리밍**: 대용량 파일 스트리밍 처리
- **청킹**: 데이터를 작은 단위로 분할
- **가비지 컬렉션**: 불필요한 객체 즉시 해제
- **메모리 모니터링**: 실시간 메모리 사용량 추적

#### 산출물 (TASK 2 범위)
- `source/data/law_open_api_client.py` - 국가법령정보센터 OpenAPI 클라이언트
- `scripts/collect_data_only.py` - 데이터 수집 전용 스크립트 (JSON 저장)
- `scripts/run_data_pipeline.py` - 통합 데이터 파이프라인 실행 스크립트
- `scripts/collect_laws.py` - 법령 수집 스크립트 (기존)
- `scripts/collect_precedents.py` - 판례 수집 스크립트 (기존)
- `data/raw/laws/` - 원본 법령 데이터 (JSON 파일)
- `data/raw/precedents/` - 원본 판례 데이터 (JSON 파일)
- `docs/data_collection_report.md` - 수집 결과 보고서

#### 향후 확장용 산출물 (프로젝트 개발 이후)
- `scripts/collect_constitutional_decisions.py` - 헌재결정례 수집 스크립트
- `scripts/collect_legal_interpretations.py` - 법령해석례 수집 스크립트
- `scripts/collect_administrative_rules.py` - 행정규칙 수집 스크립트
- `scripts/collect_local_ordinances.py` - 자치법규 수집 스크립트
- `scripts/collect_committee_decisions.py` - 위원회결정문 수집 스크립트
- `scripts/collect_administrative_appeals.py` - 행정심판례 수집 스크립트
- `scripts/collect_treaties.py` - 조약 수집 스크립트
- `data/raw/constitutional_decisions/` - 원본 헌재결정례 데이터 (JSON 파일)
- `data/raw/legal_interpretations/` - 원본 법령해석례 데이터 (JSON 파일)
- `data/raw/administrative_rules/` - 원본 행정규칙 데이터 (JSON 파일)
- `data/raw/local_ordinances/` - 원본 자치법규 데이터 (JSON 파일)
- `data/raw/committee_decisions/` - 원본 위원회결정문 데이터 (JSON 파일)
- `data/raw/administrative_appeals/` - 원본 행정심판례 데이터 (JSON 파일)
- `data/raw/treaties/` - 원본 조약 데이터 (JSON 파일)

#### 완료 기준 (TASK 2 범위)
- [X] 법령 20개 수집 완료 (JSON 파일로 저장) ✅
- [X] 판례 7,699건 수집 완료 (JSON 파일로 저장) ✅ (목표 5,000건 초과 달성)
- [X] 데이터 수집 파이프라인 자동화 완료 ✅
- [X] JSON 파일 저장 시스템 구축 완료 ✅
- [X] OpenAPI 쿼터 관리 시스템 구축 완료 (일일 1000회 제한) ✅
- [X] 에러율 5% 이하 달성 ✅ (수집 성공률 95% 이상)

#### 향후 확장 계획 (프로젝트 개발 이후)
- [ ] 헌재결정례 1,000건 수집 (JSON 파일로 저장)
- [ ] 법령해석례 2,000건 수집 (JSON 파일로 저장)
- [ ] 행정규칙 1,000건 수집 (JSON 파일로 저장)
- [ ] 자치법규 500건 수집 (JSON 파일로 저장)
- [ ] 위원회결정문 500건 수집 (JSON 파일로 저장)
- [ ] 행정심판례 1,000건 수집 (JSON 파일로 저장)
- [ ] 조약 100건 수집 (JSON 파일로 저장)

---

### TASK 2.5: 날짜 기반 판례 수집 전략 구현 (NEW) ✅ **완료**
**담당자**: 데이터 수집 엔지니어  
**예상 소요시간**: 3일  
**실제 소요시간**: 3일  
**우선순위**: High  
**상태**: 완료 (2025-01-25)

#### 📋 상세 작업 계획

##### Day 1: 날짜 기반 수집 클래스 구현 ✅ **완료**
- [X] **DateBasedPrecedentCollector 클래스 구현**
  - [X] 연도별, 분기별, 월별, 주별 수집 전략 구현
  - [X] 폴더별 raw 데이터 저장 구조 설계
  - [X] 선고일자 내림차순 최적화 (`sort: "ddes"`)
  - [X] 중복 방지 및 체크포인트 지원
  - [X] 배치 저장 시스템 (100건 단위)

- [X] **폴더 구조 설계**
  - [X] `yearly_{년도}_{타임스탬프}/` - 연도별 수집 폴더
  - [X] `quarterly_{분기}_{타임스탬프}/` - 분기별 수집 폴더
  - [X] `monthly_{년월}_{타임스탬프}/` - 월별 수집 폴더
  - [X] `weekly_{주시작일}주_{타임스탬프}/` - 주별 수집 폴더
  - [X] 수집 요약 파일 자동 생성

##### Day 2: 수집 실행 스크립트 개발 ✅ **완료**
- [X] **collect_by_date.py 스크립트 구현**
  - [X] 명령행 인터페이스 구현 (argparse)
  - [X] 전략별 수집 실행 로직
  - [X] 드라이런 모드 지원
  - [X] 재시작 모드 지원
  - [X] 상세한 진행 상황 로깅

- [X] **사용법 및 옵션 구현**
  - [X] `--strategy`: yearly, quarterly, monthly, weekly, all
  - [X] `--target`: 목표 수집 건수
  - [X] `--count`: 수집할 기간 수
  - [X] `--output`: 출력 디렉토리 지정
  - [X] `--dry-run`: 계획만 출력
  - [X] `--resume`: 중단된 지점부터 재시작

##### Day 3: 문서화 및 테스트 ✅ **완료**
- [X] **개발 문서 작성**
  - [X] `docs/development/date_based_collection_strategy.md` 작성
  - [X] 폴더 구조 및 데이터 형식 명세
  - [X] 사용법 및 예제 코드
  - [X] 성능 최적화 가이드
  - [X] 모니터링 및 로깅 가이드

- [X] **TASK별 상세 개발 계획 업데이트**
  - [X] TASK 2.5 추가
  - [X] 날짜 기반 수집 전략 반영
  - [X] 예상 수집량 업데이트

#### 🎯 수집 목표 (날짜 기반 전략)

##### 연도별 수집
- **기간**: 최근 5년 (2021-2025)
- **목표**: 연간 2,000건
- **총 목표**: 10,000건
- **폴더 수**: 5개

##### 분기별 수집
- **기간**: 최근 2년 (8분기)
- **목표**: 분기당 500건
- **총 목표**: 4,000건
- **폴더 수**: 8개

##### 월별 수집
- **기간**: 최근 1년 (12개월)
- **목표**: 월간 200건
- **총 목표**: 2,400건
- **폴더 수**: 12개

##### 주별 수집
- **기간**: 최근 3개월 (12주)
- **목표**: 주간 100건
- **총 목표**: 1,200건
- **폴더 수**: 12개

##### 전체 수집 목표
- **총 예상 수집량**: 17,600건 (기존 대비 2.3배 증가)
- **총 폴더 수**: 37개
- **데이터 구분**: 각 수집 실행마다 별도 폴더

#### 📁 산출물
- `scripts/precedent/date_based_collector.py` ✅
- `scripts/precedent/collect_by_date.py` ✅
- `docs/development/date_based_collection_strategy.md` ✅
- `data/raw/precedents/yearly_*/` - 연도별 수집 데이터 ✅
- `data/raw/precedents/quarterly_*/` - 분기별 수집 데이터 ✅
- `data/raw/precedents/monthly_*/` - 월별 수집 데이터 ✅
- `data/raw/precedents/weekly_*/` - 주별 수집 데이터 ✅

#### 완료 기준
- [X] DateBasedPrecedentCollector 클래스 구현 완료 ✅
- [X] collect_by_date.py 스크립트 구현 완료 ✅
- [X] 폴더별 raw 데이터 저장 구조 구축 완료 ✅
- [X] 선고일자 내림차순 최적화 구현 완료 ✅
- [X] 중복 방지 시스템 구현 완료 ✅
- [X] 개발 문서 작성 완료 ✅
- [X] 예상 수집량 17,600건 달성 가능 ✅
- [X] 페이지별 즉시 저장 기능 구현 완료 ✅
- [X] 판례일련번호 기준 파일명 생성 완료 ✅
- [X] 실시간 진행상황 로그 개선 완료 ✅
- [X] 오류 복구 메커니즘 구현 완료 ✅

#### 사용 예시
```bash
# 특정 연도 수집 (2025년만) - NEW
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited

# 특정 연도 수집 (2024년만)
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

# 특정 연도 수집 (2023년만)
python scripts/precedent/collect_by_date.py --strategy yearly --year 2023 --unlimited

# 연도별 수집 (최근 5년, 연간 2000건)
python scripts/precedent/collect_by_date.py --strategy yearly --target 10000

# 분기별 수집 (최근 2년, 분기당 500건)
python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000

# 월별 수집 (최근 1년, 월간 200건)
python scripts/precedent/collect_by_date.py --strategy monthly --target 2400

# 주별 수집 (최근 3개월, 주간 100건)
python scripts/precedent/collect_by_date.py --strategy weekly --target 1200

# 모든 전략 순차 실행 (총 17,600건)
python scripts/precedent/collect_by_date.py --strategy all --target 20000
```

#### 고급 옵션 예시
```bash
# 특정 연도 + 건수 제한
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --target 5000

# 특정 연도 + 출력 디렉토리 지정
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited --output data/custom/2025_precedents

# 특정 연도 + 드라이런 모드
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited --dry-run

# 재시작 모드 (중단된 지점부터 재시작)
python scripts/precedent/collect_by_date.py --strategy yearly --target 5000 --resume
```

---

### TASK 2.2: 데이터 전처리 및 구조화 (수집된 Raw 데이터) ⏳ **진행 중**
**담당자**: 데이터 사이언티스트  
**예상 소요시간**: 5-7시간  
**우선순위**: High  
**상태**: 진행 중 (2025-09-30)

#### 📝 업데이트 내용 (2025-09-30)
**전처리 대상 Raw 데이터 현황**:
- ✅ 법령 데이터: 21개 파일 수집 완료
- ✅ 판례 데이터: 연도별 수집 완료 (2024-2025년)
- ✅ 헌재결정례: 2024-2025년 데이터 수집 완료
- ✅ 법령해석례: 배치별 수집 완료
- ✅ 법률 용어: 세션별 수집 완료

**새로 구현된 전처리 스크립트**:
- ✅ `scripts/preprocess_raw_data.py` - 메인 전처리 파이프라인
- ✅ `scripts/batch_preprocess.py` - 배치 전처리 스크립트
- ✅ `scripts/validate_processed_data.py` - 데이터 검증 스크립트
- ✅ `docs/development/raw_data_preprocessing_plan.md` - 전처리 계획서

**사용법**:
```bash
# 전체 전처리 실행
python scripts/preprocess_raw_data.py

# 특정 데이터 유형만 전처리
python scripts/batch_preprocess.py --data-type laws
python scripts/batch_preprocess.py --data-type precedents

# 드라이런 모드 (계획만 확인)
python scripts/batch_preprocess.py --data-type all --dry-run

# 전처리된 데이터 검증
python scripts/validate_processed_data.py
```

#### 📋 상세 작업 계획

##### Day 1: 법률 용어 정규화 시스템 구축 ✅ **완료**
- [X] **기본 텍스트 정규화 구현**
  - [X] 공백 및 특수문자 정리
  - [X] HTML 태그 제거
  - [X] 인코딩 정규화 (UTF-8)
  - [X] 불필요한 문자 제거

- [X] **법률 용어 표준화 시스템 구현**
  - [X] 국가법령정보센터 API 기반 용어 수집
  - [X] 법률 용어 사전 구축 (`LegalTermDictionary`)
  - [X] 동의어 그룹 매핑 시스템
  - [X] 용어 일관성 검증 로직
  - [X] 다층 정규화 파이프라인 (`LegalTermNormalizer`)

##### Day 2: 텍스트 청킹 전략 구현
- [X] **다층 청킹 시스템 구현**
  - [X] 조문 중심 청킹 (법령 데이터)
  - [X] 사건별 청킹 (판례 데이터)
  - [X] 의미적 청킹 (의미 단위 분할)
  - [X] 문장 경계 인식 청킹

- [X] **청킹 품질 관리**
  - [X] 청킹 크기 최적화 (200-3000자)
  - [X] 오버랩 전략 구현 (10-25%)
  - [X] 구조적 완성도 검증

##### Day 3: 법률 구조 요소 정규화
- [X] **법률 구조 패턴 인식**
  - [X] 법률명 정규화 (민법, 상법, 형법 등)
  - [X] 조문 번호 표준화 (제X조, 제X항, 제X호)
  - [X] 사건번호 형식 통일
  - [X] 법원명 표준화

- [X] **엔티티 추출 및 정규화**
  - [X] 법률 엔티티 추출 (법률명, 조문, 사건번호)
  - [X] 날짜 형식 정규화
  - [X] 키워드 추출 및 정리

##### Day 4: 중복 데이터 제거 및 품질 검증
- [X] **중복 데이터 제거 시스템**
  - [X] 해시 기반 중복 검출
  - [X] 유사도 기반 중복 제거
  - [X] 메타데이터 기반 중복 제거

- [X] **품질 검증 시스템 구축**
  - [X] 데이터 완성도 검증
  - [X] 형식 일관성 검증
  - [X] 법률 구조 준수 검증
  - [X] 용어 정확성 검증

##### Day 5: 통합 및 최적화 ✅ **완료**
- [X] **전처리 파이프라인 통합**
  - [X] 단계별 전처리 모듈 통합
  - [X] 에러 처리 및 복구 로직
  - [X] 진행률 모니터링 시스템
  - [X] 기존 `LegalDataProcessor`와 법률 용어 정규화 통합

- [X] **성능 최적화**
  - [X] 메모리 사용량 최적화
  - [X] 처리 속도 개선
  - [X] 배치 처리 최적화
  - [X] 용어 수집 스크립트 (`collect_legal_terms.py`) 개발

#### 🛠️ 기술 구현 세부사항

##### 법률 용어 정규화 시스템 ✅ **구현 완료**

**구현된 모듈:**
- `source/data/legal_term_collection_api.py` ✅ - 국가법령정보센터 OpenAPI 연동
- `source/data/legal_term_dictionary.py` ✅ - 법률 용어 사전 관리
- `source/data/legal_term_normalizer.py` ✅ - 다층 정규화 파이프라인
- `scripts/legal_term/term_collector.py` ✅ - 메모리 최적화 수집기
- `scripts/legal_term/collect_legal_terms.py` ✅ - 용어 수집 스크립트
- `data/raw/legal_terms/session_*/` ✅ - 세션별 데이터 폴더

**🆕 새로 추가된 기능:**
- **API 기반 자동 수집**: 국가법령정보센터 OpenAPI를 통한 실시간 용어 수집
- **메모리 최적화**: 대용량 데이터 수집 시 메모리 부족 방지
- **체크포인트 시스템**: 수집 중단 시 재개 가능
- **세션별 폴더**: 수집 요청마다 독립적인 데이터 폴더 생성
- **Graceful Shutdown**: 안전한 종료 및 리소스 정리

**핵심 기능:**
```python
class LegalTermNormalizer:
    """법률 용어 정규화 핵심 클래스"""
    
    def __init__(self, dictionary_path: str = "data/legal_terms/legal_term_dictionary.json"):
        self.dictionary = LegalTermDictionary(dictionary_path)
        self.normalization_rules = self._load_normalization_rules()
    
    def normalize_text(self, text: str, context: str = None) -> Dict[str, Any]:
        """다층 정규화 파이프라인 실행"""
        result = {
            "original_text": text,
            "normalized_text": text,
            "normalization_steps": [],
            "term_mappings": {},
            "confidence_scores": {},
            "success": False
        }
        
        # Level 1: 기본 정규화
        result = self._basic_normalization(result)
        
        # Level 2: 법률 용어 표준화
        result = self._legal_term_standardization(result, context)
        
        # Level 3: 의미적 정규화
        result = self._semantic_normalization(result)
        
        # Level 4: 구조적 정규화
        result = self._structural_normalization(result)
        
        # 품질 검증
        result = self._quality_validation(result)
        
        return result
```

**API 연동:**
```python
class LegalTermCollectionAPI:
    """국가법령정보센터 API 연동"""
    
    def collect_legal_terms(self, category: str = None, max_terms: int = 1000) -> List[Dict[str, Any]]:
        """법령 용어 목록 수집"""
        
    def collect_term_definitions(self, term_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """용어별 상세 정의 수집"""
```

**용어 사전 관리:**
```python
class LegalTermDictionary:
    """법률 용어 사전 관리"""
    
    def add_term(self, term_data: Dict[str, Any]) -> bool:
        """용어 추가"""
        
    def create_synonym_group(self, group_id: str, standard_term: str, variants: List[str]) -> bool:
        """동의어 그룹 생성"""
        
    def normalize_term(self, term_name: str) -> Tuple[str, float]:
        """용어 정규화"""
```

##### 텍스트 청킹 전략
```python
class LegalTextChunker:
    """법률 텍스트 청킹 클래스"""
    
    def __init__(self):
        self.chunking_strategies = {
            "law_article": self._law_article_chunking,
            "precedent_case": self._precedent_case_chunking,
            "semantic": self._semantic_chunking
        }
    
    def chunk_text(self, text: str, strategy: str, **kwargs) -> List[Dict]:
        """텍스트 청킹 실행"""
        if strategy in self.chunking_strategies:
            return self.chunking_strategies[strategy](text, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
```

#### 📊 품질 관리 기준

##### 데이터 품질 지표
- **완성도**: 95% 이상 (필수 필드 누락 없음)
- **정확도**: 98% 이상 (원본 데이터와 일치)
- **일관성**: 90% 이상 (데이터 형식 통일)
- **용어 정규화 정확도**: 90% 이상

##### 검증 체크리스트
- [X] 법률명 표준화 검증
- [X] 조문 번호 형식 검증
- [X] 사건번호 유효성 검증
- [X] 날짜 형식 검증
- [X] 텍스트 인코딩 검증
- [X] 메타데이터 완성도 검증

#### 🚨 위험 요소 및 대응 방안

##### 기술적 위험
- **메모리 부족**: 배치 처리 및 스트리밍 방식
- **처리 속도 저하**: 병렬 처리 및 캐싱 전략
- **용어 정규화 오류**: 다중 검증 및 전문가 검토
- **데이터 손실**: 실시간 백업 및 체크포인트

##### 품질 위험
- **용어 정규화 부정확**: API 기반 용어 사전 활용
- **청킹 품질 저하**: 다층 검증 및 품질 점수 시스템
- **중복 데이터 누락**: 다중 알고리즘 기반 중복 검출

#### 📈 성능 최적화 전략

##### 처리 성능 최적화
- **병렬 처리**: 멀티프로세싱으로 동시 처리
- **배치 처리**: 대량 데이터 효율적 처리
- **캐싱**: 중복 연산 방지
- **압축**: 데이터 저장 공간 최적화

##### 메모리 최적화
- **스트리밍**: 대용량 파일 스트리밍 처리
- **청킹**: 데이터를 작은 단위로 분할
- **가비지 컬렉션**: 불필요한 객체 즉시 해제
- **메모리 모니터링**: 실시간 메모리 사용량 추적

#### 산출물 (TASK 2 범위)
- `source/data/data_processor.py` ✅ - 데이터 처리 핵심 모듈
- `source/data/legal_term_normalizer.py` ✅ - 법률 용어 정규화 모듈
- `source/data/text_chunker.py` ✅ - 텍스트 청킹 모듈
- `source/data/quality_validator.py` ✅ - 품질 검증 모듈
- `data/processed/` ✅ - 전처리된 데이터 (법령/판례 데이터)
- `data/legal_terms/` ✅ - 법률 용어 사전 데이터
- `scripts/validate_data_quality.py` ✅ - 데이터 품질 검증 스크립트
- `docs/development/legal_term_normalization_strategy.md` ✅ - 법률 용어 정규화 전략 문서

#### 완료 기준 (TASK 2 범위)
- [X] 데이터 전처리 파이프라인 완성
- [X] 법률 용어 정규화 시스템 구축
- [X] 텍스트 청킹 전략 구현
- [X] 품질 검증 시스템 구축
- [X] 중복 데이터 제거 시스템 구축
- [ ] 데이터셋 용량 3GB 이하 압축 (법령/판례 데이터)
- [ ] 용어 정규화 정확도 90% 이상 달성
- [ ] 청킹 품질 점수 85% 이상 달성

#### 향후 확장 계획 (프로젝트 개발 이후)
- [ ] 추가 데이터 유형별 전처리 파이프라인 구축
- [ ] 대용량 데이터 처리를 위한 성능 최적화
- [ ] 실시간 데이터 전처리 기능 추가
- [ ] AI 기반 용어 정규화 시스템 구축
- [ ] 다국어 법률 용어 정규화 지원

---

### TASK 2.3: 벡터DB 구축 파이프라인 (JSON → 벡터DB) - 법령/판례 데이터 ✅ **완료**
**담당자**: ML 엔지니어  
**예상 소요시간**: 3일  
**실제 소요시간**: 1일  
**우선순위**: Critical  
**상태**: 완료 (2025-09-30)

#### 세부 작업
- [X] JSON 파일 읽기 및 파싱 시스템 구현 (법령/판례 데이터) ✅
- [X] 데이터 정규화 및 전처리 로직 구현 ✅
- [X] 텍스트 청킹 전략 구현 ✅
- [X] Sentence-BERT 모델 로딩 및 임베딩 생성 ✅
- [X] FAISS 인덱스 구축 및 최적화 ✅
- [X] SQLite 데이터베이스 연동 (하이브리드 검색용) ✅
- [X] 벡터DB 구축 파이프라인 자동화 ✅

#### 📊 구축 결과
- **총 문서 수**: 642개
- **법령 문서**: 21개
- **판례 문서**: 621개
- **벡터 임베딩**: 642개 (768차원)
- **FAISS 인덱스 크기**: 642개 벡터
- **SQLite 레코드**: 642개

#### 🚀 성능 검증 결과
- **평균 검색 시간**: 0.0003초 (초고속!)
- **초당 쿼리 수**: 3,409개
- **메모리 사용량**: 최대 0.87GB (효율적)
- **검색 정확도**: 71.43% (법령 100%, 판례 33%)

#### 산출물 (TASK 2 범위)
- `scripts/enhanced_build_vector_db.py` ✅ - 향상된 벡터DB 구축 스크립트
- `scripts/test_vector_search.py` ✅ - 벡터 검색 성능 테스트 스크립트
- `source/data/enhanced_data_processor.py` ✅ - 향상된 데이터 처리 모듈
- `source/data/vector_store.py` ✅ - 벡터 스토어 관리
- `source/data/database.py` ✅ - SQLite 데이터베이스 관리
- `data/embeddings/faiss_index.bin` ✅ - FAISS 인덱스 파일
- `data/embeddings/embeddings.npy` ✅ - 벡터 임베딩 파일
- `data/embeddings/metadata.json` ✅ - 문서 메타데이터
- `data/embeddings/vector_db_build_report.json` ✅ - 구축 보고서
- `data/embeddings/vector_search_test_results.json` ✅ - 성능 테스트 결과

#### 완료 기준 (TASK 2 범위)
- [X] 법령/판례 JSON 파일에서 벡터 임베딩 생성 완료 ✅
- [X] FAISS 인덱스 구축 완료 ✅
- [X] SQLite 데이터베이스 구축 완료 (하이브리드 검색용) ✅
- [X] 벡터 검색 성능 검증 완료 ✅
- [X] 하이브리드 검색 시스템 연동 완료 ✅

#### 🛠️ 기술 구현 세부사항

##### 1. 향상된 벡터DB 구축 파이프라인
```python
# 메모리 최적화된 배치 처리
def build_vector_database(self, data_types: List[str] = None):
    # Sentence-BERT 모델 로드
    self.load_sentence_transformer()
    
    # FAISS 인덱스 생성
    self.create_faiss_index(self.dimension)
    
    # 배치 단위로 임베딩 생성 (메모리 절약)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_embeddings = self.generate_embeddings(batch_texts)
        self.faiss_index.add(batch_embeddings.astype('float32'))
```

##### 2. 벡터 검색 성능 테스트
```python
# 초고속 벡터 검색
def test_search_performance(self, query: str, k: int = 10):
    query_embedding = self.model.encode([query])
    distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
    # 평균 검색 시간: 0.0003초
```

##### 3. 검색 품질 예시
- **쿼리**: "민사소송 절차" → **결과**: 민사소송법 (유사도: 0.015) ✅
- **쿼리**: "부동산 등기" → **결과**: 부동산등기법 (유사도: 0.017) ✅
- **쿼리**: "형사처벌" → **결과**: 형사소송법, 형법 (유사도: 0.009) ✅

#### 사용법
```bash
# 벡터DB 구축
python scripts/enhanced_build_vector_db.py --mode build

# 벡터 검색 성능 테스트
python scripts/test_vector_search.py

# 특정 데이터 타입만 구축
python scripts/enhanced_build_vector_db.py --mode laws
python scripts/enhanced_build_vector_db.py --mode precedents
```

#### 향후 확장 계획 (프로젝트 개발 이후)
- [ ] 추가 데이터 유형별 벡터 임베딩 생성
- [ ] 확장된 FAISS 인덱스 구축
- [ ] 추가 데이터를 위한 SQLite 스키마 확장

---

### TASK 2.4: Q&A 데이터셋 생성 (법령/판례 기반) ✅ **완료**
**담당자**: 데이터 사이언티스트  
**예상 소요시간**: 2일  
**실제 소요시간**: 1일  
**우선순위**: Medium  
**상태**: 완료 (2025-10-10)

#### 세부 작업
- [X] 자동 Q&A 생성 파이프라인 (법령/판례 데이터 기반) ✅
- [X] 품질 점수 매기기 시스템 구현 ✅
- [X] 데이터셋 최종 검증 및 내보내기 ✅
- [X] LLM 기반 Q&A 생성 시스템 구축 ✅ (신규)
- [X] Ollama Qwen2.5:7b 모델 연동 ✅ (신규)
- [X] 자연스러운 질문-답변 생성 ✅ (신규)
- [ ] 법률 전문가 검토 (향후 진행)

#### 📊 생성 결과

**템플릿 기반 생성 (기존)**
- **총 Q&A 쌍 수**: 2,709개 (목표 3,000개의 90.3%)
- **평균 품질 점수**: 0.935 (93.5%)
- **고품질 비율**: 99.96% (2,708개/2,709개)
- **신뢰도 평균**: 0.89

**LLM 기반 생성 (신규)**
- **총 Q&A 쌍 수**: 36개 (테스트 단계)
- **평균 품질 점수**: 0.683 (68.3%)
- **질문 유형**: 12가지 다양한 유형
- **자연스러움**: 템플릿 방식 대비 400% 향상
- **실용성**: 법률 실무 중심 질문 생성

#### 🎯 데이터 소스별 분석
- **법령 데이터**: 42개 법령에서 1,284개 Q&A 생성
- **판례 데이터**: 621개 판례에서 1,425개 Q&A 생성
- **헌재결정례**: 185개 결정례 (데이터 구조 개선 필요)
- **법령해석례**: 24개 해석례 (데이터 구조 개선 필요)

#### 산출물 (TASK 2 범위)

**템플릿 기반 파일**
- `data/qa_dataset/large_scale_qa_dataset.json` ✅ - 전체 데이터셋
- `data/qa_dataset/large_scale_qa_dataset_high_quality.json` ✅ - 고품질 데이터셋
- `scripts/generate_qa_dataset.py` ✅ - 기본 생성 스크립트
- `scripts/enhanced_generate_qa_dataset.py` ✅ - 향상된 생성 스크립트
- `scripts/large_scale_generate_qa_dataset.py` ✅ - 대규모 생성 스크립트
- `docs/qa_dataset_quality_report.md` ✅ - 품질 보고서

**LLM 기반 파일 (신규)**
- `source/utils/ollama_client.py` ✅ - Ollama API 클라이언트
- `source/utils/qa_quality_validator.py` ✅ - Q&A 품질 검증 모듈
- `scripts/llm_qa_generator.py` ✅ - LLM 기반 Q&A 생성기
- `scripts/generate_qa_with_llm.py` ✅ - LLM Q&A 생성 실행 스크립트
- `data/qa_dataset/llm_generated/` ✅ - LLM 생성 데이터셋
- `docs/llm_qa_dataset_quality_report.md` ✅ - LLM 품질 보고서

#### 완료 기준 (TASK 2 범위)

**템플릿 기반 기준**
- [X] Q&A 데이터셋 2,709쌍 생성 (법령/판례 기반) ✅ (목표 대비 90.3%)
- [X] 품질 점수 93.5% 달성 ✅ (목표 90% 초과)
- [X] 자동 품질 검증 시스템 구축 ✅

**LLM 기반 기준 (신규)**
- [X] Ollama Qwen2.5:7b 모델 연동 완료 ✅
- [X] 자연스러운 질문-답변 생성 시스템 구축 ✅
- [X] 12가지 다양한 질문 유형 생성 ✅
- [X] 품질 검증 및 중복 제거 시스템 구축 ✅
- [X] 템플릿 방식 대비 400% 자연스러움 향상 ✅
- [ ] 품질 점수 80% 이상 달성 (현재 68.3%)
- [ ] 3,000개 이상 Q&A 생성 (현재 36개)

#### 향후 확장 계획 (프로젝트 개발 이후)

**템플릿 기반 확장**
- [ ] 추가 데이터 유형 기반 Q&A 생성
- [ ] Q&A 데이터셋 5,000쌍으로 확장
- [ ] 다양한 법률 영역별 Q&A 추가

**LLM 기반 확장 (신규)**
- [ ] JSON 파싱 오류 해결 및 안정성 향상
- [ ] 프롬프트 엔지니어링 개선으로 품질 점수 80% 이상 달성
- [ ] 판례 데이터 처리 개선 및 맞춤형 프롬프트 개발
- [ ] 하이브리드 접근법: 기본 정보는 템플릿, 복잡한 설명은 LLM
- [ ] 법률 전문가 검토 시스템 구축
- [ ] 실시간 Q&A 생성 API 개발

---

### TASK 2.5: 통합 데이터 파이프라인 구축 (법령/판례 데이터)
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 1일  
**우선순위**: High

#### 세부 작업
- [X] 데이터 수집과 벡터DB 구축 통합 스크립트 구현 (법령/판례 데이터)
- [X] 파이프라인 실행 모드 설정 (수집만, 구축만, 전체)
- [X] 에러 처리 및 복구 로직 구현
- [X] 진행률 모니터링 및 로깅 시스템 구축
- [ ] 파이프라인 성능 최적화

#### 산출물 (TASK 2 범위)
- `scripts/run_data_pipeline.py` - 통합 데이터 파이프라인 실행 스크립트
- `docs/data_pipeline_guide.md` - 데이터 파이프라인 사용 가이드
- `logs/data_pipeline.log` - 파이프라인 실행 로그

#### 완료 기준 (TASK 2 범위)
- [X] 통합 데이터 파이프라인 구현 완료
- [X] 법령/판례 데이터 수집과 벡터DB 구축 독립 실행 가능
- [X] 파이프라인 모니터링 시스템 구축 완료
- [ ] 에러 복구 및 재시작 기능 구현 완료

#### 향후 확장 계획 (프로젝트 개발 이후)
- [ ] 추가 데이터 유형을 위한 파이프라인 확장
- [ ] 대용량 데이터 처리를 위한 성능 최적화
- [ ] 실시간 데이터 업데이트 기능 추가

---

## 🗓️ Week 5-6: 한국어 법률 챗봇 모델 개발

### TASK 3.1: 모델 선택 및 파인튜닝
**담당자**: ML 엔지니어  
**예상 소요시간**: 4일  
**우선순위**: Critical

#### 세부 작업
- [ ] KoBART 모델 로딩 및 설정
- [ ] LoRA 기반 파인튜닝 구현
- [ ] 법률 특화 프롬프트 템플릿 작성
- [ ] 모델 성능 평가

#### 산출물
- `source/models/kobart_model.py`
- `source/models/model_manager.py`
- `data/training/`
- `models/finetuned/`

#### 완료 기준
- [ ] 파인튜닝된 모델 완성
- [ ] 법률 질의응답 정확도 75% 이상
- [ ] 모델 크기 2GB 이하 최적화

---

### TASK 3.2: 하이브리드 검색 시스템 구현
**담당자**: ML 엔지니어  
**예상 소요시간**: 4일  
**우선순위**: Critical

#### 세부 작업
- [ ] 정확한 매칭 검색 엔진 구현 (SQLite 기반)
- [ ] 의미적 검색 엔진 구현 (FAISS 기반)
- [ ] 하이브리드 검색 엔진 구현
- [ ] 결과 통합 및 랭킹 시스템 구현
- [ ] 검색 API 엔드포인트 구현

#### 산출물
- `source/services/hybrid_search_engine.py`
- `source/services/exact_search_engine.py`
- `source/services/semantic_search_engine.py`
- `source/services/result_merger.py`
- `source/services/result_ranker.py`
- `tests/test_hybrid_search.py`

#### 완료 기준
- [ ] 하이브리드 검색 시스템 구현 완료
- [ ] 정확한 매칭 검색 정확도 95% 이상
- [ ] 의미적 검색 정확도 80% 이상
- [ ] 검색 응답 시간 1초 이내

---

### TASK 3.3: RAG 시스템 구현
**담당자**: ML 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: Critical

#### 세부 작업
- [ ] 하이브리드 검색 기반 RAG 시스템 구현
- [ ] 컨텍스트 생성 로직
- [ ] RAG 기반 답변 생성
- [ ] 성능 최적화

#### 산출물
- `source/services/rag_service.py`
- `tests/test_rag_system.py`

#### 완료 기준
- [ ] RAG 시스템 구현 완료
- [ ] 벡터 검색 정확도 80% 이상
- [ ] 응답 생성 시간 10초 이내

---

### TASK 3.4: 모델 경량화 및 최적화
**담당자**: ML 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] INT8 양자화 적용
- [ ] ONNX 변환
- [ ] 메모리 사용량 최적화
- [ ] 추론 속도 개선

#### 산출물
- `scripts/optimize_model.py`
- `models/optimized/`
- `docs/optimization_report.md`

#### 완료 기준
- [ ] 모델 크기 50% 이상 감소
- [ ] 추론 속도 2배 이상 개선
- [ ] 메모리 사용량 14GB 이하

---

### TASK 3.5: 캐싱 시스템 구현
**담당자**: 백엔드 개발자  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] 메모리 기반 캐시 구현
- [ ] LRU 캐시 정책 적용
- [ ] 캐시 히트율 모니터링
- [ ] 자동 캐시 정리 시스템

#### 산출물
- `source/utils/cache_manager.py`
- `tests/test_cache_system.py`

#### 완료 기준
- [ ] 캐싱 시스템 구현 완료
- [ ] 응답 속도 50% 개선
- [ ] 메모리 사용량 효율적 관리

---

## 🗓️ Week 7-8: 챗봇 인터페이스 개발

### TASK 4.1: Gradio 메인 인터페이스 구현
**담당자**: 프론트엔드 개발자  
**예상 소요시간**: 3일  
**우선순위**: Critical

#### 세부 작업
- [ ] Gradio 기본 인터페이스 구성
- [ ] 탭 기반 UI 설계
- [ ] 반응형 디자인 적용
- [ ] 커스텀 CSS 스타일링

#### 산출물
- `gradio/app.py`
- `gradio/static/custom.css`
- `gradio/components/`

#### 완료 기준
- [ ] 완전한 대화형 인터페이스 구현
- [ ] 4가지 특화 모드 구현
- [ ] 모바일 반응형 디자인 완료

---

### TASK 4.2: 채팅 기능 구현
**담당자**: 프론트엔드 개발자  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] 실시간 채팅 인터페이스
- [ ] 타이핑 인디케이터
- [ ] 메시지 히스토리 관리
- [ ] 키보드 단축키 지원

#### 산출물
- `gradio/components/chat_interface.py`
- `gradio/static/chat.js`

#### 완료 기준
- [ ] 실시간 채팅 기능 완성
- [ ] 사용자 경험 최적화
- [ ] 접근성 기능 구현

---

### TASK 4.3: 파일 업로드 및 분석 기능
**담당자**: 풀스택 개발자  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [ ] 파일 업로드 컴포넌트
- [ ] PDF/DOCX 파싱 기능
- [ ] 계약서 분석 UI
- [ ] 분석 결과 시각화

#### 산출물
- `gradio/components/document_analyzer.py`
- `source/services/analysis_service.py`
- `gradio/static/analysis.js`

#### 완료 기준
- [ ] 파일 업로드 기능 완성
- [ ] 계약서 분석 UI 구현
- [ ] 분석 결과 시각화 완료

---

### TASK 4.4: 실시간 기능 및 최적화
**담당자**: 프론트엔드 개발자  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] 실시간 용어 검색
- [ ] 자동완성 기능
- [ ] 로딩 상태 관리
- [ ] 에러 핸들링 UI

#### 산출물
- `gradio/components/realtime_features.py`
- `gradio/static/realtime.js`

#### 완료 기준
- [ ] 실시간 기능 구현 완료
- [ ] 사용자 인터랙션 최적화
- [ ] 에러 처리 UI 완성

---

## 🗓️ Week 9: 특화 기능 구현

### TASK 5.1: 판례 검색봇 구현
**담당자**: ML 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [ ] 유사 판례 검색 알고리즘
- [ ] 법률 키워드 추출
- [ ] 검색 결과 랭킹
- [ ] 판례 요약 기능

#### 산출물
- `source/services/precedent_search.py`
- `gradio/components/precedent_search.py`
- `tests/test_precedent_search.py`

#### 완료 기준
- [ ] 판례 검색봇 구현 완료
- [ ] 유사도 검색 정확도 80% 이상
- [ ] 검색 결과 품질 검증

---

### TASK 5.2: 계약서 분석봇 구현
**담당자**: ML 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [ ] 위험 조항 탐지 알고리즘
- [ ] 개선 제안 생성
- [ ] 용어 해설 기능
- [ ] 위험도 점수 계산

#### 산출물
- `source/services/contract_analyzer.py`
- `gradio/components/contract_analysis.py`
- `tests/test_contract_analysis.py`

#### 완료 기준
- [ ] 계약서 분석봇 구현 완료
- [ ] 위험 조항 탐지율 90% 이상
- [ ] 개선 제안 품질 검증

---

### TASK 5.3: 법령 해설봇 구현
**담당자**: ML 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] 법조문 해석 모델
- [ ] 쉬운 설명 생성
- [ ] 관련 판례 연동
- [ ] 실무 적용 예시

#### 산출물
- `source/services/law_explainer.py`
- `gradio/components/law_search.py`
- `tests/test_law_explanation.py`

#### 완료 기준
- [ ] 법령 해설봇 구현 완료
- [ ] 이해도 개선 70% 이상
- [ ] 해설 품질 전문가 검증

---

### TASK 5.4: 법률 용어 사전 구현 ✅ **완료**
**담당자**: 백엔드 개발자  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [X] 용어 데이터베이스 구축
- [X] 유사 용어 검색
- [X] 용어 정의 및 예시
- [X] 발음 기호 추가
- [X] **국가법령정보센터 OpenAPI 연동**
- [X] **법령용어 본문 조회 가이드API 구현**
- [X] **메모리 최적화 및 체크포인트 시스템**
- [X] **세션별 폴더 구분 시스템**
- [X] **Graceful Shutdown 기능**

#### 산출물
- `source/data/legal_term_dictionary.py` ✅ - 법률 용어 사전 관리
- `source/data/legal_term_collection_api.py` ✅ - API 연동 및 데이터 수집
- `scripts/legal_term/term_collector.py` ✅ - 메모리 최적화 수집기
- `scripts/legal_term/collect_legal_terms.py` ✅ - 수집 스크립트
- `data/raw/legal_terms/session_*/` ✅ - 세션별 데이터 폴더
- `gradio/components/term_search.py`

#### 완료 기준
- [X] 법률 용어 사전 수집 시스템 구축
- [X] 실시간 검색 기능 완성
- [X] 용어 품질 검증
- [X] **API 기반 자동 수집 시스템**
- [X] **메모리 최적화 및 안정성 확보**
- [X] **세션별 데이터 관리**

#### 🆕 새로 구현된 기능들

##### 1. 국가법령정보센터 OpenAPI 연동
- **API 엔드포인트**: `/lawSearch.do`, `/DRF/lawService.do`
- **지원 기능**: 법령용어 목록 조회, 상세정보 조회
- **파라미터**: 날짜별 필터링, 검색어 기반 수집
- **응답 형식**: JSON, XML 지원

##### 2. 법령용어 본문 조회 가이드API
- **상세정보 수집**: 법령용어정의, 출처, 한자명 등
- **필드 매핑**: 
  - `법령용어일련번호` → `term_sequence_number`
  - `법령용어명_한글` → `term_name_korean`
  - `법령용어명_한자` → `term_name_chinese`
  - `법령용어정의` → `definition`
  - `출처` → `source`

##### 3. 메모리 최적화 및 체크포인트 시스템
- **배치 처리**: 메모리 사용량에 따른 동적 배치 크기 조정
- **체크포인트**: 수집 중단 시 재개 가능
- **가비지 컬렉션**: 메모리 정리 자동화
- **메모리 모니터링**: 실시간 메모리 사용량 추적

##### 4. 세션별 폴더 구분 시스템
- **폴더 구조**: `session_YYYYMMDD_HHMMSS_타입_연도/`
- **파일 분리**: 사전 파일과 체크포인트 파일 분리 저장
- **데이터 보존**: 기존 데이터 삭제 없이 새 세션 생성
- **정리 기능**: `--clear-dictionary` 옵션으로 전체 삭제

##### 5. Graceful Shutdown 기능
- **신호 처리**: SIGINT, SIGTERM, SIGHUP, SIGQUIT 지원
- **안전한 종료**: 진행 중인 작업 완료 후 종료
- **상태 저장**: 체크포인트 자동 저장
- **리소스 정리**: 메모리 정리 및 임시 파일 삭제

#### 사용법
```bash
# 기본 수집 (세션별 폴더 생성)
python scripts/legal_term/collect_legal_terms.py --collection-type year --target-year 2024 --max-terms 100

# 기존 데이터 삭제 후 새로 시작
python scripts/legal_term/collect_legal_terms.py --collection-type year --target-year 2024 --max-terms 100 --clear-dictionary

# 특정 용어 상세조회
python scripts/legal_term/collect_legal_terms.py --search-detail "계약"

# 상태 확인
python scripts/legal_term/collect_legal_terms.py --status
```

---

## 🗓️ Week 10: HuggingFace Spaces 최적화 및 배포

### TASK 6.1: 성능 최적화
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 3일  
**우선순위**: Critical

#### 세부 작업
- [ ] 메모리 사용량 최적화
- [ ] 모델 로딩 최적화
- [ ] 추론 속도 개선
- [ ] 리소스 모니터링 구현

#### 산출물
- `scripts/optimize_performance.py`
- `source/utils/memory_manager.py`
- `docs/performance_optimization.md`

#### 완료 기준
- [ ] 메모리 사용량 14GB 이하
- [ ] 응답 시간 15초 이내
- [ ] 동시 사용자 10명 처리

---

### TASK 6.2: Docker 컨테이너 최적화
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] 멀티스테이지 빌드 적용
- [ ] 이미지 크기 최적화
- [ ] 보안 설정 강화
- [ ] 헬스체크 구현

#### 산출물
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

#### 완료 기준
- [ ] Docker 이미지 크기 최적화
- [ ] 보안 취약점 해결
- [ ] 컨테이너 안정성 확보

---

### TASK 6.3: 모니터링 시스템 구축
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] 시스템 메트릭 수집
- [ ] 로그 관리 시스템
- [ ] 알림 시스템 구축
- [ ] 대시보드 구현

#### 산출물
- `source/utils/monitoring.py`
- `scripts/setup_monitoring.py`
- `docs/monitoring_guide.md`

#### 완료 기준
- [ ] 실시간 모니터링 구축
- [ ] 알림 시스템 작동 확인
- [ ] 대시보드 완성

---

### TASK 6.4: 배포 자동화
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] GitHub Actions 워크플로우
- [ ] 자동 테스트 실행
- [ ] 자동 배포 파이프라인
- [ ] 롤백 시스템 구현

#### 산출물
- `.github/workflows/deploy.yml`
- `scripts/deploy.sh`
- `docs/deployment_guide.md`

#### 완료 기준
- [ ] CI/CD 파이프라인 구축
- [ ] 자동 배포 시스템 완성
- [ ] 롤백 기능 구현

---

## 🗓️ Week 11: 베타 테스트 및 피드백 수집

### TASK 7.1: 베타 테스트 계획 수립
**담당자**: QA 매니저  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] 테스터 모집 계획
- [ ] 테스트 시나리오 작성
- [ ] 피드백 수집 시스템 구축
- [ ] 테스트 환경 준비

#### 산출물
- `docs/beta_test_plan.md`
- `tests/beta_test_scenarios.md`
- `source/utils/feedback_collector.py`

#### 완료 기준
- [ ] 베타 테스트 계획 완성
- [ ] 테스터 20명 모집
- [ ] 피드백 시스템 구축

---

### TASK 7.2: 사용성 테스트 실행
**담당자**: UX 디자이너  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [ ] 사용자 행동 분석
- [ ] 히트맵 생성
- [ ] 사용성 문제점 파악
- [ ] 개선사항 도출

#### 산출물
- `docs/usability_test_report.md`
- `scripts/analyze_user_behavior.py`
- `docs/improvement_recommendations.md`

#### 완료 기준
- [ ] 사용성 테스트 완료
- [ ] 개선사항 20개 이상 수집
- [ ] 사용자 만족도 4.0/5.0 이상

---

### TASK 7.3: A/B 테스트 구현
**담당자**: 데이터 사이언티스트  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] A/B 테스트 프레임워크 구축
- [ ] 테스트 그룹 분할 로직
- [ ] 통계적 유의성 검증
- [ ] 결과 분석 및 보고

#### 산출물
- `source/utils/ab_test_manager.py`
- `tests/ab_test_scenarios.py`
- `docs/ab_test_results.md`

#### 완료 기준
- [ ] A/B 테스트 시스템 구축
- [ ] 테스트 결과 분석 완료
- [ ] 최적화 방안 도출

---

### TASK 7.4: 피드백 분석 및 개선
**담당자**: 풀스택 개발자  
**예상 소요시간**: 3일  
**우선순위**: High

#### 세부 작업
- [ ] 피드백 데이터 분석
- [ ] 우선순위별 개선사항 정리
- [ ] 버그 수정 및 기능 개선
- [ ] 성능 최적화

#### 산출물
- `docs/feedback_analysis_report.md`
- `docs/improvement_roadmap.md`
- `changelog.md`

#### 완료 기준
- [ ] 주요 버그 95% 이상 해결
- [ ] 사용자 피드백 반영 완료
- [ ] 성능 목표 달성

---

## 🗓️ Week 12: 정식 서비스 론칭

### TASK 8.1: 최종 QA 및 통합 테스트
**담당자**: QA 매니저  
**예상 소요시간**: 3일  
**우선순위**: Critical

#### 세부 작업
- [ ] 전체 기능 통합 테스트
- [ ] 성능 테스트 실행
- [ ] 보안 테스트 수행
- [ ] 사용자 시나리오 테스트

#### 산출물
- `tests/integration/test_full_system.py`
- `docs/qa_test_report.md`
- `docs/security_test_report.md`

#### 완료 기준
- [ ] 모든 테스트 통과
- [ ] 성능 목표 달성
- [ ] 보안 취약점 해결

---

### TASK 8.2: 문서화 완성
**담당자**: 기술 문서 작성자  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] README.md 완성
- [ ] API 문서 작성
- [ ] 사용자 가이드 작성
- [ ] 개발자 가이드 작성

#### 산출물
- `README.md`
- `docs/api/`
- `docs/user_guide/`
- `docs/developer_guide/`

#### 완료 기준
- [ ] 모든 문서 완성
- [ ] 문서 품질 검토 통과
- [ ] 사용자 피드백 반영

---

### TASK 8.3: 마케팅 및 홍보
**담당자**: 마케팅 매니저  
**예상 소요시간**: 2일  
**우선순위**: Medium

#### 세부 작업
- [ ] HuggingFace Spaces 등록
- [ ] 소셜 미디어 홍보
- [ ] 커뮤니티 공유
- [ ] 언론 보도 자료 작성

#### 산출물
- `docs/marketing_materials/`
- `docs/press_release.md`
- `social_media_posts/`

#### 완료 기준
- [ ] HuggingFace Spaces 공개
- [ ] 초기 사용자 100명 확보
- [ ] 커뮤니티 반응 수집

---

### TASK 8.4: 유지보수 체계 구축
**담당자**: DevOps 엔지니어  
**예상 소요시간**: 2일  
**우선순위**: High

#### 세부 작업
- [ ] 모니터링 시스템 강화
- [ ] 장애 대응 절차 수립
- [ ] 업데이트 프로세스 정의
- [ ] 사용자 지원 체계 구축

#### 산출물
- `docs/maintenance_guide.md`
- `docs/incident_response_plan.md`
- `scripts/maintenance/`

#### 완료 기준
- [ ] 유지보수 체계 구축 완료
- [ ] 장애 대응 절차 수립
- [ ] 지속적 개선 프로세스 확립

---

## 📊 TASK 관리 및 추적

### 우선순위 정의
- **Critical**: 프로젝트 성공에 필수적인 작업
- **High**: 중요한 기능이나 품질에 영향을 주는 작업
- **Medium**: 개선사항이나 추가 기능
- **Low**: 선택적 기능이나 장기적 개선

### 완료 기준 체크리스트
각 TASK는 다음 기준을 모두 만족해야 완료로 간주됩니다:
- [ ] 모든 세부 작업 완료
- [ ] 산출물 검토 통과
- [ ] 테스트 통과
- [ ] 문서화 완료
- [ ] 코드 리뷰 완료

### 리스크 관리
- **기술적 리스크**: 모델 성능, 메모리 제약, API 제한
- **일정 리스크**: 작업 지연, 의존성 문제, 리소스 부족
- **품질 리스크**: 버그 발생, 성능 저하, 사용자 불만

### 의존성 관리
- TASK 간 의존성을 명확히 정의
- 병렬 작업 가능한 TASK 식별
- 크리티컬 패스 관리
- 리소스 충돌 방지

---

## 🎯 성공 지표

### 기술적 지표
- [ ] 모든 핵심 기능 구현 완료
- [ ] 성능 목표 달성 (응답시간 15초 이내)
- [ ] 메모리 사용량 14GB 이하
- [ ] 에러율 5% 이하

### 품질 지표
- [ ] 코드 커버리지 80% 이상
- [ ] 사용자 만족도 4.0/5.0 이상
- [ ] 전문가 검토 통과
- [ ] 보안 취약점 0개

### 비즈니스 지표
- [ ] 초기 사용자 100명 확보
- [ ] 커뮤니티 피드백 수집
- [ ] 오픈소스 기여 활성화
- [ ] 지속 가능한 운영 체계 구축

이 TASK별 상세 개발 계획을 통해 LawFirmAI 프로젝트를 체계적이고 효율적으로 진행할 수 있습니다.

## 📚 참고 문서

- [프로젝트 구조 문서](../architecture/project_structure.md)
- [시스템 아키텍처 문서](../architecture/system_architecture.md)
- [모듈 인터페이스 문서](../architecture/module_interfaces.md)
- [하이브리드 검색 아키텍처 문서](../architecture/hybrid_search_architecture.md)
- [API 문서](../../api/law_open_api/README.md) - 국가법령정보센터 OpenAPI 가이드
- [API 매핑 테이블](../../api/law_open_api/API_MAPPING_TABLE.md) - API 파라미터 매핑
- [개발자 가이드](../../api/law_open_api/DEVELOPER_GUIDE.md) - 실무 개발 가이드
