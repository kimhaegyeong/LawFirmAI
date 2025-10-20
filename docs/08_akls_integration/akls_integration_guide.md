# AKLS 통합 가이드

## 📚 개요

**AKLS (법률전문대학원협의회)** 표준판례 데이터를 LawFirmAI 시스템에 통합하는 과정과 사용법을 설명합니다.

## 🎯 통합 목표

- 법률전문대학원협의회의 표준판례 데이터를 LawFirmAI에 통합
- 기존 Assembly 데이터와 AKLS 데이터를 통합한 통합 검색 시스템 구축
- 표준판례 전용 검색 기능 제공
- Gradio 인터페이스에 AKLS 전용 탭 추가

## 📁 데이터 구조

### 원본 데이터
```
data/raw/akls/
├── 표준판례 전체(최종보고서).pdf
├── 형법표준판례 연구보고서.pdf
├── 230425 민법 표준판례 2023년 (1).pdf
├── 상법표준판례.pdf
├── 민사소송법 표준판례.pdf
└── ... (총 14개 PDF 파일)
```

### 처리된 데이터
```
data/processed/akls/
├── 형법표준판례_연구보고서.json
├── 민법_표준판례_2023년.json
├── 상법표준판례.json
└── ... (각 PDF별 JSON 파일)
```

### 벡터 인덱스
```
data/embeddings/akls_precedents/
├── akls_index.faiss          # FAISS 벡터 인덱스
├── akls_metadata.json        # 메타데이터
└── akls_documents.json       # 문서 정보
```

## 🔧 핵심 컴포넌트

### 1. AKLSProcessor
**파일**: `source/services/akls_processor.py`

**주요 기능**:
- PDF 파일에서 텍스트 추출
- 법률 영역 자동 분류 (형법, 민법, 상법, 민사소송법 등)
- 표준판례 구조 파싱 (사건번호, 법원, 선고일자 등)
- 메타데이터 추출 및 정규화

**사용법**:
```python
from source.services.akls_processor import AKLSProcessor

processor = AKLSProcessor()
documents = processor.process_akls_documents("data/raw/akls")
```

### 2. AKLSSearchEngine
**파일**: `source/services/akls_search_engine.py`

**주요 기능**:
- AKLS 전용 벡터 검색
- 법률 영역별 필터링 검색
- 표준판례 특화 검색 기능
- 검색 결과 랭킹 및 점수 계산

**사용법**:
```python
from source.services.akls_search_engine import AKLSSearchEngine

search_engine = AKLSSearchEngine()
results = search_engine.search("계약 해지", top_k=5)
```

### 3. EnhancedRAGService
**파일**: `source/services/enhanced_rag_service.py`

**주요 기능**:
- 기존 RAG 서비스와 AKLS 검색 통합
- 쿼리 라우팅 (표준판례 우선 vs 일반 검색)
- 검색 결과 통합 및 랭킹
- 향상된 답변 생성

**사용법**:
```python
from source.services.enhanced_rag_service import EnhancedRAGService

enhanced_rag = EnhancedRAGService()
result = enhanced_rag.search_with_akls("계약 해지에 대한 표준판례")
```

### 4. AKLSSearchInterface
**파일**: `gradio/components/akls_search_interface.py`

**주요 기능**:
- Gradio 기반 AKLS 전용 검색 인터페이스
- 법률 영역별 필터링 옵션
- 검색 결과 테이블 표시
- 통계 정보 제공

## 🚀 사용 방법

### 1. 데이터 처리
```bash
# AKLS 문서 처리 및 벡터 인덱스 생성
python scripts/process_akls_documents.py
```

### 2. Gradio 앱 실행
```bash
# Gradio 앱 실행 (AKLS 탭 포함)
cd gradio
python app.py
```

### 3. API 사용
```python
# Enhanced RAG Service 사용
from source.services.enhanced_rag_service import EnhancedRAGService

enhanced_rag = EnhancedRAGService()

# 표준판례 우선 검색
result = enhanced_rag.search_with_akls("형법 제250조 관련 판례")

# 법률 영역별 검색
result = enhanced_rag.search_by_law_area("계약 해지", "civil_law")
```

## 📊 검색 기능

### 1. 기본 검색
- 의미적 유사도 기반 검색
- 법률 영역별 자동 분류
- 검색 결과 점수 및 랭킹

### 2. 필터링 검색
- 법률 영역별 필터링 (형법, 민법, 상법, 민사소송법)
- 사건 유형별 필터링
- 법원별 필터링

### 3. 통합 검색
- 기존 Assembly 데이터와 AKLS 데이터 통합 검색
- 쿼리 유형에 따른 자동 라우팅
- 검색 결과 통합 및 랭킹

## 🧪 테스트

### 테스트 실행
```bash
# AKLS 통합 테스트
python tests/akls/test_akls_integration.py

# Gradio 인터페이스 테스트
python tests/akls/test_akls_gradio.py

# 성능 벤치마크 테스트
python tests/akls/test_akls_performance.py
```

### 테스트 결과
- ✅ 데이터 처리 테스트 통과
- ✅ 검색 엔진 테스트 통과
- ✅ RAG 통합 테스트 통과
- ✅ 성능 벤치마크 테스트 통과
- ✅ Gradio 인터페이스 테스트 통과

## 📈 성능 지표

### 검색 성능
- **평균 검색 시간**: 0.034초
- **최소 검색 시간**: 0.026초
- **최대 검색 시간**: 0.065초
- **성공률**: 100% (10/10 테스트 통과)

### 데이터 현황
- **처리된 PDF 파일**: 14개
- **벡터 인덱스 문서 수**: 14개
- **법률 영역 분포**: 형법, 민법, 상법, 민사소송법 등
- **메타데이터 필드**: 사건번호, 법원, 선고일자, 법률영역 등

## 🔧 설정

### pipeline_config.yaml 업데이트
```yaml
data_sources:
  akls_precedents:
    enabled: true
    priority: 5
    raw_path: "data/raw/akls"
    processed_path: "data/processed/akls"
    file_pattern: "*.pdf"
    metadata_key: "source"
    expected_value: "akls"
    document_type: "standard_precedent"

vectorization:
  akls_index_path: "data/embeddings/ml_enhanced_ko_sroberta_akls"

database:
  tables:
    akls_precedents:
      enabled: true
      fts_enabled: true
    akls_sections:
      enabled: true
      fts_enabled: true
```

## 🚨 주의사항

### 1. 메모리 사용량
- 벡터 인덱스 로딩 시 메모리 사용량 증가
- 대용량 문서 처리 시 충분한 메모리 확보 필요

### 2. 모델 의존성
- ko-sroberta-multitask 모델 필요
- FAISS 라이브러리 필요
- PyPDF2 또는 pypdf 라이브러리 필요

### 3. 파일 경로
- 상대 경로 기반으로 설정되어 있음
- 프로젝트 루트에서 실행 필요

## 🔄 업데이트 및 유지보수

### 새로운 AKLS 데이터 추가
1. `data/raw/akls/` 디렉토리에 새 PDF 파일 추가
2. `python scripts/process_akls_documents.py` 실행
3. 벡터 인덱스 자동 업데이트

### 성능 모니터링
- 검색 시간 모니터링
- 메모리 사용량 체크
- 검색 결과 품질 평가

## 📚 참고 자료

- [AKLS 공식 웹사이트](https://www.akls.or.kr/)
- [법률전문대학원협의회 표준판례](https://www.akls.or.kr/standard-precedent)
- [FAISS 문서](https://faiss.ai/)
- [Sentence Transformers 문서](https://www.sbert.net/)

## 🤝 기여 방법

1. 새로운 AKLS 데이터 추가
2. 검색 알고리즘 개선
3. 성능 최적화
4. 테스트 케이스 추가
5. 문서 개선
