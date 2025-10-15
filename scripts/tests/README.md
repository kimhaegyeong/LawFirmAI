# Tests Directory

이 디렉토리는 LawFirmAI 프로젝트의 다양한 테스트 스크립트들을 포함합니다.

## 📁 파일 목록

### 벡터 임베딩 관련 테스트
- **`test_bge_m3_korean.py`**: BGE-M3-Korean 모델 테스트 및 Sentence-BERT와의 성능 비교
- **`test_simple_embedding.py`**: 간단한 벡터 임베딩 생성 및 검색 테스트
- **`test_vector_builder.py`**: ML 강화 벡터 빌더 테스트
- **`test_vector_store.py`**: 벡터 스토어 직접 테스트

### 데이터베이스 관련 테스트
- **`test_law_record.py`**: 법률 레코드 준비 함수 테스트
- **`test_real_data.py`**: 실제 데이터를 사용한 데이터베이스 임포트 테스트

## 🚀 사용법

### 개별 테스트 실행
```bash
# 벡터 임베딩 테스트
python scripts/tests/test_simple_embedding.py

# BGE-M3 모델 테스트
python scripts/tests/test_bge_m3_korean.py

# 데이터베이스 테스트
python scripts/tests/test_law_record.py
```

### 전체 테스트 실행
```bash
# 모든 테스트 실행 (추후 구현 예정)
python scripts/tests/run_all_tests.py
```

## 📋 테스트 카테고리

### 1. 벡터 임베딩 테스트
- **목적**: 벡터 임베딩 생성 및 검색 기능 검증
- **모델**: ko-sroberta-multitask, BGE-M3
- **기능**: 임베딩 생성, 검색, 성능 비교

### 2. 데이터베이스 테스트
- **목적**: 데이터베이스 스키마 및 임포트 기능 검증
- **기능**: 레코드 준비, 데이터 변환, 스키마 검증

### 3. 통합 테스트
- **목적**: 전체 시스템의 통합 동작 검증
- **기능**: 엔드투엔드 테스트, 성능 벤치마크

## 🔧 테스트 환경 요구사항

### 필수 라이브러리
```bash
pip install torch transformers sentence-transformers
pip install FlagEmbedding  # BGE-M3 테스트용
pip install faiss-cpu
pip install sqlite3
```

### 데이터 요구사항
- **벡터 임베딩 테스트**: 별도 데이터 불필요
- **데이터베이스 테스트**: `data/lawfirm.db` 필요
- **실제 데이터 테스트**: `data/processed/assembly/law/20251013_ml/` 필요

## 📊 테스트 결과

### 성공 기준
- ✅ 모든 테스트가 오류 없이 완료
- ✅ 예상 결과와 실제 결과 일치
- ✅ 성능 지표가 기준치 이상

### 실패 시 대응
- ❌ Import 오류: 필요한 라이브러리 설치 확인
- ❌ 데이터 오류: 데이터 파일 경로 및 형식 확인
- ❌ 성능 오류: 시스템 리소스 및 설정 확인

## 🚀 향후 계획

### 추가 예정 테스트
- **API 엔드포인트 테스트**: FastAPI 서비스 테스트
- **Gradio 인터페이스 테스트**: 웹 UI 기능 테스트
- **성능 벤치마크**: 대용량 데이터 처리 성능 테스트
- **통합 테스트**: 전체 시스템 엔드투엔드 테스트

### 자동화 계획
- **CI/CD 통합**: GitHub Actions를 통한 자동 테스트
- **테스트 리포트**: 자동화된 테스트 결과 리포트 생성
- **성능 모니터링**: 지속적인 성능 지표 추적

---

**마지막 업데이트**: 2025-10-15  
**관리자**: LawFirmAI 개발팀
