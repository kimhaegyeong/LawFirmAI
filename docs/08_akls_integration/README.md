# AKLS 통합 문서

## 📚 문서 개요

이 디렉토리는 **AKLS (법률전문대학원협의회)** 표준판례 데이터를 LawFirmAI 시스템에 통합하는 과정과 사용법에 대한 문서를 포함합니다.

## 📁 문서 구조

- `akls_integration_guide.md` - AKLS 통합 가이드 (메인 문서)

## 🎯 주요 내용

### 통합 과정
- AKLS PDF 데이터 처리
- 벡터 임베딩 생성
- 검색 엔진 통합
- Gradio 인터페이스 추가

### 핵심 컴포넌트
- **AKLSProcessor**: PDF 데이터 처리
- **AKLSSearchEngine**: 전용 검색 엔진
- **EnhancedRAGService**: 통합 RAG 서비스
- **AKLSSearchInterface**: Gradio 인터페이스

### 성능 지표
- 평균 검색 시간: 0.034초
- 처리된 문서: 14개 PDF 파일
- 검색 성공률: 100%

## 🚀 빠른 시작

1. **데이터 처리**
   ```bash
   python scripts/process_akls_documents.py
   ```

2. **Gradio 앱 실행**
   ```bash
   cd gradio
   python app.py
   ```

3. **테스트 실행**
   ```bash
   python tests/akls/test_akls_integration.py
   ```

## 📊 통합 현황

- ✅ **데이터 처리**: 14개 PDF 파일 완료
- ✅ **벡터 인덱스**: FAISS 인덱스 생성 완료
- ✅ **검색 엔진**: AKLS 전용 검색 엔진 구현 완료
- ✅ **RAG 통합**: Enhanced RAG Service 구현 완료
- ✅ **UI 통합**: Gradio 인터페이스 추가 완료
- ✅ **테스트**: 종합 테스트 통과 완료

## 🔗 관련 문서

- [프로젝트 개요](../01_project_overview/project_overview.md)
- [RAG 시스템](../05_rag_system/rag_architecture.md)
- [데이터 처리](../03_data_processing/preprocessing_guide.md)
- [벡터 임베딩](../04_vector_embedding/embedding_guide.md)
