# RAG 시스템 (RAG System)

이 섹션은 LawFirmAI 프로젝트의 LangChain 기반 RAG(Retrieval-Augmented Generation) 시스템 아키텍처와 구현에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[LangChain RAG 아키텍처](rag_architecture.md)**: LangChain 기반 RAG 시스템의 상세 아키텍처 및 구현
- **[하이브리드 검색 구현 가이드](hybrid_search_guide.md)**: 하이브리드 검색 시스템 구현 방법

## 🎯 현재 상태 (2025-10-19)

### ✅ Phase 4 업데이트 완료
- **Google Gemini 2.5 Flash Lite 통합**: Ollama 대신 클라우드 LLM으로 업그레이드
- **LangChain 기반 LLM 관리**: 안정적인 API 키 관리 및 LLM 호출
- **법률 용어 확장 시스템**: LLM 기반 동의어 및 관련 용어 자동 생성
- **검색 성능 최적화**: IVF 인덱스 및 PQ 양자화로 검색 속도 99.8% 향상
- **성능 모니터링**: 실시간 RAG 시스템 성능 추적

### 📊 RAG 시스템 성능
- **응답 생성 시간**: 평균 5.8초 (Gemini 2.5 Flash Lite)
- **검색 성능**: 평균 0.043초 (최적화된 벡터 인덱스)
- **벡터화 문서**: 33,598개 (6개 카테고리 판례 포함)
- **신뢰도 시스템**: 다중 요소 기반 답변 신뢰도 계산

## 🔗 관련 섹션

- [02_Data_Embedding](../02_data/embedding/README.md): 벡터 임베딩
- [04_Models_Performance](../04_models/performance/README.md): 모델 성능 분석
- [07_API_Documentation](../07_api/README.md): API 문서
