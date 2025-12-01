# RAG 시스템 (RAG System)

이 섹션은 LawFirmAI 프로젝트의 LangChain 기반 RAG(Retrieval-Augmented Generation) 시스템 아키텍처와 구현에 대한 문서를 포함합니다.

## 📋 문서 목록

### 핵심 아키텍처
- **[LangChain RAG 아키텍처](rag_architecture.md)**: LangChain 기반 RAG 시스템의 상세 아키텍처 및 구현
- **[LangGraph 통합 가이드](langgraph_integration_guide.md)**: LangGraph 통합 및 개발 가이드
- **[LangChain/LangGraph 개발 규칙](langchain_langgraph_development_rules.md)**: 개발 규칙 및 가이드라인

### MLflow 통합
- **[MLflow 통합 가이드](mlflow_guide.md)**: MLflow를 활용한 인덱스 관리, 검색 품질 최적화, 최적 파라미터 통합 종합 가이드

### 성능 개선
- **[RAG 검색 성능 개선 계획](rag_search_performance_improvement_plan.md)**: 검색 성능 개선 계획 및 방안

## 🎯 현재 상태

### 📊 RAG 시스템 성능 (최신)
- **응답 생성 시간**: 
  - Generate Answer Final: 4.78초 (임계값 이하)
  - Generate Answer Stream: 5.63초 (31% 개선)

## 🔗 관련 섹션

- [02_Data_Embedding](../02_data/embedding/README.md): 벡터 임베딩
- [04_Models_Performance](../04_models/performance/README.md): 모델 성능 분석
- [07_API_Documentation](../07_api/README.md): API 문서
