# RAG 시스템 (RAG System)

이 섹션은 LawFirmAI 프로젝트의 LangChain/LangGraph 기반 RAG(Retrieval-Augmented Generation) 시스템 아키텍처와 구현에 대한 문서를 포함합니다.

## 📋 문서 목록

- **[LangChain RAG 아키텍처](rag_architecture.md)**: LangChain 기반 RAG 시스템의 상세 아키텍처 및 구현
- **[LangChain/LangGraph 개발 규칙](langchain_langgraph_development_rules.md)**: LangChain/LangGraph 개발 가이드
- **[LangGraph 통합 가이드](langgraph_integration_guide.md)**: LangGraph 통합 방법 및 워크플로우
- **[LangGraph 마이그레이션 요약](../08_langgraph_migration_summary.md)**: LangGraph 전체 마이그레이션 내용
- **[LangGraph 테스트 결과](../08_langgraph_test_results.md)**: LangGraph 테스트 결과 및 검증
- **[LangGraph 최종 보고서](../08_langgraph_final_summary.md)**: LangGraph 최종 완료 보고서

## 🆕 LangGraph 워크플로우 (2024)

### ✅ LangGraph 통합 완료
- **Phase 1-5 완료**: 모든 핵심 기능 구현 완료
- **20개 노드 구현**: 입력 검증 → 특수 쿼리 처리 → 하이브리드 분석 → 폴백 체인 → 후처리
- **조건부 라우팅**: 6개 라우팅 포인트로 스마트 워크플로우 제어
- **실제 서비스 연결**: CurrentLawSearchEngine, UnifiedSearchEngine 통합

### 🔧 워크플로우 구조
```
입력 검증 → 특수 쿼리 감지 → 질문 분류 → 하이브리드 분석 
→ 법률 제한 검증 → 문서 검색 → Phase 처리 (대화 맥락, 개인화, 장기 기억)
→ 답변 생성 → 폴백 체인 (4단계) → 후처리 → 최종 답변
```

### 📊 주요 Phase
1. **Phase 1**: 입력 검증 및 특수 쿼리 감지 (법률 조문, 계약서)
2. **Phase 2**: 하이브리드 질문 분석 및 법률 제한 검증
3. **Phase 3**: Phase 시스템 통합 (대화 맥락 강화, 개인화, 장기 기억)
4. **Phase 4**: 4단계 폴백 체인 (답변 생성 보장)
5. **Phase 5**: 후처리 및 품질 검증

### 🎯 테스트 결과
- **성공률**: 100% (모든 Phase 테스트 통과)
- **처리 단계**: 13개 단계
- **평균 처리 시간**: 4.14초
- **오류 발생**: 0개

자세한 내용은 [LangGraph 마이그레이션 요약](../08_langgraph_migration_summary.md) 참조

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

- [04_Vector_Embedding](../04_vector_embedding/README.md): 벡터 임베딩
- [06_Models_Performance](../06_models_performance/README.md): 모델 성능 분석
- [08_API_Documentation](../08_api_documentation/README.md): API 문서

---

**LawFirmAI 개발팀**
