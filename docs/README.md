# LawFirmAI 문서 인덱스

## 📚 문서 구조

이 문서는 LawFirmAI 프로젝트의 전체 문서 구조를 안내합니다.

LawFirmAI는 **LangGraph 기반 법률 AI 어시스턴트**로, 법률 질문 처리, 문서 분석, 판례 검색 등의 기능을 제공합니다.

## 📁 문서 디렉토리

### 01. 시작하기 (`01_getting_started/`)
프로젝트 개요, 아키텍처, 구조 등 시작에 필요한 정보
- [프로젝트 개요](01_getting_started/project_overview.md)
- [프로젝트 구조](01_getting_started/project_structure.md)
- [아키텍처](01_getting_started/architecture.md)
- [프론트엔드 개발 가이드](01_getting_started/frontend_guide.md)

### 02. 데이터 (`02_data/`)
데이터 수집, 처리, 임베딩 관련 문서
- **collection/**: 데이터 수집 가이드
- **processing/**: 데이터 전처리 가이드
- **embedding/**: 벡터 임베딩 가이드

### 03. RAG 시스템 (`05_rag_system/`)
LangGraph 기반 RAG 시스템 문서
- [RAG 아키텍처](05_rag_system/rag_architecture.md)
- [LangGraph 통합 가이드](05_rag_system/langgraph_integration_guide.md)
- [개발 규칙](05_rag_system/langchain_langgraph_development_rules.md)

### 04. 모델 (`04_models/`)
AI 모델 성능 최적화 및 벤치마크
- **performance/**: 성능 최적화 가이드 및 보고서

### 05. 품질 관리 (`05_quality/`)
품질 개선, 키워드 시스템, 프롬프트 강화
- [품질 개선 시스템](05_quality/quality_improvement_system.md)
- [키워드 확장 보고서](05_quality/keyword_expansion_report.md)
- [하이브리드 키워드 시스템](05_quality/hybrid_keyword_management.md)
- [프롬프트 시스템 강화](05_quality/prompt_system_enhancement.md)

### 06. 배포 (`06_deployment/`)
배포 가이드 및 운영 문서
- [배포 가이드](06_deployment/Deployment_Guide.md)
- [AWS 배포 가이드](06_deployment/aws_deployment_quickstart.md)
- [HuggingFace Spaces 최적화](06_deployment/huggingface_spaces_optimization_plan.md)

### 07. API (`07_api/`)
API 문서 및 통합 가이드
- [API 문서](07_api/API_Documentation.md)
- [API 엔드포인트](07_api/api_endpoints.md)
- **open_law/**: 국가법령정보센터 Open API 가이드
- **integrations/**: 외부 시스템 통합 가이드

### 08. 기능 (`08_features/`)
특정 기능 개발 계획
- [법률 용어 확장 계획](08_features/legal_term_expansion_development_plan.md)

### 09. 사용자 가이드 (`09_user_guide/`)
사용자 가이드 및 문제 해결
- [사용자 가이드](09_user_guide/user_guide.md)
- [사용자 가이드 (메인)](09_user_guide/User_Guide_main.md)

### 10. 기술 참고 (`10_technical_reference/`)
기술 상세 참고 문서
- [Core 모듈 가이드](10_technical_reference/core_modules_guide.md)
- [데이터베이스 스키마](10_technical_reference/database_schema.md)
- [LangGraph Node I/O](10_technical_reference/langgraph_node_io.md)
- [개발 규칙](10_technical_reference/development_rules.md)
- [인코딩 개발 규칙](10_technical_reference/encoding_development_rules.md)
- [문제 해결 가이드](10_technical_reference/Troubleshooting_Guide.md)

### 참고 자료 (`reference/`)
참고 문서 및 개선 계획
- **improvement_plans/**: 개선 계획서
  - [LangGraph 워크플로우 개선 계획](reference/improvement_plans/langgraph_improvement.md)

## 🔍 빠른 찾기

### 개발자용
- 프로젝트 구조: [01_getting_started/project_structure.md](01_getting_started/project_structure.md)
- 프론트엔드 가이드: [01_getting_started/frontend_guide.md](01_getting_started/frontend_guide.md)
- 개발 규칙: [10_technical_reference/development_rules.md](10_technical_reference/development_rules.md)
- Core 모듈 가이드: [10_technical_reference/core_modules_guide.md](10_technical_reference/core_modules_guide.md)

### 배포 관련
- 배포 가이드: [06_deployment/Deployment_Guide.md](06_deployment/Deployment_Guide.md)
- AWS 배포: [06_deployment/aws_deployment_quickstart.md](06_deployment/aws_deployment_quickstart.md)

### 사용자용
- 사용자 가이드: [09_user_guide/user_guide.md](09_user_guide/user_guide.md)
- API 문서: [07_api/API_Documentation.md](07_api/API_Documentation.md)

## 📝 문서 작성 가이드

새 문서를 추가할 때는 다음 규칙을 따르세요:

1. **적절한 폴더에 배치**: 주제에 맞는 폴더에 문서를 추가하세요
2. **README.md 업데이트**: 해당 폴더의 README.md에 새 문서를 추가하세요
3. **링크 일관성**: 상대 경로를 사용하여 다른 문서로의 링크를 작성하세요
4. **네이밍 규칙**: 파일명은 소문자+언더스코어로 작성하세요 (`snake_case.md`)

## 🔄 문서 업데이트 이력

- **2024-12**: 폴더 구조 재구성 (제안 1: 주제 기반 계층 구조 적용)
