# LawFirmAI 프로젝트 개요

## 🎯 프로젝트 소개

**프로젝트명**: LawFirmAI - 지능형 법률 AI 어시스턴트  
**목표**: HuggingFace Spaces 배포를 위한 법률 AI 시스템 개발  
**현재 상태**: ✅ Phase 8 완료 - Enhanced Chat Service 및 통합 검색 시스템 완료  
**마지막 업데이트**: 2025-10-25 (Enhanced Chat Service 및 통합 검색 엔진 완료)

## 🚀 핵심 기능

### 1. 데이터베이스 시스템
- **SQLite 데이터베이스**: 법률 및 판례 문서 저장
- **Assembly 법률**: 국회 법률정보시스템 데이터 수집 완료
- **판례 데이터**: 민사/형사/가사/조세/행정/특허 판례 분류 처리
- **AKLS 표준판례**: 법률전문대학원협의회 표준판례 데이터 통합 완료
- **법률 용어 사전**: 국가법령정보센터 법령용어사전 API 기반 수집 시스템
- **메타데이터 관리**: 구조화된 법령/판례 정보 저장

### 2. 벡터 임베딩 시스템
- **FAISS 벡터 인덱스**: 법률 및 판례 문서 벡터 임베딩
- **AKLS 전용 인덱스**: 표준판례 전용 벡터 인덱스 분리 관리
- **임베딩 모델**: ko-sroberta-multitask (768차원)
- **검색 성능**: 평균 응답 시간 < 1초
- **증분 업데이트**: 새로운 데이터 자동 처리

### 3. 하이브리드 검색 시스템
- **벡터 검색**: 의미적 유사도 기반 검색
- **정확 매칭**: 키워드 기반 정확한 매칭
- **AKLS 통합 검색**: 표준판례 전용 검색 엔진 통합
- **결과 통합**: 가중 평균으로 검색 결과 결합

### 4. Enhanced Chat Service (신규)
- **통합 채팅 서비스**: 모든 Phase 시스템을 통합한 완전한 채팅 서비스
- **Phase 1-3 통합**: 대화 맥락 강화, 개인화 분석, 장기 기억 시스템 완전 통합
- **법률 제한 시스템**: ML 통합 검증 시스템으로 안전한 법률 자문 제공
- **성능 최적화**: 메모리 관리, 캐싱 시스템, 실시간 모니터링
- **자연스러운 대화**: 감정 톤 조절, 개인화 스타일 학습, 실시간 피드백

### 5. 통합 검색 엔진 시스템 (신규)
- **Unified Search Engine**: 모든 검색 기능을 통합한 단일 검색 엔진
- **Integrated Law Search Service**: 통합 조문 검색 서비스
- **Enhanced Law Search Engine**: 향상된 법령 검색 엔진 (1,299라인)
- **현행법령 검색**: 국가법령정보센터 OpenAPI 기반 실시간 법령 검색
- **성능 모니터링**: 실시간 성능 추적 및 최적화

### 6. AI 모델 시스템
- **Google Gemini 2.5 Flash Lite**: 클라우드 LLM 모델
- **LangChain 기반**: 안정적인 LLM 관리
- **법률 용어 확장**: LLM 기반 동의어 자동 생성

## 🔧 기술 스택

- **백엔드**: FastAPI, SQLite, FAISS, LangChain
- **AI/ML**: Google Gemini 2.5 Flash Lite, ko-sroberta-multitask
- **프론트엔드**: Streamlit (현대적 웹 인터페이스)
- **검색**: 하이브리드 검색 (의미적 + 정확 매칭)
- **RAG**: ML 강화 RAG 시스템
- **배포**: Docker, HuggingFace Spaces 준비

## 📊 데이터 현황

- **법률 문서**: Assembly 데이터 수집 완료
- **AKLS 표준판례**: 14개 PDF 파일 처리 완료 (형법, 민법, 상법, 민사소송법 등)
- **벡터 임베딩**: ko-sroberta-multitask 모델 사용
- **검색 성능**: 평균 응답 시간 < 1초

## 📁 프로젝트 구조

```
LawFirmAI/
├── streamlit/                          # Streamlit 애플리케이션
│   └── streamlit_app.py               # Streamlit 메인 애플리케이션
├── source/                          # 핵심 모듈
│   ├── services/                    # 비즈니스 로직 (140+ 서비스)
│   │   ├── enhanced_chat_service.py # Enhanced Chat Service (2,574라인)
│   │   ├── unified_search_engine.py # 통합 검색 엔진 (460라인)
│   │   ├── integrated_law_search_service.py # 통합 조문 검색 (578라인)
│   │   ├── enhanced_law_search_engine.py # 향상된 법령 검색 (1,299라인)
│   │   ├── chat_service.py          # 기본 채팅 서비스
│   │   ├── rag_service.py           # RAG 서비스
│   │   ├── hybrid_search_engine.py  # 하이브리드 검색
│   │   ├── question_classifier.py   # 질문 분류기
│   │   ├── performance_monitor.py   # 성능 모니터링 (356라인)
│   │   └── ...                      # 기타 서비스들
│   ├── data/                        # 데이터 처리
│   │   ├── database.py              # 데이터베이스 관리
│   │   └── vector_store.py          # 벡터 저장소
│   ├── models/                      # AI 모델
│   └── utils/                       # 유틸리티
├── data/                            # 데이터 파일
│   ├── lawfirm.db                   # SQLite 데이터베이스
│   ├── embeddings/                  # 벡터 임베딩
│   │   └── akls_precedents/         # AKLS 전용 벡터 인덱스
│   ├── raw/                         # 원본 데이터
│   │   └── akls/                    # AKLS 원본 PDF 파일
│   └── processed/                   # 전처리된 데이터
│       └── akls/                    # AKLS 처리된 JSON 파일
├── scripts/                         # 유틸리티 스크립트
│   └── process_akls_documents.py    # AKLS 문서 처리 스크립트
├── tests/                           # 테스트 코드
│   └── akls/                        # AKLS 통합 테스트
└── docs/                            # 문서
```

## 🎉 주요 성과

### 시스템 완성도
- ✅ **Enhanced Chat Service**: Phase 1-3 통합 완전한 채팅 서비스 (2,574라인)
- ✅ **통합 검색 엔진**: 모든 검색 기능을 통합한 단일 엔진 시스템
- ✅ **현행법령 검색**: 국가법령정보센터 OpenAPI 기반 실시간 검색
- ✅ **성능 최적화**: 실시간 모니터링 및 메모리 관리 시스템
- ✅ **완전한 RAG 시스템**: LangChain 기반 검색 증강 생성
- ✅ **AKLS 통합**: 법률전문대학원협의회 표준판례 완전 통합
- ✅ **하이브리드 검색**: 의미적 검색 + 정확 매칭 통합
- ✅ **지능형 챗봇**: 질문 유형별 최적화된 답변 시스템
- ✅ **완전한 API**: RESTful API 및 웹 인터페이스
- ✅ **컨테이너화**: Docker 기반 배포 준비 완료

### 기술적 혁신
- ✅ **규칙 기반 파서**: 안정적인 법률 문서 구조 분석
- ✅ **하이브리드 아키텍처**: 다중 검색 방식 통합
- ✅ **확장 가능한 설계**: 모듈화된 서비스 아키텍처
- ✅ **지능형 질문 분류**: 질문 유형 자동 분류 및 최적화
- ✅ **컨텍스트 최적화**: 토큰 제한 내에서 관련성 높은 정보 선별

## 🚀 다음 단계 계획

### 1. 데이터 확장 (우선순위: 높음)
- 추가 판례 데이터 수집 및 처리
- 헌재결정례 데이터 수집 및 임베딩
- 법령해석례 데이터 수집 및 임베딩

### 2. 시스템 고도화 (우선순위: 중간)
- API 성능 최적화
- 법률 용어 사전 확장 및 업데이트
- 질문 유형 분류 정확도 향상

### 3. 기능 확장 (우선순위: 중간)
- 계약서 분석 기능 고도화
- 다국어 지원 (영어, 일본어)
- 개인화된 답변 시스템