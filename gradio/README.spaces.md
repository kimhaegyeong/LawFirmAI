# ⚖️ LawFirmAI - 법률 AI 어시스턴트

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)](https://github.com/your-repo/lawfirm-ai)

> **Phase 2 완료** - 지능형 챗봇 시스템 구현 완료된 법률 AI 어시스턴트

## 🎯 프로젝트 개요

LawFirmAI는 한국 법률 문서를 기반으로 한 AI 어시스턴트입니다. Phase 2에서 구현된 지능형 질문 분류, 동적 검색 가중치, 구조화된 답변, 신뢰도 시스템을 통해 법률 질문에 정확하고 신뢰할 수 있는 답변을 제공합니다.

### 🚀 Phase 2 신규 기능

- **지능형 질문 분류**: 사용자 질문을 자동으로 분석하여 최적의 답변 전략 선택
- **동적 검색 가중치**: 질문 유형에 따라 법률과 판례 검색 비중 자동 조정
- **구조화된 답변**: 일관된 형식의 전문적이고 읽기 쉬운 답변 제공
- **신뢰도 표시**: 답변의 신뢰성을 수치화하여 사용자에게 투명성 제공
- **컨텍스트 최적화**: 토큰 제한 내에서 가장 관련성 높은 정보만 선별
- **법률 용어 확장**: 동의어 및 관련 용어를 통한 검색 정확도 향상
- **대화 맥락 관리**: 세션 기반 대화 이력 관리로 연속성 있는 답변 제공

### 📊 데이터 현황

- **법률 문서**: 4,321개 법률 문서 완전한 전처리 및 구조화
- **법률 조문**: 180,684개 조문의 벡터 임베딩 생성
- **판례 데이터**: 397개 민사 판례 파일 처리 완료
- **판례 섹션**: 15,589개 판례 섹션 벡터 임베딩 생성
- **FAISS 인덱스**: 법률과 판례를 별도 인덱스로 관리

## 🛠️ 기술 스택

### AI/ML
- **KoBART**: 한국어 생성 모델 (법률 특화 파인튜닝)
- **Sentence-BERT**: 텍스트 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **FAISS**: 벡터 검색 엔진
- **질문 분류 모델**: 사용자 질문 유형 자동 분류

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스 (정확한 매칭 검색)
- **FAISS**: 벡터 데이터베이스 (의미적 검색)
- **지능형 검색 엔진**: 질문 유형별 동적 가중치 검색
- **신뢰도 계산 시스템**: 답변 신뢰성 수치화

### Frontend
- **Gradio**: 웹 인터페이스
- **HuggingFace Spaces**: 배포 플랫폼

## 🚀 빠른 시작

### HuggingFace Spaces에서 실행

1. **Space 생성**: HuggingFace에서 새로운 Space 생성
2. **Docker 설정**: Docker 설정으로 배포
3. **환경 변수**: 필요한 환경 변수 설정
4. **배포**: 자동 배포 완료

### 로컬에서 실행

```bash
# 저장소 클론
git clone https://github.com/your-repo/lawfirm-ai.git
cd lawfirm-ai

# 가상환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r gradio/requirements_spaces.txt

# 애플리케이션 실행
cd gradio
python app.py
```

## 💬 사용법

### 질문 유형별 사용법

#### 1. 판례 검색 질문
- **예시**: "손해배상 관련 판례를 찾아주세요"
- **특징**: 판례 검색에 최적화 (판례 가중치 80%)
- **답변**: 사건명, 사건번호, 법원 정보, 판시사항, 판결요지 포함

#### 2. 법률 문의 질문
- **예시**: "민법 제750조의 내용이 무엇인가요?"
- **특징**: 법률 조문 해석에 최적화 (법률 가중치 80%)
- **답변**: 정확한 법률 조문 인용 및 해석

#### 3. 법적 조언 질문
- **예시**: "계약 해제 방법을 조언해주세요"
- **특징**: 법률과 판례 균형 검색 (각각 50%)
- **답변**: 단계별 해결 방법 및 필요한 증거 자료

#### 4. 절차 안내 질문
- **예시**: "이혼 절차는 어떻게 진행하나요?"
- **특징**: 절차 설명에 최적화 (법률 가중치 60%)
- **답변**: 단계별 절차, 필요 서류, 처리 기간

#### 5. 용어 해설 질문
- **예시**: "불법행위의 정의를 알려주세요"
- **특징**: 용어 해설에 최적화 (법률 가중치 70%)
- **답변**: 정확한 법률적 정의 및 관련 법률 조문

#### 6. 일반 질문
- **예시**: "법률에 대해 궁금한 것이 있습니다"
- **특징**: 균형잡힌 검색 (법률/판례 각각 40%)
- **답변**: 포괄적인 정보 제공

### 신뢰도 시스템

| 수준 | 점수 범위 | 설명 | 권장사항 |
|------|----------|------|----------|
| 🟢 **HIGH** | 80% 이상 | 높은 신뢰도 | 전문가 수준 답변, 신뢰 가능 |
| 🟡 **MEDIUM** | 60-80% | 보통 신뢰도 | 참고용 답변, 추가 확인 권장 |
| 🟠 **LOW** | 40-60% | 낮은 신뢰도 | 기본 정보만 제공, 전문가 상담 권장 |
| 🔴 **VERY_LOW** | 40% 미만 | 매우 낮은 신뢰도 | 전문가 상담 필수 |

## 📈 성능 지표

### 현재 달성된 성능

| 지표 | 값 | 설명 |
|------|-----|------|
| **평균 검색 시간** | < 1초 | 매우 빠른 검색 성능 |
| **처리 성능** | 17분 | 397개 판례 파일 → 15,589개 벡터 임베딩 |
| **성공률** | 99.9% | 높은 안정성 |
| **메모리 사용량** | 최적화됨 | HuggingFace Spaces 환경 최적화 |
| **질문 분류 정확도** | 85% | 6가지 질문 유형 자동 분류 |

## 🔧 환경 변수

### 필수 환경 변수

```bash
# HuggingFace Spaces 설정
HUGGINGFACE_SPACES=true
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# 데이터베이스 경로
DATABASE_PATH=data/lawfirm.db

# 벡터 임베딩 경로
VECTOR_STORE_PATH=data/embeddings/ml_enhanced_ko_sroberta
PRECEDENT_VECTOR_STORE_PATH=data/embeddings/ml_enhanced_ko_sroberta_precedents
```

### 선택적 환경 변수

```bash
# Ollama 설정 (로컬 LLM 사용 시)
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5:7b

# 로깅 레벨
LOG_LEVEL=INFO

# 메모리 최적화
MAX_MEMORY_PERCENT=85.0
```

## 📚 API 문서

### 주요 엔드포인트

- `POST /api/v1/chat/intelligent-v2` - 지능형 채팅 v2 (모든 개선사항 통합)
- `GET /api/v1/system/status` - 시스템 상태 확인 (모든 컴포넌트 점검)

### 사용 예제

```python
import requests

# 지능형 채팅 v2 요청
response = requests.post(
    "http://localhost:7860/api/v1/chat/intelligent-v2",
    json={
        "message": "계약 해제 조건이 무엇인가요?",
        "session_id": "user_session_123",
        "max_results": 10,
        "include_law_sources": True,
        "include_precedent_sources": True,
        "include_conversation_history": True,
        "context_optimization": True,
        "answer_formatting": True
    }
)

result = response.json()
print(f"질문 유형: {result['question_type']}")
print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence']['reliability_level']}")
```

## ⚠️ 주의사항 및 면책 조항

### 면책 조항
- 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.
- 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.
- 답변의 신뢰도가 낮은 경우 전문가 상담을 권장합니다.

### 사용 권장사항

1. **신뢰도 확인**: 답변의 신뢰도 수준을 확인하세요
2. **추가 확인**: 중요한 법률 문제는 반드시 변호사와 상담하세요
3. **개인정보 보호**: 개인정보나 민감한 정보는 질문에 포함하지 마세요

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [HuggingFace](https://huggingface.co/) - AI 모델 및 Spaces 플랫폼 제공
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [Gradio](https://gradio.app/) - UI 프레임워크
- [LangChain](https://langchain.com/) - RAG 시스템 프레임워크

---

*LawFirmAI는 법률 전문가의 도구로 사용되며, 법률 자문을 대체하지 않습니다. 중요한 법률 문제는 반드시 자격을 갖춘 법률 전문가와 상담하시기 바랍니다.*
