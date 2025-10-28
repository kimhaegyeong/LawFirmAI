# 🔄 LawFirmAI 리팩토링 가이드

## 📋 개요

이 문서는 LawFirmAI 프로젝트의 리팩토링된 구조에 대한 개발 가이드입니다.

## 🏗️ 프로젝트 구조

### 새로운 디렉토리 구조

```
LawFirmAI/
├── core/                          # 핵심 비즈니스 로직
│   ├── agents/                   # LangGraph 에이전트
│   │   ├── workflow_service.py
│   │   ├── legal_workflow_enhanced.py
│   │   └── ...
│   ├── services/                 # 도메인별 서비스
│   │   ├── search/              # 검색 서비스
│   │   │   ├── hybrid_search_engine.py
│   │   │   ├── exact_search_engine.py
│   │   │   ├── semantic_search_engine.py
│   │   │   └── precedent_search_engine.py
│   │   ├── generation/          # 답변 생성
│   │   │   ├── answer_generator.py
│   │   │   ├── improved_answer_generator.py
│   │   │   └── answer_formatter.py
│   │   ├── enhancement/         # 품질 개선
│   │   │   └── confidence_calculator.py
│   │   └── integration/         # 통합 서비스
│   ├── models/                   # AI 모델
│   │   ├── model_manager.py
│   │   ├── sentence_bert.py
│   │   └── gemini_client.py
│   └── data/                     # 데이터 레이어
│       ├── database.py
│       ├── vector_store.py
│       └── data_processor.py
│
├── apps/                          # 애플리케이션
│   ├── streamlit/               # Streamlit 앱
│   └── api/                      # FastAPI 앱
│
├── infrastructure/               # 인프라
│   ├── utils/                    # 유틸리티
│   │   ├── langgraph_config.py
│   │   ├── logger.py
│   │   └── config.py
│   ├── monitoring/              # 모니터링
│   └── config/                  # 설정
│
└── tests/                        # 테스트
    ├── test_core_imports.py
    ├── test_core_workflow.py
    └── ...
```

## 📦 Import 패턴

### ❌ 이전 방식 (제거됨)
```python
# 상대 경로 (혼란스러움)
from ..utils.langgraph_config import LangGraphConfig
from ...services import ChatService

# 불명확한 경로
from source.services.langgraph.workflow_service import ...
```

### ✅ 현재 방식 (권장)
```python
# 절대 경로 (명확함)
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig
from core.services.search import HybridSearchEngine
from core.services.enhancement import ConfidenceCalculator
```

## 🔧 개발 방법

### 1. 모듈 추가

#### Search 서비스 추가
```python
# core/services/search/new_search.py
class NewSearchEngine:
    """새 검색 엔진"""
    pass

# core/services/search/__init__.py에 추가
from .new_search import NewSearchEngine
__all__ = [..., "NewSearchEngine"]
```

#### Generation 서비스 추가
```python
# core/services/generation/new_generator.py
class NewAnswerGenerator:
    """새 답변 생성기"""
    pass

# core/services/generation/__init__.py에 추가
from .new_generator import NewAnswerGenerator
__all__ = [..., "NewAnswerGenerator"]
```

### 2. Import 규칙

#### 절대 경로 사용
```python
# 프로젝트 루트 추가
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# core 모듈 import
from core.agents.workflow_service import LangGraphWorkflowService
from core.services.search import HybridSearchEngine
from infrastructure.utils.langgraph_config import LangGraphConfig
```

#### 모듈 간 Import
```python
# 같은 디렉토리 내
from .exact_search_engine import ExactSearchEngine

# 하위 디렉토리
from core.services.generation import AnswerGenerator

# 부모 디렉토리
from core.data import DatabaseManager
```

### 3. 서비스 작성 규칙

#### Naming Convention
```python
# 파일명: snake_case
answer_generator.py
hybrid_search_engine.py

# 클래스명: PascalCase
class AnswerGenerator:
    pass

# 함수/변수: snake_case
def generate_answer():
    pass
```

#### 구조
```python
# core/services/search/my_service.py
"""
서비스에 대한 설명
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MyService:
    """서비스 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
    
    def process(self, data: Dict[str, Any]) -> Any:
        """프로세싱 로직"""
        try:
            # 비즈니스 로직
            pass
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise
```

## 🧪 테스트 작성

### 새 테스트 추가
```python
# tests/test_new_service.py
"""
새 서비스 테스트
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_new_service():
    """새 서비스 테스트"""
    from core.services.new_module import NewService
    
    service = NewService()
    result = service.process({"test": "data"})
    
    assert result is not None
    return True


def run_all_tests():
    """모든 테스트 실행"""
    tests = [
        ("New Service", test_new_service),
    ]
    
    for name, test_func in tests:
        print(f"Testing {name}...")
        result = test_func()
        print(f"✅ {name}: {'PASS' if result else 'FAIL'}")
    
    return True


if __name__ == "__main__":
    run_all_tests()
```

## 🎯 주요 변경사항

### 제거된 것들
- ❌ `source/services/langgraph/` → ✅ `core/agents/`
- ❌ 상대 경로 import → ✅ 절대 경로 import
- ❌ 중복 서비스 → ✅ 통합된 서비스
- ❌ 평탄한 구조 → ✅ 계층적 구조

### 추가된 것들
- ✅ `core/` - 핵심 비즈니스 로직
- ✅ `apps/` - 애플리케이션 레이어
- ✅ `infrastructure/` - 인프라 레이어
- ✅ 명확한 도메인 분리

## 📚 참고 자료

### 핵심 모듈
- `core/agents/workflow_service.py` - LangGraph 워크플로우
- `core/services/search/` - 검색 엔진
- `core/services/generation/` - 답변 생성
- `core/models/` - AI 모델
- `core/data/` - 데이터 레이어

### 설정
- `infrastructure/utils/langgraph_config.py` - LangGraph 설정
- `infrastructure/utils/config.py` - 일반 설정

### 테스트
- `tests/test_core_imports.py` - Import 테스트
- `tests/test_core_workflow.py` - Workflow 테스트

## 🚀 빠른 시작

### Streamlit 앱 실행
```bash
cd streamlit
streamlit run app.py
```

### 테스트 실행
```bash
python tests/test_core_imports.py
python tests/test_core_workflow.py
```

### 새 서비스 추가
1. `core/services/{domain}/new_service.py` 생성
2. `core/services/{domain}/__init__.py` 업데이트
3. 테스트 작성
4. 문서 업데이트

## 💡 베스트 프랙티스

1. **절대 경로 사용**: 명확한 의존성 관리
2. **도메인 분리**: 각 서비스를 명확한 디렉토리에
3. **테스트 작성**: 모든 새 기능에 테스트 추가
4. **문서화**: 모든 클래스와 함수에 docstring
5. **로깅**: 적절한 로깅 사용

## 📝 체크리스트

새 기능 추가 시:
- [ ] 적절한 디렉토리에 파일 생성
- [ ] 절대 경로로 import
- [ ] `__init__.py` 업데이트
- [ ] 테스트 작성
- [ ] 문서 업데이트
- [ ] 로깅 추가
