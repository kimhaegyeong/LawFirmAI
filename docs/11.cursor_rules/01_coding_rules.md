# 코딩 규칙

## 0. 코드 수정 우선 원칙 (CRITICAL)

**기존 소스코드 수정을 최우선으로 하고, 신규 파일 생성은 꼭 필요한 경우에만 허용합니다.**

### 원칙
1. **기존 파일 수정 우선**
   - 기능 추가/수정 시 기존 파일을 먼저 검토하고 수정
   - 기존 클래스/함수에 기능을 확장하는 방식으로 구현
   - 기존 코드 구조를 최대한 유지하면서 개선

2. **신규 파일 생성 조건**
   - 기존 파일 수정으로는 요구사항을 충족할 수 없는 경우에만 허용
   - 완전히 새로운 기능 영역이 필요한 경우
   - 기존 파일이 이미 너무 크고 복잡하여 분리가 필요한 경우 (1000줄 이상)
   - 아키텍처상 독립적인 모듈이 필요한 경우

3. **파일 생성 전 체크리스트**
   - [ ] 기존 파일에서 유사한 기능을 찾았는가?
   - [ ] 기존 클래스/함수에 기능을 추가할 수 있는가?
   - [ ] 기존 파일을 확장하는 방식으로 구현할 수 있는가?
   - [ ] 신규 파일이 정말 필요한가? (아키텍처상 분리가 필요한가?)

4. **권장 사항**
   ```python
   # ❌ 나쁜 예: 새로운 기능마다 새 파일 생성
   # services/chat_service_v2.py
   # services/chat_service_enhanced.py
   # services/chat_service_improved.py
   
   # ✅ 좋은 예: 기존 파일에 기능 추가
   # services/chat_service.py에 새로운 메서드 추가
   class ChatService:
       def process_message(self, message: str) -> str:
           # 기존 기능
           pass
       
       def process_message_with_context(self, message: str, context: str) -> str:
           # 새로운 기능 추가
           pass
   ```

5. **예외 사항**
   - 테스트 파일 (`test_*.py`, `*_test.py`)은 기능별로 분리 허용
   - 설정 파일 (`config.py`, `settings.py`)은 환경별 분리 허용
   - 문서 파일 (`*.md`)은 주제별 분리 허용
   - 마이그레이션 스크립트는 버전별 분리 허용

6. **리팩토링 가이드**
   - 기존 파일이 너무 커진 경우 (1000줄 이상):
     - 먼저 기존 파일 내에서 클래스/함수로 모듈화 시도
     - 그래도 부족한 경우에만 파일 분리 고려
   - 중복 코드 발견 시:
     - 공통 유틸리티 함수로 추출하여 기존 파일에 추가
     - 여러 파일에서 사용되는 경우에만 별도 유틸리티 파일 생성

## 0-1. 코드 변경 최소화 원칙 (CRITICAL)

**원본 코드를 최대한 보존하고, 불필요한 주석이나 설명을 과도하게 추가하지 않습니다.**

### 원칙
1. **원본 코드 보존 우선**
   - 기존 코드의 구조와 스타일을 최대한 유지
   - 기능 수정/추가 시에도 원본 코드의 형태를 최대한 보존
   - 불필요한 리팩토링이나 스타일 변경 금지

2. **주석 추가 제한**
   - 기존 코드에 주석이 없는 경우, 새로 추가하지 않음
   - 기존 주석이 있는 경우에만 유지하거나 최소한으로 수정
   - 코드 자체로 이해 가능한 경우 주석 추가 금지
   - 복잡한 로직이나 비즈니스 규칙이 있는 경우에만 최소한의 주석 허용

3. **코드 변경 가이드라인**
   ```python
   # ❌ 나쁜 예: 과도한 주석 추가로 원본 코드 변경
   def process_message(self, message: str) -> str:
       """
       메시지를 처리하는 함수입니다.
       이 함수는 사용자로부터 받은 메시지를 분석하고
       적절한 응답을 생성하여 반환합니다.
       
       Args:
           message (str): 사용자가 입력한 메시지
       
       Returns:
           str: 처리된 응답 메시지
       """
       # 메시지 전처리
       processed = message.strip()
       # 응답 생성
       response = self._generate_response(processed)
       # 결과 반환
       return response
   
   # ✅ 좋은 예: 원본 코드 최대한 보존
   def process_message(self, message: str) -> str:
       processed = message.strip()
       response = self._generate_response(processed)
       return response
   ```

4. **주석 추가 허용 조건**
   - 복잡한 알고리즘이나 비즈니스 로직의 경우
   - 외부 API나 라이브러리의 특이한 동작을 설명하는 경우
   - 버그 수정이나 임시 해결책을 표시하는 경우 (TODO, FIXME, HACK 등)
   - 기존 주석이 잘못된 정보를 포함하고 있어 수정이 필요한 경우

5. **금지 사항**
   - 명확한 코드에 설명 주석 추가
   - 변수명이나 함수명으로 충분히 설명되는 경우 주석 추가
   - 기존 코드 스타일과 다른 주석 스타일 강제 적용
   - 문서화 문자열(docstring)을 모든 함수에 강제로 추가

6. **예외 사항**
   - 공개 API나 라이브러리 함수는 docstring 필요
   - 복잡한 수학적 알고리즘이나 비즈니스 규칙은 주석 허용
   - 팀 내부 규칙으로 문서화가 필요한 경우

## 1. 파일 명명 규칙
- **Python 파일**: snake_case (예: `chat_service.py`)
- **클래스**: PascalCase (예: `ChatService`)
- **함수/변수**: snake_case (예: `process_message`)
- **상수**: UPPER_SNAKE_CASE (예: `MAX_RETRY_COUNT`)

## 2. Import 규칙
```python
# 표준 라이브러리
import os
import sys
from typing import List, Dict, Optional

# 서드파티 라이브러리
import torch
import numpy as np
from fastapi import FastAPI
from transformers import AutoTokenizer
from langgraph.graph import StateGraph, END

# 로컬 모듈 (lawfirm_langgraph 우선)
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.utils.config import Config
```

## 3. 클래스 구조 규칙
```python
class ServiceName:
    """서비스 클래스에 대한 간단한 설명"""
    
    def __init__(self, config: Config):
        """초기화 메서드"""
        self.config = config
        self._initialize_components()
    
    def _initialize_components(self):
        """내부 컴포넌트 초기화 (private 메서드)"""
        pass
    
    def public_method(self, param: str) -> str:
        """공개 메서드에 대한 설명"""
        pass
```

## 4. 에러 처리 규칙
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_operation(self, data: str) -> Optional[str]:
    """안전한 연산 수행"""
    try:
        result = self._process_data(data)
        return result
    except ValueError as e:
        logger.error(f"Value error in safe_operation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in safe_operation: {e}")
        raise
```

## 5. Linter 오류 방지 규칙 (CRITICAL)

**파이썬 개발 시 linter 오류가 발생하지 않도록 코드를 작성해야 합니다.**

### 원칙
1. **Linter 오류 제로 목표**
   - 코드 작성 시 linter 오류가 발생하지 않도록 주의
   - 코드 수정 후 linter 검사를 실행하여 오류 확인
   - 모든 linter 오류는 반드시 수정해야 함

2. **주요 Linter 오류 유형 및 해결 방법**

   **a. 사용되지 않는 변수 (Unused Variable)**
   ```python
   # ❌ 나쁜 예: 사용되지 않는 변수
   result = asyncio.run(run_query_test(query))
   return 0  # result 변수가 사용되지 않음
   
   # ✅ 좋은 예 1: 변수 제거
   asyncio.run(run_query_test(query))
   return 0
   
   # ✅ 좋은 예 2: 변수 사용
   result = asyncio.run(run_query_test(query))
   if result:
       logger.info(f"Test completed with result: {result}")
   return 0
   
   # ✅ 좋은 예 3: 의도적으로 사용하지 않는 경우 언더스코어 사용
   _result = asyncio.run(run_query_test(query))  # linter 경고 무시
   return 0
   ```

   **b. 플레이스홀더가 없는 f-string**
   ```python
   # ❌ 나쁜 예: 플레이스홀더가 없는 f-string
   logger.warning(f"\n⚠️  related_questions가 없습니다!")
   logger.info(f"   전체 구조:")
   
   # ✅ 좋은 예: 일반 문자열 사용
   logger.warning("\n⚠️  related_questions가 없습니다!")
   logger.info("   전체 구조:")
   
   # ✅ 좋은 예: 플레이스홀더가 있는 경우만 f-string 사용
   logger.warning(f"\n⚠️  related_questions가 없습니다! (count: {len(questions)})")
   logger.info(f"   전체 구조: {structure}")
   ```

   **c. 사용되지 않는 import**
   ```python
   # ❌ 나쁜 예: 사용되지 않는 import
   import os
   import sys
   from typing import List, Dict, Optional
   
   def process_data(data: str) -> str:
       return data.upper()  # os, sys, List, Dict, Optional 미사용
   
   # ✅ 좋은 예: 사용하는 것만 import
   def process_data(data: str) -> str:
       return data.upper()
   ```

   **d. 타입 힌트 누락**
   ```python
   # ❌ 나쁜 예: 타입 힌트 누락
   def process_data(data):
       return data.upper()
   
   # ✅ 좋은 예: 타입 힌트 포함
   def process_data(data: str) -> str:
       return data.upper()
   ```

   **e. 너무 긴 줄 (Line too long)**
   ```python
   # ❌ 나쁜 예: 100자 이상의 긴 줄
   result = some_very_long_function_name(param1, param2, param3, param4, param5, param6, param7, param8)
   
   # ✅ 좋은 예: 여러 줄로 분리
   result = some_very_long_function_name(
       param1, param2, param3, param4,
       param5, param6, param7, param8
   )
   ```

   **f. 빈 except 절**
   ```python
   # ❌ 나쁜 예: 빈 except 절
   try:
       process_data()
   except:
       pass
   
   # ✅ 좋은 예: 구체적인 예외 처리
   try:
       process_data()
   except ValueError as e:
       logger.error(f"Value error: {e}")
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
   ```

3. **Linter 검사 실행 방법**
   ```bash
   # flake8 검사
   flake8 source/ --max-line-length=100 --extend-ignore=E203,W503
   
   # pylint 검사
   pylint source/
   
   # ruff 검사 (권장: 빠르고 현대적)
   ruff check source/
   ruff check source/ --fix  # 자동 수정
   
   # mypy 타입 체크
   mypy source/
   ```

4. **코드 작성 전 체크리스트**
   - [ ] 사용하지 않는 변수나 import가 없는가?
   - [ ] 플레이스홀더가 없는 f-string을 사용하지 않았는가?
   - [ ] 모든 함수에 타입 힌트가 있는가?
   - [ ] 한 줄이 100자를 초과하지 않는가?
   - [ ] 빈 except 절을 사용하지 않았는가?
   - [ ] 코드 수정 후 linter 검사를 실행했는가?

5. **자동 수정 도구 활용**
   ```bash
   # ruff로 자동 수정 가능한 오류 수정
   ruff check source/ --fix
   
   # black으로 코드 포맷팅
   black source/
   
   # isort로 import 정렬
   isort source/
   ```

6. **예외 처리 (Linter 오류 무시)**
   ```python
   # 필요한 경우에만 사용 (최소화)
   
   # 특정 줄의 특정 오류 무시
   result = asyncio.run(run_query_test(query))  # noqa: F841
   
   # 특정 블록의 특정 오류 무시
   # pylint: disable=unused-variable
   result = asyncio.run(run_query_test(query))
   # pylint: enable=unused-variable
   
   # 파일 전체의 특정 오류 무시 (비권장)
   # flake8: noqa
   ```

7. **금지 사항**
   - Linter 오류를 무시하고 코드를 커밋하는 것 금지
   - `# noqa` 주석을 남용하는 것 금지 (정말 필요한 경우에만 사용)
   - Linter 검사를 건너뛰는 것 금지
   - 사용되지 않는 코드를 남겨두는 것 금지

8. **권장 Linter 설정**
   - **ruff**: 빠르고 현대적인 linter (권장)
   - **flake8**: 전통적인 linter
   - **pylint**: 상세한 코드 분석
   - **mypy**: 정적 타입 체크
   - **black**: 코드 포맷터
   - **isort**: import 정렬 도구

## 6. 프론트엔드 ESLint / TypeScript / npm audit 규칙 (CRITICAL)

**`.github/workflows/security-check.yml`에서 실행되는 `npm run lint`, `npm run type-check`, `npm audit --audit-level=moderate` 기준으로, 프론트엔드 코드 변경 시 항상 이 검사를 통과하도록 코드를 작성해야 합니다.**

1. **ESLint 규칙 준수**
   - 프론트엔드 코드 작성/수정 시 `npm run lint` 기준으로 경고/오류가 발생하지 않도록 작성합니다.
   - 불가피하게 규칙을 무시해야 할 경우:
     - 왜 필요한지 코드 근처에 아주 짧게 주석으로 남깁니다.
     - `eslint-disable` 류의 주석은 최소 범위(한 줄)로만 사용합니다.
     - PR 단위에서 팀과 상의 없이 전역/파일 단위 disable은 금지합니다.

2. **TypeScript 타입 체크 통과**
   - `npm run type-check`를 기준으로 모든 새로운/수정된 컴포넌트에 타입 오류가 없어야 합니다.
   - `any` 사용은 마지막 수단으로만 허용하며, 가능한 구체적인 타입 또는 제네릭 사용을 우선합니다.
   - 외부 라이브러리 타입이 애매한 경우:
     - 우선 `@types/*` 패키지 존재 여부를 확인합니다.
     - 불가피하게 `any`를 사용할 때는 타입 보완 계획을 간단히 주석으로 남깁니다.

3. **npm audit (보안 점검) 고려**
   - 의존성 추가/업데이트 전후로 `npm audit --audit-level=moderate` 결과를 확인합니다.
   - 새로운 취약점이 발견되면:
     - 가능한 경우 `npm update` / 패치 버전 업데이트로 해결을 우선 시도합니다.
     - 해결이 어렵거나 상위 의존성 문제인 경우, 이슈/PR 설명에 취약점 ID와 상황을 간단히 기록합니다.
   - 불필요한 패키지(사용하지 않는 라이브러리)는 즉시 제거하여 공격 면을 최소화합니다.

4. **로컬 검증 체크리스트 (프론트엔드 작업 후)**
   - [ ] `cd frontend` 후 `npm run lint`를 실행했고, 오류가 없는가?
   - [ ] `npm run type-check`를 실행했고, 타입 오류가 없는가?
   - [ ] 의존성을 변경했다면 `npm audit --audit-level=moderate` 결과를 확인했는가?
   - [ ] ESLint 규칙 무시 주석(`eslint-disable`)을 새로 추가했다면, 정말 필요한 최소 범위인가?

