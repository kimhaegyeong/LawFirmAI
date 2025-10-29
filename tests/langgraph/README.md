# LangGraph 모니터링 전환 테스트

LangSmith와 Langfuse를 번갈아가며 사용할 수 있는 테스트 시스템입니다.

## 개요

이 디렉토리는 LangGraph 워크플로우의 모니터링 도구 전환 기능을 테스트하는 파일들을 포함합니다. 
모니터링 모드별로 워크플로우를 전환하여 테스트할 수 있으며, 환경변수 격리와 워크플로우 캐싱을 지원합니다.

## 구조

```
tests/langgraph/
├── __init__.py
├── README.md                     # 이 문서
├── ENV_PROFILES_EXAMPLE.md       # 환경변수 프로필 예시
├── monitoring_switch.py          # 모니터링 전환 유틸리티
├── test_monitoring_switch_basic.py # 모니터링 전환 기본 테스트
├── test_profile_loading.py       # 프로필 로딩 테스트
├── test_with_monitoring_switch.py # 통합 테스트 스크립트
└── fixtures/
    ├── __init__.py
    ├── monitoring_configs.py     # 모니터링 설정 픽스처
    └── workflow_factory.py       # 워크플로우 팩토리
```

## 관련 테스트 파일

이 디렉토리 외에도 루트 레벨에 LangGraph 관련 테스트가 있습니다:

- `test_langgraph.py` - 기본 LangGraph 워크플로우 테스트
- `test_langgraph_state_optimization.py` - State 최적화 테스트
- `test_langgraph_multi_turn.py` - 멀티턴 대화 테스트
- `test_all_state_systems.py` - State 시스템 통합 테스트
- `test_core_state_systems.py` - Core State 시스템 테스트
- `test_state_reduction_performance.py` - State Reduction 성능 테스트

## 사용 방법

### 1. 단일 모드 테스트

```python
from tests.langgraph.monitoring_switch import MonitoringMode, MonitoringSwitch
from tests.langgraph.fixtures.workflow_factory import WorkflowFactory

# LangSmith 모드로 테스트
with MonitoringSwitch.set_mode(MonitoringMode.LANGSMITH):
    service = WorkflowFactory.get_workflow(MonitoringMode.LANGSMITH)
    result = await service.process_query("테스트 쿼리")
```

### 2. 번갈아가며 테스트

```bash
# 전체 테스트 실행 (모든 모드)
python tests/langgraph/test_with_monitoring_switch.py

# 특정 모드만 테스트
python tests/langgraph/test_with_monitoring_switch.py langsmith
python tests/langgraph/test_with_monitoring_switch.py langfuse
python tests/langgraph/test_with_monitoring_switch.py both
python tests/langgraph/test_with_monitoring_switch.py none
```

### 3. pytest 사용

```bash
# 모든 모드로 테스트 (파라미터화)
pytest tests/langgraph/ -v

# 특정 모드 픽스처 사용
pytest tests/langgraph/ -v -k "langsmith"
```

## 모니터링 모드

- **LANGSMITH**: LangSmith만 활성화
- **LANGFUSE**: Langfuse만 활성화  
- **BOTH**: 두 도구 모두 활성화
- **NONE**: 모니터링 비활성화

## 환경변수 프로필

`.env.profiles/` 디렉토리에 각 모드별 환경변수 프로필을 생성할 수 있습니다:

- `langsmith.env` - LangSmith 전용 설정
- `langfuse.env` - Langfuse 전용 설정
- `both.env` - 둘 다 사용 설정
- `none.env` - 모니터링 없음

프로필 파일 예시는 `.env.profiles/README.md`를 참조하세요.

## 주요 기능

### MonitoringSwitch

환경변수를 관리하고 모니터링 모드를 전환합니다.

```python
with MonitoringSwitch.set_mode(MonitoringMode.LANGSMITH):
    # 이 블록 내에서 LangSmith 모드로 실행
    pass
```

### WorkflowFactory

모니터링 모드별 워크플로우 인스턴스를 생성하고 캐싱합니다.

```python
# LangSmith 모드 워크플로우 생성
service = WorkflowFactory.get_workflow(MonitoringMode.LANGSMITH)

# 캐시 재생성
service = WorkflowFactory.get_workflow(MonitoringMode.LANGSMITH, trompu_recreate=True)
```

### 검증

```python
# 현재 모드 확인
current_mode = MonitoringSwitch.get_current_mode()

# 서비스 모드 검증
verification = MonitoringSwitch.verify_mode(service, MonitoringMode.LANGSMITH)
```

## 테스트 실행 예제

### 기본 테스트 실행

```bash
# 모든 모니터링 모드로 테스트
python tests/langgraph/test_with_monitoring_switch.py

# 특정 모드만 테스트
python tests/langgraph/test_with_monitoring_switch.py langsmith
python tests/langgraph/test_with_monitoring_switch.py langfuse

# 기본 기능 테스트
pytest tests/langgraph/test_monitoring_switch_basic.py -v

# 프로필 로딩 테스트
pytest tests/langgraph/test_profile_loading.py -v
```

### pytest로 실행

```bash
# LangGraph 디렉토리 전체 테스트
pytest tests/langgraph/ -v

# 특정 모드 필터링
pytest tests/langgraph/ -v -k "langsmith"
pytest tests/langgraph/ -v -k "langfuse"
```

## 주요 클래스 및 함수

### MonitoringMode

모니터링 모드 Enum:
- `LANGSMITH` - LangSmith만 활성화
- `LANGFUSE` - Langfuse만 활성화
- `BOTH` - 두 도구 모두 활성화
- `NONE` - 모니터링 비활성화

### MonitoringSwitch

모니터링 모드 전환을 관리하는 클래스:

- `set_mode(mode)` - 모니터링 모드 설정 (컨텍스트 매니저)
- `get_current_mode()` - 현재 모드 확인
- `verify_mode(service, mode)` - 서비스 모드 검증
- `load_profile(profile_name)` - 환경변수 프로필 로드

### WorkflowFactory

워크플로우 인스턴스를 생성하고 관리하는 팩토리:

- `get_workflow(mode, force_recreate=False)` - 워크플로우 인스턴스 가져오기
- `clear_cache()` - 캐시 정리

## 주의사항

1. **LangSmith 재컴파일**: LangSmith는 워크플로우 컴파일 시점에 결정되므로, 모드 변경 시 워크플로우를 재생성해야 합니다. `WorkflowFactory`가 이를 자동으로 처리합니다.

2. **환경변수 격리**: 각 테스트는 컨텍스트 매니저로 환경변수가 격리되므로, 테스트 간 충돌이 없습니다.

3. **리소스 관리**: 테스트 후 캐시를 정리하려면 `WorkflowFactory.clear_cache()`를 호출하세요.

4. **프로필 파일**: `.env.profiles/` 디렉토리에 모드별 환경변수 프로필을 생성하면 자동으로 로드됩니다.

## 문제 해결

### 환경변수 오류

프로필 파일이 제대로 로드되지 않으면 `ENV_PROFILES_EXAMPLE.md`를 참조하여 올바른 형식으로 작성했는지 확인하세요.

### 워크플로우 캐시 문제

모드 변경 후에도 이전 모드의 워크플로우가 사용되는 경우:
```python
WorkflowFactory.clear_cache()
service = WorkflowFactory.get_workflow(mode, force_recreate=True)
```

### 모드 검증 실패

`verify_mode()`가 실패하는 경우, 환경변수가 올바르게 설정되었는지 확인하세요:
```python
from tests.langgraph.monitoring_switch import MonitoringSwitch
print(MonitoringSwitch.get_current_mode())
```

## 관련 문서

- [메인 테스트 가이드](../README.md)
- [프로젝트 메인 README](../../README.md)

## 업데이트 이력

- **2025-01**: 문서 업데이트 및 구조 정리
