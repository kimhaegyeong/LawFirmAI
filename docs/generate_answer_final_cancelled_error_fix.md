# generate_answer_final 노드 CancelledError 처리 개선

## 문제 상황

`generate_answer_final` 노드 실행 중 `CancelledError`가 발생하여 워크플로우가 예기치 않게 중단되는 문제가 발생했습니다.

**에러 스택 트레이스**:
```
asyncio.exceptions.CancelledError
  File "langgraph/_internal/_runnable.py", line 839, in astream
  File "langgraph/_internal/_runnable.py", line 904, in _consume_aiter
  File "langchain_core/tracers/event_stream.py", line 191, in tap_output_aiter
  File "langchain_core/utils/aiter.py", line 76, in anext_impl
  File "langchain_core/runnables/base.py", line 1578, in atransform
  File "langchain_core/runnables/base.py", line 1147, in astream
  File "langgraph/_internal/_runnable.py", line 473, in ainvoke
  File "langgraph/graph/_branch.py", line 189, in _aroute
  File "langgraph/_internal/_runnable.py", line 464, in ainvoke
  File "langchain_core/runnables/config.py", line 603, in run_in_executor
```

## 원인 분석

1. **비동기 실행 중 취소**: LangGraph가 노드를 비동기로 실행할 때 타임아웃이나 외부 요인으로 인해 취소될 수 있음
2. **단계별 처리 부재**: 노드 내부의 각 단계(`_restore_state_data_for_final`, `_validate_and_handle_regeneration`, `_handle_format_errors`, `_format_and_finalize`)에서 `CancelledError`를 개별적으로 처리하지 않음
3. **상태 손실**: 취소된 경우에도 현재까지 처리된 상태를 보존하지 않음

## 해결 방법

### 1. `generate_answer_final` 노드에 단계별 `CancelledError` 처리 추가

**위치**: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py` - 1605-1636줄

**개선 내용**:
- 각 단계별로 `CancelledError`를 개별적으로 처리
- 취소된 경우 기존 답변을 보존하고 상태를 안전하게 반환
- 최상위 `CancelledError` 처리에서 답변이 없으면 기본 메시지 설정

```python
def generate_answer_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
    """최종 검증 및 포맷팅 노드 - 검증과 포맷팅만 수행"""
    try:
        # ... 각 단계별로 try-except로 CancelledError 처리 ...
        
        try:
            self._restore_state_data_for_final(state)
        except asyncio.CancelledError:
            self.logger.warning("⚠️ [FINAL NODE] State restoration was cancelled.")
            raise
        
        try:
            quality_check_passed = self._validate_and_handle_regeneration(state)
        except asyncio.CancelledError:
            # 기존 답변 보존 및 반환
            ...
            return state
        
        # ... 다른 단계들도 동일하게 처리 ...
        
    except asyncio.CancelledError:
        # 최상위 CancelledError 처리
        # 답변이 있으면 보존, 없으면 기본 메시지 설정
        # 예외를 다시 발생시키지 않고 상태를 보존한 채로 반환
        ...
        return state
```

### 2. `_format_and_finalize` 메서드 개선

**위치**: `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py` - 1431-1458줄

**개선 내용**:
- `CancelledError` 발생 시 기본 포맷으로 처리 완료
- 예외를 다시 발생시키지 않고 기본 포맷으로 처리하여 노드가 정상적으로 완료되도록 함

## 개선 효과

1. **단계별 에러 처리**: 각 단계에서 `CancelledError`를 개별적으로 처리하여 더 세밀한 제어 가능
2. **상태 보존**: 취소된 경우에도 현재까지 처리된 상태를 보존하여 부분 결과를 반환 가능
3. **안정성 향상**: 취소 발생 시에도 노드가 정상적으로 완료되어 워크플로우가 계속 진행 가능
4. **로깅**: 각 단계별로 명확한 로그 메시지로 디버깅 용이

## 처리 전략

### CancelledError 처리 원칙

1. **상태 보존 우선**: 취소된 경우에도 현재까지의 상태를 보존
2. **기존 답변 활용**: 기존 답변이 있으면 보존하여 사용자에게 제공
3. **기본값 설정**: 답변이 없으면 기본 메시지 설정
4. **예외 전파 제어**: 노드 레벨에서는 예외를 다시 발생시키지 않고 상태를 보존한 채로 반환

### 단계별 처리

1. **State 복원 단계**: 취소 시 예외를 다시 발생시켜 상위에서 처리
2. **검증 단계**: 취소 시 기존 답변 보존 및 반환
3. **형식 오류 처리 단계**: 취소 시 기존 답변 보존 및 반환
4. **포맷팅 단계**: 취소 시 기본 포맷으로 처리 완료

## 테스트 방법

1. **타임아웃 시뮬레이션**: 긴 실행 시간이 필요한 쿼리로 테스트
2. **수동 취소**: 워크플로우 실행 중 `Ctrl+C`로 취소하여 `CancelledError` 처리 확인
3. **로그 확인**: 각 단계별로 취소 발생 시 경고 메시지가 로그에 기록되는지 확인
4. **상태 보존 확인**: 취소 후에도 기존 답변이 보존되는지 확인

## 관련 파일

- `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`: 메인 워크플로우 클래스
- `lawfirm_langgraph/core/workflow/workflow_service.py`: 워크플로우 서비스 (이미 `CancelledError` 처리 있음)

## 참고 사항

- `CancelledError`는 비동기 작업이 취소되었을 때 발생하는 예외입니다
- 노드 레벨에서는 예외를 다시 발생시키지 않고 상태를 보존한 채로 반환하여 워크플로우가 계속 진행될 수 있도록 합니다
- 각 단계별로 `CancelledError`를 처리하여 더 세밀한 제어가 가능합니다
- 기존 답변이 있으면 보존하여 사용자에게 제공하고, 없으면 기본 메시지를 설정합니다

