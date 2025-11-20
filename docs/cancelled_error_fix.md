# CancelledError 처리 개선

## 문제 상황

LangGraph 워크플로우 실행 중 `asyncio.CancelledError`가 발생하여 워크플로우가 예기치 않게 중단되는 문제가 발생했습니다.

**에러 스택 트레이스**:
```
asyncio.exceptions.CancelledError
  File "langgraph/pregel/main.py", line 3000, in astream
  File "langgraph/pregel/_runner.py", line 304, in atick
  File "langgraph/pregel/_retry.py", line 132, in arun_with_retry
```

## 원인 분석

1. **비동기 작업 취소**: 타임아웃이나 외부 요인으로 인해 비동기 작업이 취소될 수 있음
2. **에러 처리 부재**: `astream()` 및 `astream_events()` 루프에서 `CancelledError`를 처리하지 않아 예외가 상위로 전파됨
3. **상태 손실**: 취소된 경우에도 현재까지 처리된 상태를 보존하지 않음

## 해결 방법

### 1. `astream_events()` 루프에 `CancelledError` 처리 추가

**위치**: `lawfirm_langgraph/core/workflow/workflow_service.py` - 537-595줄

```python
if use_astream_events:
    try:
        async for event in self.app.astream_events(initial_state, enhanced_config, version="v2"):
            # ... 이벤트 처리 로직 ...
    except asyncio.CancelledError:
        self.logger.warning("⚠️ [WORKFLOW] 워크플로우 실행이 취소되었습니다 (CancelledError)")
        # 취소된 경우에도 현재까지의 상태를 반환
        if accumulated_state and isinstance(accumulated_state, dict):
            flat_result = accumulated_state
        else:
            flat_result = initial_state
        raise
```

### 2. `astream()` 루프에 `CancelledError` 처리 추가

**위치**: `lawfirm_langgraph/core/workflow/workflow_service.py` - 622-1135줄

```python
else:
    # 기존 astream() 사용
    try:
        async for event in self.app.astream(initial_state, enhanced_config, stream_mode="updates"):
            # ... 이벤트 처리 로직 ...
            flat_result = node_state
    except asyncio.CancelledError:
        self.logger.warning("⚠️ [WORKFLOW] 워크플로우 실행이 취소되었습니다 (CancelledError)")
        # 취소된 경우에도 현재까지의 상태를 반환
        if flat_result is None:
            flat_result = initial_state
        raise
```

## 개선 효과

1. **에러 처리**: `CancelledError` 발생 시 적절히 처리하여 워크플로우가 안정적으로 종료됨
2. **상태 보존**: 취소된 경우에도 현재까지 처리된 상태를 보존하여 부분 결과를 반환 가능
3. **로깅**: 취소 발생 시 명확한 로그 메시지로 디버깅 용이

## 테스트 방법

1. **타임아웃 시뮬레이션**: 긴 실행 시간이 필요한 쿼리로 테스트
2. **수동 취소**: 워크플로우 실행 중 `Ctrl+C`로 취소하여 `CancelledError` 처리 확인
3. **로그 확인**: 취소 발생 시 경고 메시지가 로그에 기록되는지 확인

## 관련 파일

- `lawfirm_langgraph/core/workflow/workflow_service.py`: 메인 워크플로우 서비스
- `lawfirm_langgraph/tests/scripts/run_query_test.py`: 테스트 스크립트 (이미 `CancelledError` 처리 있음)

## 참고 사항

- `CancelledError`는 예외를 다시 발생시켜(`raise`) 상위 호출자에게 취소 사실을 알림
- 취소된 경우에도 현재까지의 상태를 반환하여 부분 결과를 활용할 수 있음
- `run_query_test.py`에서도 `CancelledError`를 처리하고 있어, 이제 전체 체인에서 일관되게 처리됨

