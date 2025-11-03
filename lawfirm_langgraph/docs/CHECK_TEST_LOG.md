# 테스트 로그 확인 가이드

## 테스트 실행 방법

PowerShell에서 다음 명령어를 실행하세요:

```powershell
cd D:\project\LawFirmAI\LawFirmAI\lawfirm_langgraph
.venv\Scripts\python.exe run_test_with_log.py
```

또는 직접 종합 테스트 실행:

```powershell
.venv\Scripts\python.exe test_workflow.py > test_workflow_log.txt 2>&1
```

## 로그 파일 확인

테스트 실행 후 다음 파일을 확인하세요:

1. **test_workflow_log.txt** - 테스트 실행 로그

PowerShell에서 로그 확인:

```powershell
Get-Content test_workflow_log.txt
```

또는:

```powershell
type test_workflow_log.txt
```

## 예상 로그 내용

성공적인 테스트 실행 시 로그 내용:

```
============================================================
LangGraph 종합 테스트 실행 로그
실행 시간: 2025-01-XX XX:XX:XX
============================================================

============================================================
LangGraph 워크플로우 종합 테스트
============================================================

1. Import 테스트
------------------------------------------------------------
  - LangGraph 모듈...
    ✓ LangGraph 버전: 1.0.x
  - 설정 모듈...
    ✓ Config 모듈 로드 성공
  - 워크플로우 모듈...
    ✓ 워크플로우 모듈 로드 성공
  - Graph export 모듈...
    ✓ Graph export 모듈 로드 성공

2. 설정 로딩 테스트
------------------------------------------------------------
  - 환경 변수에서 설정 로드...
    ✓ LangGraph 활성화: True
    ✓ LLM 제공자: google
    ✓ Google 모델: gemini-2.5-flash-lite
    ✓ 최대 반복 횟수: 10
    ✓ 재귀 제한: 25
  - 설정 유효성 검사...
    ✓ 설정 유효성 검사 통과

3. 그래프 생성 테스트
------------------------------------------------------------
  - create_graph() 함수 실행...
    ✓ 그래프 생성 성공: StateGraph
  - graph 인스턴스 확인...
    ✓ 그래프 인스턴스 생성: StateGraph
    ✓ 그래프 노드 수: XX
    ✓ 그래프 엣지 수: XX

4. 앱 생성 테스트
------------------------------------------------------------
  - create_app() 함수 실행...
    ✓ 앱 생성 성공: CompiledGraph
  - app 인스턴스 확인...
    ✓ 앱 인스턴스 생성: CompiledGraph
    ✓ invoke 메서드 존재
    ✓ stream 메서드 존재
    ✓ get_graph 메서드 존재

5. 워크플로우 실행 테스트 (간단한 입력)
------------------------------------------------------------
  - 앱 인스턴스 생성...
  - 워크플로우 실행 (간단한 질문)...
    입력: 안녕하세요. 간단한 테스트 질문입니다.
    ✓ 워크플로우 실행 완료
    ✓ 결과 타입: dict
    ✓ 메시지 수: 2
    ✓ 응답 길이: XXX 문자
    ✓ 응답 미리보기: ...

테스트 결과 요약
============================================================
  ✓ PASS: Import 테스트
  ✓ PASS: 설정 로딩
  ✓ PASS: 그래프 생성
  ✓ PASS: 앱 생성
  ✓ PASS: langgraph.json 설정
  ✓ PASS: LangGraph CLI
  ✓ PASS: 워크플로우 실행

총 7/7 테스트 통과

✓ 모든 테스트 통과! LangGraph가 정상적으로 동작합니다.

다음 단계:
  1. 'langgraph dev' 명령어로 LangGraph Studio 실행
  2. 브라우저에서 http://localhost:8123 접속

============================================================
종료 코드: 0
종료 시간: 2025-01-XX XX:XX:XX
============================================================
```

## 문제 해결

### Import 오류

```
✗ Import 실패: No module named 'xxx'
```

**해결 방법:**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 설정 오류

```
✗ 설정 로딩 실패: ...
```

**해결 방법:**
- 상위 프로젝트의 `.env` 파일 확인
- 필요한 환경 변수 설정 확인

### 워크플로우 실행 오류

```
⚠ 워크플로우 실행 실패: ...
```

**참고:**
- 이것은 LLM API 키 설정 문제일 수 있습니다
- Graph와 App 생성이 성공했다면 정상입니다
- LangGraph Studio를 사용할 수 있습니다

## 로그 파일 위치

로그 파일은 다음 위치에 생성됩니다:

- `lawfirm_langgraph/test_workflow_log.txt`

## 다음 단계

모든 테스트가 통과하면:

```powershell
langgraph dev
```

브라우저에서 `http://localhost:8123`에 접속하여 LangGraph Studio를 사용할 수 있습니다.
