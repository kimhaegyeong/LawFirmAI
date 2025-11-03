# LangGraph 마이그레이션 예제

이 디렉토리에는 LangGraph 프로젝트의 마이그레이션 및 사용 예제가 포함되어 있습니다.

## 파일 목록

### migration_example.py
State Reduction 기능을 기존 코드에 적용하는 방법을 보여주는 예제입니다.

주요 내용:
- 데코레이터를 사용한 State Reduction 적용
- state_adapter를 사용한 자동 변환
- 실제 사용 예제 코드

## 사용 방법

프로젝트 루트에서 실행:

```bash
cd lawfirm_langgraph
python docs/examples/migration_example.py
```

또는 Python 인터프리터에서:

```python
from docs.examples.migration_example import classify_query_example
```

## 주의사항

이 예제 파일들은 참고용이며, 실제 프로덕션 코드에 직접 사용하기 전에 프로젝트 구조에 맞게 수정해야 합니다.
