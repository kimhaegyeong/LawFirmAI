"""
스트리밍 관련 상수 정의
"""

class StreamingConstants:
    """스트리밍 관련 상수"""
    
    # 타겟 노드 목록
    TARGET_NODES = [
        "generate_answer_stream",
        "generate_answer_enhanced",
        "generate_and_validate_answer",
        "direct_answer"
    ]
    
    # 답변 생성 노드 목록
    ANSWER_GENERATION_NODES = [
        "generate_answer_stream",
        "generate_answer_enhanced",
        "generate_and_validate_answer"
    ]
    
    # 답변 생성 완료 노드 목록
    ANSWER_COMPLETION_NODES = [
        "generate_answer_stream",
        "generate_and_validate_answer"
    ]
    
    # 디버그 로그 제한
    MAX_DEBUG_LOGS = 10
    MAX_DETAILED_LOGS = 5
    MAX_SKIP_LOGS = 3  # 건너뛰기 로그 최대 출력 횟수
    
    # 타임아웃 설정
    STATE_TIMEOUT = 2.0
    CALLBACK_QUEUE_TIMEOUT = 0.1
    STATE_RETRY_DELAY = 0.2
    CALLBACK_MONITORING_INTERVAL = 0.01
    
    # 이벤트 타입
    EVENT_TYPE_STREAM = "stream"
    EVENT_TYPE_FINAL = "final"
    EVENT_TYPE_ERROR = "error"
    EVENT_TYPE_PROGRESS = "progress"
    EVENT_TYPE_VALIDATION = "validation"
    EVENT_TYPE_DONE = "done"
    
    # 콜백 큐 타입
    CALLBACK_CHUNK_TYPE = "chunk"

