"""
스트리밍 설정
"""
from dataclasses import dataclass
from typing import FrozenSet
import re
import os


@dataclass
class StreamConfig:
    """스트리밍 설정"""
    debug_stream: bool = False
    max_event_history: int = 100
    max_output_tokens: int = 8192
    chunk_size: int = 2000
    
    # JSON 감지 패턴
    json_pattern: re.Pattern = re.compile(r'^\s*"[^"]+"\s*:\s*')
    json_keywords: FrozenSet[str] = frozenset([
        '"complexity"', '"confidence"', '"reasoning"', '"core_keywords"',
        '"query_intent"', '"is_valid"', '"quality_score"', '"final_score"',
        '"score"', '"issues"', '"strengths"', '"recommendations"',
        '"needs_improvement"', '"improvement_instructions"', '"preserve_content"',
        '"focus_areas"', '"meets_quality_threshold"', '"summary"'
    ])
    
    # 관련 이벤트 타입
    relevant_event_types: FrozenSet[str] = frozenset([
        "on_llm_stream", "on_chat_model_stream", "on_chain_stream",
        "on_chain_start", "on_chain_end", "on_llm_end", "on_chat_model_end"
    ])
    
    # 답변 생성 노드 이름
    answer_generation_nodes: FrozenSet[str] = frozenset([
        "generate_answer_enhanced",
        "generate_and_validate_answer"
    ])
    
    @classmethod
    def from_env(cls) -> "StreamConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            debug_stream=os.getenv("DEBUG_STREAM", "false").lower() == "true",
            max_event_history=int(os.getenv("MAX_EVENT_HISTORY", "100")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "2000"))
        )

