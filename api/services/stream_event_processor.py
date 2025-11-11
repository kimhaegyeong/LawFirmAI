"""
스트리밍 이벤트 처리기
LangGraph 이벤트를 처리하고 토큰을 추출합니다.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
import re
import logging

from .stream_config import StreamConfig

logger = logging.getLogger(__name__)


class StreamEventProcessor:
    """스트리밍 이벤트 처리기"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.json_pattern = config.json_pattern
        self.json_keywords = config.json_keywords
        
        # 상태 추적
        self.full_answer = ""
        self.answer_found = False
        self.tokens_received = 0
        self.answer_generation_started = False
        self.json_output_detected = False
        self.last_node_name = None
    
    def reset(self):
        """상태 초기화"""
        self.full_answer = ""
        self.answer_found = False
        self.tokens_received = 0
        self.answer_generation_started = False
        self.json_output_detected = False
        self.last_node_name = None
    
    def is_answer_node(self, event: Dict[str, Any]) -> bool:
        """답변 생성 노드인지 확인"""
        event_name = event.get("name", "")
        event_tags = event.get("tags", [])
        event_parent = event.get("parent", {})
        
        # 방법 1: 이벤트 이름으로 직접 판단
        if "generate_answer" in event_name.lower() or \
           "generate_and_validate" in event_name.lower() or \
           event_name in ["generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"]:
            return True
        
        # 방법 2: 상위 노드가 답변 생성 노드인지 확인
        parent_node_name = None
        if isinstance(event_parent, dict):
            parent_node_name = event_parent.get("name", "")
        elif isinstance(event_tags, list):
            for tag in event_tags:
                if isinstance(tag, str) and ("generate_answer" in tag.lower() or "generate_and_validate" in tag.lower()):
                    parent_node_name = tag
                    break
        
        if parent_node_name and (
            "generate_answer" in parent_node_name.lower() or 
            "generate_and_validate" in parent_node_name.lower() or
            parent_node_name in ["generate_answer_enhanced", "generate_and_validate_answer"]
        ):
            return True
        
        # 방법 3: ChatGoogleGenerativeAI인 경우
        if event_name == "ChatGoogleGenerativeAI" and self.answer_generation_started:
            if self.last_node_name == "generate_and_validate_answer":
                return True
            elif self.last_node_name == "generate_answer_enhanced":
                return True
        
        return False
    
    def is_json_output(self, chunk: str) -> bool:
        """JSON 형식 출력인지 확인"""
        chunk_stripped = chunk.strip()
        
        # 방법 1: 청크 시작 부분이 JSON 형식인지 확인
        if chunk_stripped.startswith("{") or chunk.startswith("{"):
            return True
        elif chunk_stripped.startswith("```json") or chunk.startswith("```json"):
            return True
        elif chunk_stripped.startswith("```") and "json" in chunk_stripped[:20].lower():
            return True
        elif chunk_stripped.startswith("```") or chunk.startswith("```"):
            return True
        elif "```json" in chunk or "``` json" in chunk:
            return True
        
        # 방법 2: 누적된 답변과 현재 청크를 합쳐서 JSON 형식인지 확인
        if self.full_answer:
            combined_text = (self.full_answer + chunk).strip()
            if combined_text.startswith("{") or combined_text.startswith("```json"):
                return True
            elif combined_text.startswith("```") and "json" in combined_text[:20].lower():
                return True
            elif combined_text.startswith("```"):
                return True
            elif "```json" in combined_text or "``` json" in combined_text:
                return True
        elif chunk_stripped.startswith("{") or chunk.startswith("{") or chunk_stripped.startswith("```") or chunk.startswith("```"):
            return True
        
        # 방법 3: JSON 키워드 패턴 감지
        if any(keyword in chunk for keyword in self.json_keywords):
            return True
        elif self.full_answer:
            combined_text = self.full_answer + chunk
            if any(keyword in combined_text for keyword in self.json_keywords):
                return True
        
        # 방법 4: JSON 구조 패턴 감지
        if self.json_pattern.match(chunk_stripped) or self.json_pattern.match(chunk):
            return True
        
        return False
    
    def extract_chunk_from_chain_stream(self, event_data: Dict[str, Any]) -> Optional[str]:
        """on_chain_stream 이벤트에서 청크 추출"""
        if not isinstance(event_data, dict):
            return None
        
        chain_output = event_data.get("chunk") or event_data.get("output")
        if chain_output is None:
            # 대체 경로 1: event_data에서 직접 추출
            text_content = event_data.get("text") or event_data.get("content")
            if text_content and isinstance(text_content, str):
                if len(text_content) > len(self.full_answer):
                    new_part = text_content[len(self.full_answer):]
                    self.full_answer = text_content
                    return new_part
            return None
        
        # 체인 출력이 딕셔너리인 경우
        if isinstance(chain_output, dict):
            answer_group = chain_output.get("answer", {})
            if isinstance(answer_group, dict):
                full_answer_from_event = answer_group.get("answer", "") or answer_group.get("content", "")
            elif isinstance(answer_group, str):
                full_answer_from_event = answer_group
            else:
                full_answer_from_event = ""
            
            if not full_answer_from_event:
                full_answer_from_event = chain_output.get("answer", "") or chain_output.get("content", "")
            
            if full_answer_from_event and isinstance(full_answer_from_event, str):
                if len(full_answer_from_event) > len(self.full_answer):
                    new_part = full_answer_from_event[len(self.full_answer):]
                    self.full_answer = full_answer_from_event
                    return new_part
        
        # 체인 출력이 문자열인 경우
        elif isinstance(chain_output, str):
            if len(chain_output) > len(self.full_answer):
                new_part = chain_output[len(self.full_answer):]
                self.full_answer = chain_output
                return new_part
        
        # 체인 출력이 객체인 경우 content 속성 확인
        elif hasattr(chain_output, "content"):
            content = chain_output.content
            if isinstance(content, str):
                if len(content) > len(self.full_answer):
                    new_part = content[len(self.full_answer):]
                    self.full_answer = content
                    return new_part
            elif isinstance(content, dict):
                answer_text = content.get("answer", "") or content.get("content", "")
                if answer_text and isinstance(answer_text, str):
                    if len(answer_text) > len(self.full_answer):
                        new_part = answer_text[len(self.full_answer):]
                        self.full_answer = answer_text
                        return new_part
        
        return None
    
    def extract_chunk_from_llm_stream(self, event_data: Dict[str, Any]) -> Optional[str]:
        """on_llm_stream 또는 on_chat_model_stream 이벤트에서 청크 추출"""
        if not isinstance(event_data, dict):
            return None
        
        chunk_obj = event_data.get("chunk")
        if chunk_obj is None:
            chunk = event_data.get("text") or event_data.get("content")
        else:
            # AIMessageChunk 객체 처리
            if hasattr(chunk_obj, "content"):
                content = chunk_obj.content
                if isinstance(content, str):
                    chunk = content
                elif isinstance(content, list) and len(content) > 0:
                    chunk = content[0] if isinstance(content[0], str) else str(content[0])
                else:
                    chunk = str(content)
            elif isinstance(chunk_obj, str):
                chunk = chunk_obj
            elif hasattr(chunk_obj, "text"):
                chunk = chunk_obj.text
            elif hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
                try:
                    content = getattr(chunk_obj, "content", None)
                    if isinstance(content, str):
                        chunk = content
                    elif isinstance(content, list) and len(content) > 0:
                        chunk = content[0] if isinstance(content[0], str) else str(content[0])
                    elif content is not None:
                        chunk = str(content)
                    else:
                        chunk = None
                except Exception:
                    chunk = None
            else:
                chunk = None
        
        # delta 형식 (LangGraph v2)
        if not chunk and "delta" in event_data:
            delta = event_data["delta"]
            if isinstance(delta, dict):
                chunk = delta.get("content") or delta.get("text")
            elif isinstance(delta, str):
                chunk = delta
        
        return chunk
    
    def process_stream_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """스트리밍 이벤트 처리 및 스트림 이벤트 생성"""
        event_type = event.get("event", "")
        event_name = event.get("name", "")
        
        # on_chain_start: 답변 생성 노드 시작 감지
        if event_type == "on_chain_start":
            node_name = event_name
            if node_name in self.config.answer_generation_nodes:
                self.answer_generation_started = True
                self.json_output_detected = False
                self.last_node_name = node_name
                
                return {
                    "type": "progress",
                    "message": "답변 생성 중...",
                    "node_name": node_name,
                    "timestamp": datetime.now().isoformat()
                }
        
        # on_chain_end: 답변 생성 노드 완료
        elif event_type == "on_chain_end":
            node_name = event_name
            if node_name == "generate_and_validate_answer":
                self.answer_generation_started = False
                # 최종 답변 확인 로직은 별도로 처리
        
        # LLM 스트리밍 이벤트 처리
        elif event_type in ["on_llm_stream", "on_chat_model_stream", "on_chain_stream"]:
            if not self.answer_generation_started:
                return None
            
            if not self.is_answer_node(event):
                return None
            
            event_data = event.get("data", {})
            
            # 이벤트 타입별 청크 추출
            if event_type == "on_chain_stream":
                chunk = self.extract_chunk_from_chain_stream(event_data)
            else:
                chunk = self.extract_chunk_from_llm_stream(event_data)
            
            if not chunk or not isinstance(chunk, str):
                return None
            
            # JSON 형식 출력 필터링
            if self.is_json_output(chunk):
                self.json_output_detected = True
                if self.config.debug_stream:
                    logger.debug(f"JSON 형식 출력 감지 및 무시: {chunk[:100]}...")
                return None
            
            # JSON 출력이 아닌 실제 답변이 시작되면 플래그 리셋
            if not self.is_json_output(chunk) and self.json_output_detected and len(chunk.strip()) > 0:
                reset_keywords = frozenset(['"complexity"', '"confidence"', '"reasoning"'])
                if not any(keyword in chunk for keyword in reset_keywords):
                    self.json_output_detected = False
                    if self.config.debug_stream:
                        logger.debug("실제 답변이 시작되어 JSON 출력 플래그 리셋")
            
            # 토큰 전송
            if len(chunk) > 0:
                self.full_answer += chunk
                self.tokens_received += 1
                self.answer_found = True
                
                return {
                    "type": "stream",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat()
                }
        
        # on_llm_end 또는 on_chat_model_end: 누락된 부분 확인
        elif event_type in ["on_llm_end", "on_chat_model_end"]:
            event_data = event.get("data", {})
            if isinstance(event_data, dict):
                output = event_data.get("output")
                if output is not None:
                    final_answer = None
                    
                    if hasattr(output, "content"):
                        final_answer = output.content
                    elif isinstance(output, str):
                        final_answer = output
                    elif isinstance(output, dict):
                        final_answer = output.get("content") or output.get("text") or str(output)
                    else:
                        final_answer = str(output)
                    
                    if final_answer and isinstance(final_answer, str):
                        if len(final_answer) > len(self.full_answer):
                            missing_part = final_answer[len(self.full_answer):]
                            if missing_part:
                                self.full_answer = final_answer
                                return {
                                    "type": "stream",
                                    "content": missing_part,
                                    "timestamp": datetime.now().isoformat()
                                }
        
        # on_chain_end: generate_and_validate_answer 노드의 answer 필드 확인
        elif event_type == "on_chain_end":
            node_name = event_name
            if node_name == "generate_and_validate_answer":
                event_data = event.get("data", {})
                if isinstance(event_data, dict):
                    output = event_data.get("output")
                    if output is not None and isinstance(output, dict):
                        answer_text = output.get("answer", "")
                        
                        if not answer_text and "answer" in output:
                            answer_group = output.get("answer", {})
                            if isinstance(answer_group, dict):
                                answer_text = answer_group.get("answer", "")
                            elif isinstance(answer_group, str):
                                answer_text = answer_group
                        
                        if not answer_text and "common" in output:
                            common = output.get("common", {})
                            if isinstance(common, dict):
                                answer_text = common.get("answer", "")
                        
                        if answer_text and isinstance(answer_text, str) and len(answer_text) > 0:
                            answer_stripped = answer_text.strip()
                            is_json_answer = (
                                answer_stripped.startswith("{") or 
                                answer_stripped.startswith("```json") or
                                (answer_stripped.startswith("```") and "json" in answer_stripped[:20].lower())
                            )
                            
                            if not is_json_answer:
                                if len(answer_text) > len(self.full_answer):
                                    missing_part = answer_text[len(self.full_answer):]
                                    if missing_part:
                                        self.full_answer = answer_text
                                        return {
                                            "type": "stream",
                                            "content": missing_part,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                elif not self.answer_found:
                                    self.full_answer = answer_text
                                    return {
                                        "type": "stream",
                                        "content": answer_text,
                                        "timestamp": datetime.now().isoformat()
                                    }
        
        return None

