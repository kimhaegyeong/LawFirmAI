"""
스트리밍 이벤트 처리기
LangGraph 이벤트를 처리하고 토큰을 추출합니다.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .stream_config import StreamConfig

logger = logging.getLogger(__name__)


class StreamEventProcessor:
    """스트리밍 이벤트 처리기"""
    
    _ANSWER_NODE_KEYWORDS = frozenset(["generate_answer", "generate_and_validate", "direct_answer"])
    _ANSWER_NODE_NAMES = frozenset(["generate_answer_enhanced", "generate_and_validate_answer", "direct_answer"])
    _JSON_START_PATTERNS = frozenset(["{", "```json", "```"])
    _JSON_RESET_KEYWORDS = frozenset(['"complexity"', '"confidence"', '"reasoning"'])
    _CHAT_GOOGLE_GENERATIVE_AI = "ChatGoogleGenerativeAI"
    _GENERATE_AND_VALIDATE_ANSWER = "generate_and_validate_answer"
    
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
    
    def _check_node_name(self, node_name: str) -> bool:
        """노드 이름이 답변 생성 노드인지 확인"""
        if not node_name:
            return False
        node_name_lower = node_name.lower()
        return (
            any(keyword in node_name_lower for keyword in self._ANSWER_NODE_KEYWORDS) or
            node_name in self._ANSWER_NODE_NAMES
        )
    
    def _get_parent_node_name(self, event: Dict[str, Any]) -> Optional[str]:
        """이벤트에서 상위 노드 이름 추출"""
        event_parent = event.get("parent", {})
        event_tags = event.get("tags", [])
        
        if isinstance(event_parent, dict):
            return event_parent.get("name", "")
        elif isinstance(event_tags, list):
            for tag in event_tags:
                if isinstance(tag, str) and any(keyword in tag.lower() for keyword in self._ANSWER_NODE_KEYWORDS):
                    return tag
        return None
    
    def is_answer_node(self, event: Dict[str, Any]) -> bool:
        """답변 생성 노드인지 확인"""
        event_name = event.get("name", "")
        
        if self._check_node_name(event_name):
            return True
        
        parent_node_name = self._get_parent_node_name(event)
        if parent_node_name and self._check_node_name(parent_node_name):
            return True
        
        if event_name == self._CHAT_GOOGLE_GENERATIVE_AI and self.answer_generation_started:
            return self.last_node_name in self.config.answer_generation_nodes
        
        return False
    
    def _check_json_start_pattern(self, text: str) -> bool:
        """텍스트가 JSON 시작 패턴으로 시작하는지 확인"""
        text_stripped = text.strip()
        if text_stripped.startswith("{") or text_stripped.startswith("```json"):
            return True
        if text_stripped.startswith("```"):
            return "json" in text_stripped[:20].lower()
        return False
    
    def _check_json_keywords(self, text: str) -> bool:
        """텍스트에 JSON 키워드가 포함되어 있는지 확인"""
        return any(keyword in text for keyword in self.json_keywords)
    
    def _check_json_pattern(self, text: str) -> bool:
        """텍스트가 JSON 패턴과 일치하는지 확인"""
        text_stripped = text.strip()
        return bool(self.json_pattern.match(text_stripped) or self.json_pattern.match(text))
    
    def is_json_output(self, chunk: str) -> bool:
        """JSON 형식 출력인지 확인"""
        if not chunk:
            return False
        
        chunk_stripped = chunk.strip()
        
        if self._check_json_start_pattern(chunk_stripped):
            return True
        
        if "```json" in chunk or "``` json" in chunk:
            return True
        
        if self.full_answer:
            combined_text = (self.full_answer + chunk).strip()
            if self._check_json_start_pattern(combined_text):
                return True
            if "```json" in combined_text or "``` json" in combined_text:
                return True
        
        if self._check_json_keywords(chunk):
            return True
        
        if self.full_answer and self._check_json_keywords(self.full_answer + chunk):
            return True
        
        if self._check_json_pattern(chunk):
            return True
        
        return False
    
    def _extract_text_from_dict(self, data: Dict[str, Any], keys: tuple) -> Optional[str]:
        """딕셔너리에서 여러 키를 시도하여 텍스트 추출"""
        for key in keys:
            value = data.get(key)
            if value:
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    return value.get("answer", "") or value.get("content", "")
        return None
    
    def _get_new_chunk(self, full_text: str) -> Optional[str]:
        """전체 텍스트에서 새로운 청크 부분 추출"""
        if not full_text or not isinstance(full_text, str):
            return None
        if len(full_text) > len(self.full_answer):
            new_part = full_text[len(self.full_answer):]
            self.full_answer = full_text
            return new_part
        return None
    
    def extract_chunk_from_chain_stream(self, event_data: Dict[str, Any]) -> Optional[str]:
        """on_chain_stream 이벤트에서 청크 추출"""
        if not isinstance(event_data, dict):
            return None
        
        chain_output = event_data.get("chunk") or event_data.get("output")
        if chain_output is None:
            text_content = self._extract_text_from_dict(event_data, ("text", "content"))
            if text_content:
                return self._get_new_chunk(text_content)
            return None
        
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
            
            if full_answer_from_event:
                return self._get_new_chunk(full_answer_from_event)
        
        elif isinstance(chain_output, str):
            return self._get_new_chunk(chain_output)
        
        elif hasattr(chain_output, "content"):
            content = chain_output.content
            if isinstance(content, str):
                return self._get_new_chunk(content)
            elif isinstance(content, dict):
                answer_text = content.get("answer", "") or content.get("content", "")
                if answer_text:
                    return self._get_new_chunk(answer_text)
        
        return None
    
    def _extract_content_from_object(self, obj: Any) -> Optional[str]:
        """객체에서 content 추출"""
        if hasattr(obj, "content"):
            content = obj.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and len(content) > 0:
                return content[0] if isinstance(content[0], str) else str(content[0])
            else:
                return str(content) if content is not None else None
        return None
    
    def extract_chunk_from_llm_stream(self, event_data: Dict[str, Any]) -> Optional[str]:
        """on_llm_stream 또는 on_chat_model_stream 이벤트에서 청크 추출"""
        if not isinstance(event_data, dict):
            return None
        
        chunk_obj = event_data.get("chunk")
        if chunk_obj is None:
            return event_data.get("text") or event_data.get("content")
        
        if isinstance(chunk_obj, str):
            return chunk_obj
        
        if hasattr(chunk_obj, "text"):
            return chunk_obj.text
        
        chunk = self._extract_content_from_object(chunk_obj)
        if chunk:
            return chunk
        
        if hasattr(chunk_obj, "__class__") and "AIMessageChunk" in str(type(chunk_obj)):
            try:
                content = getattr(chunk_obj, "content", None)
                if isinstance(content, str):
                    return content
                elif isinstance(content, list) and len(content) > 0:
                    return content[0] if isinstance(content[0], str) else str(content[0])
                elif content is not None:
                    return str(content)
            except Exception:
                pass
        
        if "delta" in event_data:
            delta = event_data["delta"]
            if isinstance(delta, dict):
                return delta.get("content") or delta.get("text")
            elif isinstance(delta, str):
                return delta
        
        return None
    
    def _create_stream_event(self, content: str, event_type: str = "stream", node_name: Optional[str] = None) -> Dict[str, Any]:
        """스트림 이벤트 딕셔너리 생성"""
        result = {
            "type": event_type,
            "timestamp": datetime.now().isoformat()
        }
        if event_type == "stream":
            result["content"] = content
        elif event_type == "progress":
            result["message"] = content
            if node_name:
                result["node_name"] = node_name
        return result
    
    def _handle_chain_start(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """on_chain_start 이벤트 처리"""
        node_name = event.get("name", "")
        if node_name in self.config.answer_generation_nodes:
            self.answer_generation_started = True
            self.json_output_detected = False
            self.last_node_name = node_name
            return self._create_stream_event("답변 생성 중...", "progress", node_name)
        return None
    
    def _handle_chain_end(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """on_chain_end 이벤트 처리"""
        node_name = event.get("name", "")
        if node_name == self._GENERATE_AND_VALIDATE_ANSWER:
            self.answer_generation_started = False
            return self._extract_final_answer_from_chain_end(event)
        return None
    
    def _extract_final_answer_from_chain_end(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """on_chain_end에서 최종 답변 추출"""
        event_data = event.get("data", {})
        if not isinstance(event_data, dict):
            return None
        
        output = event_data.get("output")
        if not output or not isinstance(output, dict):
            return None
        
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
        
        if not answer_text or not isinstance(answer_text, str) or len(answer_text) == 0:
            return None
        
        answer_stripped = answer_text.strip()
        is_json_answer = (
            answer_stripped.startswith("{") or 
            answer_stripped.startswith("```json") or
            (answer_stripped.startswith("```") and "json" in answer_stripped[:20].lower())
        )
        
        if is_json_answer:
            return None
        
        if len(answer_text) > len(self.full_answer):
            missing_part = answer_text[len(self.full_answer):]
            if missing_part:
                self.full_answer = answer_text
                return self._create_stream_event(missing_part)
        elif not self.answer_found:
            self.full_answer = answer_text
            return self._create_stream_event(answer_text)
        
        return None
    
    def _extract_final_answer_from_output(self, output: Any) -> Optional[str]:
        """출력 객체에서 최종 답변 추출"""
        if hasattr(output, "content"):
            return output.content
        elif isinstance(output, str):
            return output
        elif isinstance(output, dict):
            return output.get("content") or output.get("text") or str(output)
        else:
            return str(output)
    
    def _handle_llm_end(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """on_llm_end 또는 on_chat_model_end 이벤트 처리"""
        event_data = event.get("data", {})
        if not isinstance(event_data, dict):
            return None
        
        output = event_data.get("output")
        if output is None:
            return None
        
        final_answer = self._extract_final_answer_from_output(output)
        if not final_answer or not isinstance(final_answer, str):
            return None
        
        if len(final_answer) > len(self.full_answer):
            missing_part = final_answer[len(self.full_answer):]
            if missing_part:
                self.full_answer = final_answer
                return self._create_stream_event(missing_part)
        
        return None
    
    def _process_streaming_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """스트리밍 청크 처리 및 이벤트 생성"""
        if not chunk or not isinstance(chunk, str) or len(chunk) == 0:
            return None
        
        if self.is_json_output(chunk):
            self.json_output_detected = True
            if self.config.debug_stream:
                logger.debug(f"JSON 형식 출력 감지 및 무시: {chunk[:100]}...")
            return None
        
        if not self.is_json_output(chunk) and self.json_output_detected and len(chunk.strip()) > 0:
            if not any(keyword in chunk for keyword in self._JSON_RESET_KEYWORDS):
                self.json_output_detected = False
                if self.config.debug_stream:
                    logger.debug("실제 답변이 시작되어 JSON 출력 플래그 리셋")
        
        self.full_answer += chunk
        self.tokens_received += 1
        self.answer_found = True
        
        return self._create_stream_event(chunk)
    
    def _handle_streaming_events(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """스트리밍 이벤트 처리 (on_llm_stream, on_chat_model_stream, on_chain_stream)"""
        event_type = event.get("event", "")
        event_name = event.get("name", "")
        
        # 디버깅 로깅
        if self.config.debug_stream:
            logger.debug(
                f"[_handle_streaming_events] 이벤트 처리 시작: "
                f"type={event_type}, name={event_name}, "
                f"answer_generation_started={self.answer_generation_started}, "
                f"is_answer_node={self.is_answer_node(event)}"
            )
        
        # answer_generation_started가 False인 경우에도 로깅
        if not self.answer_generation_started:
            if self.config.debug_stream:
                logger.debug(
                    f"[_handle_streaming_events] answer_generation_started=False로 건너뜀: "
                    f"type={event_type}, name={event_name}"
                )
            # answer_generation_started가 False여도 답변 생성 노드인 경우 시작 플래그 설정
            if self.is_answer_node(event):
                self.answer_generation_started = True
                self.last_node_name = event_name
                logger.info(f"[_handle_streaming_events] ✅ answer_generation_started를 True로 설정: name={event_name}")
            else:
                return None
        
        if not self.is_answer_node(event):
            if self.config.debug_stream:
                logger.debug(
                    f"[_handle_streaming_events] is_answer_node=False로 건너뜀: "
                    f"type={event_type}, name={event_name}"
                )
            return None
        
        event_data = event.get("data", {})
        
        if event.get("event") == "on_chain_stream":
            chunk = self.extract_chunk_from_chain_stream(event_data)
        else:
            chunk = self.extract_chunk_from_llm_stream(event_data)
        
        if self.config.debug_stream and chunk:
            logger.debug(
                f"[_handle_streaming_events] ✅ 청크 추출 성공: "
                f"chunk_length={len(chunk)}, chunk_preview={chunk[:50]}..."
            )
        elif self.config.debug_stream and not chunk:
            logger.warning(
                f"[_handle_streaming_events] ⚠️ 청크 추출 실패: "
                f"type={event_type}, name={event_name}, "
                f"data_keys={list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}"
            )
        
        return self._process_streaming_chunk(chunk) if chunk else None
    
    def process_stream_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """스트리밍 이벤트 처리 및 스트림 이벤트 생성"""
        event_type = event.get("event", "")
        
        if event_type == "on_chain_start":
            return self._handle_chain_start(event)
        elif event_type == "on_chain_end":
            return self._handle_chain_end(event)
        elif event_type in ["on_llm_stream", "on_chat_model_stream", "on_chain_stream"]:
            return self._handle_streaming_events(event)
        elif event_type in ["on_llm_end", "on_chat_model_end"]:
            return self._handle_llm_end(event)
        
        return None

