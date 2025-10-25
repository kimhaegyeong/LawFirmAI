# -*- coding: utf-8 -*-
"""
Enhanced Chat Service - Contract Extension
대화형 계약서 기능 확장
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def add_contract_methods_to_enhanced_chat_service():
    """EnhancedChatService에 계약서 관련 메서드들을 추가하는 함수"""
    
    def _is_contract_related_query(self, message: str) -> bool:
        """계약서 관련 질문인지 감지"""
        try:
            message_lower = message.lower()
            
            # 계약서 관련 키워드 패턴
            contract_keywords = [
                "계약서", "계약", "작성", "만들", "체결", "서명",
                "용역계약", "근로계약", "부동산계약", "제휴계약",
                "계약서 작성", "계약서 만들", "계약서 양식",
                "계약서 템플릿", "계약서 예시", "계약서 샘플"
            ]
            
            # 질문 패턴
            question_patterns = [
                "어떻게", "방법", "가이드", "도움", "조언",
                "알려주", "설명", "안내", "가르쳐"
            ]
            
            # 계약서 키워드가 있고 질문 패턴이 있는 경우
            has_contract_keyword = any(keyword in message_lower for keyword in contract_keywords)
            has_question_pattern = any(pattern in message_lower for pattern in question_patterns)
            
            return has_contract_keyword and has_question_pattern
            
        except Exception as e:
            self.logger.error(f"계약서 관련 질문 감지 실패: {e}")
            return False
    
    async def _handle_interactive_contract_query(self, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """대화형 계약서 질문 처리"""
        try:
            self.logger.info(f"Handling interactive contract query: {message}")
            
            # 대화형 계약서 어시스턴트가 초기화되지 않은 경우
            if not self.interactive_contract_assistant:
                self.logger.warning("Interactive contract assistant not initialized")
                return {
                    "response": "계약서 작성 기능을 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
                    "session_id": session_id,
                    "error": "contract_assistant_not_available"
                }
            
            # 대화형 계약서 처리
            result = await self.interactive_contract_assistant.process_contract_query(
                message, session_id, user_id
            )
            
            # 처리 시간 추가
            result["processing_time"] = time.time() - time.time()
            
            # 세션에 대화 기록 추가
            if self.integrated_session_manager:
                try:
                    self.integrated_session_manager.add_turn(
                        session_id=session_id,
                        user_query=message,
                        bot_response=result.get("response", ""),
                        question_type="contract_related",
                        user_id=user_id
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to add turn to session: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Interactive contract query handling failed: {e}")
            return {
                "response": "계약서 작성 중 오류가 발생했습니다. 다시 시도해주세요.",
                "error": str(e),
                "session_id": session_id
            }
    
    return _is_contract_related_query, _handle_interactive_contract_query
