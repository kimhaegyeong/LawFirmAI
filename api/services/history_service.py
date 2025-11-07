"""
히스토리 관리 서비스
"""
import logging
from typing import List, Dict, Any, Optional
from api.services.session_service import session_service

logger = logging.getLogger(__name__)


class HistoryService:
    """히스토리 관리 서비스"""
    
    def get_history(
        self,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """히스토리 조회"""
        try:
            if session_id:
                # 특정 세션의 메시지 조회
                messages = session_service.get_messages(session_id)
                return {
                    "messages": messages,
                    "total": len(messages),
                    "page": page,
                    "page_size": page_size
                }
            else:
                # 세션 목록 기반 조회
                sessions, total = session_service.list_sessions(
                    category=category,
                    search=search,
                    page=page,
                    page_size=page_size
                )
                
                # 각 세션의 메시지를 가져옴
                all_messages = []
                for session in sessions:
                    session_messages = session_service.get_messages(session["session_id"])
                    all_messages.extend(session_messages)
                
                return {
                    "messages": all_messages,
                    "total": total,
                    "page": page,
                    "page_size": page_size
                }
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {
                "messages": [],
                "total": 0,
                "page": page,
                "page_size": page_size
            }


# 전역 서비스 인스턴스
history_service = HistoryService()

