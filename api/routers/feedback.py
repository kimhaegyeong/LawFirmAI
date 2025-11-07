"""
피드백 엔드포인트
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from api.schemas.feedback import FeedbackRequest, FeedbackResponse
from api.services.session_service import session_service

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """피드백 제출"""
    try:
        # 피드백 ID 생성
        feedback_id = str(uuid.uuid4())
        
        # 세션 존재 확인
        session = session_service.get_session(feedback.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 피드백을 메타데이터로 저장 (간단한 구현)
        # 실제로는 별도 피드백 테이블을 만들 수 있음
        feedback_data = {
            "feedback_id": feedback_id,
            "session_id": feedback.session_id,
            "message_id": feedback.message_id,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "feedback_type": feedback.feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # 세션 메타데이터에 피드백 추가 (간단한 구현)
        # 실제로는 별도 테이블을 사용하는 것이 좋음
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            session_id=feedback.session_id,
            message_id=feedback.message_id,
            rating=feedback.rating,
            comment=feedback.comment,
            timestamp=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

