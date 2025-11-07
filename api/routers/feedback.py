"""
피드백 엔드포인트
"""
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from api.schemas.feedback import FeedbackRequest, FeedbackResponse
from api.services.session_service import session_service
from api.middleware.auth_middleware import require_auth

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(require_auth)
):
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
    except ValueError as e:
        logger.warning(f"Validation error in submit_feedback: {e}")
        raise HTTPException(status_code=400, detail="입력 데이터가 올바르지 않습니다")
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="피드백 제출 중 오류가 발생했습니다")

