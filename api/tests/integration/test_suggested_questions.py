"""
추천 질문 기능 테스트
"""
import pytest
import asyncio
import json
from api.services.chat_service import ChatService


@pytest.mark.integration
@pytest.mark.asyncio
class TestSuggestedQuestions:
    """추천 질문 기능 테스트"""
    
    async def test_suggested_questions_generation(self):
        """추천 질문 생성 테스트"""
        chat_service = ChatService()
        
        test_queries = [
            "계약서 작성 시 주의사항은 무엇인가요?",
            "손해배상 청구는 어떻게 하나요?",
            "이혼 절차를 알려주세요",
        ]
        
        for i, query in enumerate(test_queries, 1):
            result = await chat_service.process_message(
                message=query,
                session_id=f"test-session-{i}",
                enable_checkpoint=False
            )
            
            # 결과 확인
            assert "answer" in result
            assert len(result.get("answer", "")) > 0
            
            # metadata 확인
            metadata = result.get("metadata", {})
            
            # related_questions 확인
            related_questions = metadata.get("related_questions")
            if related_questions:
                assert isinstance(related_questions, list)
                assert len(related_questions) > 0

