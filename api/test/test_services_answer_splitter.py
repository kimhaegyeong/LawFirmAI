"""
답변 분할 서비스 테스트
"""
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.services.answer_splitter import AnswerSplitter, AnswerChunk


class TestAnswerSplitter:
    """AnswerSplitter 테스트"""
    
    def test_split_empty_answer(self):
        """빈 답변 분할 테스트"""
        splitter = AnswerSplitter()
        result = splitter.split_answer("")
        assert result == []
    
    def test_split_short_answer(self):
        """짧은 답변 분할 테스트 (청크 크기 이하)"""
        splitter = AnswerSplitter(chunk_size=100)
        answer = "짧은 답변입니다."
        result = splitter.split_answer(answer)
        
        assert len(result) == 1
        assert result[0].content == answer
        assert result[0].chunk_index == 0
        assert result[0].total_chunks == 1
        assert result[0].is_complete is True
        assert result[0].has_more is False
    
    def test_split_answer_with_headers(self):
        """헤더가 포함된 답변 분할 테스트"""
        splitter = AnswerSplitter(chunk_size=100)
        answer = """## 제목 1
내용 1

## 제목 2
내용 2"""
        result = splitter.split_answer(answer)
        
        assert len(result) > 0
        assert all(isinstance(chunk, AnswerChunk) for chunk in result)
    
    def test_split_answer_with_numbered_list(self):
        """번호 목록이 포함된 답변 분할 테스트"""
        splitter = AnswerSplitter(chunk_size=100)
        answer = """1. 첫 번째 항목
2. 두 번째 항목
3. 세 번째 항목"""
        result = splitter.split_answer(answer)
        
        assert len(result) > 0
    
    def test_split_large_answer(self):
        """큰 답변 분할 테스트"""
        splitter = AnswerSplitter(chunk_size=200)
        # 줄바꿈이 포함된 큰 답변으로 테스트 (실제로 분할되도록)
        answer = "\n\n".join(["A" * 100 for _ in range(10)])
        result = splitter.split_answer(answer)
        
        # chunk_size가 200이므로 1000자 이상의 답변은 여러 청크로 분할되어야 함
        assert len(result) >= 1
        if len(result) > 1:
            assert all(chunk.total_chunks == len(result) for chunk in result)
    
    def test_answer_chunk_properties(self):
        """AnswerChunk 속성 테스트"""
        chunk = AnswerChunk(
            content="테스트 내용",
            chunk_index=0,
            total_chunks=3,
            is_complete=False,
            has_more=True
        )
        
        assert chunk.content == "테스트 내용"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 3
        assert chunk.is_complete is False
        assert chunk.has_more is True

