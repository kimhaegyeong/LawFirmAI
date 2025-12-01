# -*- coding: utf-8 -*-
"""
UnifiedPromptManager 테스트
통합 프롬프트 관리 시스템 단위 테스트
"""

import pytest
from unittest.mock import patch

from lawfirm_langgraph.core.services.unified_prompt_manager import (
    UnifiedPromptManager,
    LegalDomain,
    ModelType,
    QuestionType
)


@pytest.fixture
def prompt_manager():
    """UnifiedPromptManager 인스턴스 픽스처"""
    return UnifiedPromptManager(prompts_dir="streamlit/prompts")


@pytest.fixture
def sample_document():
    """샘플 문서 픽스처"""
    return {
        "content": "민법 제543조에 따르면 계약 해지권은 당사자 일방이나 쌍방이 해지 또는 해제의 권리가 있는 때에는 그 해지 또는 해제는 상대방에 대한 의사표시로 한다.",
        "law_name": "민법",
        "article_no": "543",
        "relevance_score": 0.85,
        "source": "민법 제543조",
        "metadata": {
            "law_name": "민법",
            "article_no": "543"
        }
    }


@pytest.fixture
def sample_document_with_metadata():
    """메타데이터가 포함된 샘플 문서 픽스처"""
    return {
        "content": "{'query': '계약 해지', 'cross_encoder_score': 0.85, 'original_score': 0.75} 민법 제543조에 따르면...",
        "law_name": "민법",
        "article_no": "543",
        "relevance_score": 0.85,
        "query": "계약 해지",
        "cross_encoder_score": 0.85,
        "original_score": 0.75,
        "keyword_match_score": 0.2,
        "combined_relevance_score": 0.82
    }


@pytest.fixture
def sample_case_document():
    """판례 문서 픽스처"""
    return {
        "content": "대법원 2020. 5. 15. 선고 2019다12345 판결에 따르면 계약 해지 시 손해배상 책임이 발생할 수 있다.",
        "court": "대법원",
        "case_number": "2019다12345",
        "case_name": "손해배상",
        "announce_date": "2020-05-15",
        "relevance_score": 0.78,
        "source": "대법원 판결"
    }


@pytest.fixture
def sample_context():
    """샘플 컨텍스트 픽스처"""
    return {
        "structured_documents": {
            "documents": [
                {
                    "content": "민법 제543조에 따르면...",
                    "law_name": "민법",
                    "article_no": "543",
                    "relevance_score": 0.85
                },
                {
                    "content": "대법원 판결에 따르면...",
                    "court": "대법원",
                    "case_number": "2019다12345",
                    "relevance_score": 0.78
                }
            ]
        },
        "document_count": 2
    }


class TestCleanContent:
    """_clean_content 메서드 테스트"""
    
    def test_clean_content_removes_json_metadata(self, prompt_manager):
        """JSON 형태의 메타데이터 제거 테스트"""
        content = "{'query': '계약 해지', 'score': 0.85} 민법 제543조에 따르면..."
        cleaned = prompt_manager._clean_content(content)
        
        assert "'query':" not in cleaned
        assert "'score':" not in cleaned
        assert "민법 제543조" in cleaned
    
    def test_clean_content_removes_query_info(self, prompt_manager):
        """검색 쿼리 정보 제거 테스트"""
        content = "'query': '계약 해지' 민법 제543조"
        cleaned = prompt_manager._clean_content(content)
        
        assert "'query':" not in cleaned
        assert "민법 제543조" in cleaned
    
    def test_clean_content_removes_score_info(self, prompt_manager):
        """점수 정보 제거 테스트"""
        content = "'cross_encoder_score': 0.85, 'original_score': 0.75 민법 제543조"
        cleaned = prompt_manager._clean_content(content)
        
        assert "'cross_encoder_score':" not in cleaned
        assert "'original_score':" not in cleaned
        assert "민법 제543조" in cleaned
    
    def test_clean_content_removes_metadata_keys(self, prompt_manager):
        """메타데이터 키 제거 테스트"""
        content = "'strategy': 'standard', 'id': 123, 'doc_id': 'doc1' 민법 제543조"
        cleaned = prompt_manager._clean_content(content)
        
        assert "'strategy':" not in cleaned
        assert "'id':" not in cleaned
        assert "'doc_id':" not in cleaned
        assert "민법 제543조" in cleaned
    
    def test_clean_content_removes_empty_brackets(self, prompt_manager):
        """빈 괄호 제거 테스트"""
        content = "민법 제543조 () {} []"
        cleaned = prompt_manager._clean_content(content)
        
        assert "()" not in cleaned
        assert "{}" not in cleaned
        assert "[]" not in cleaned
    
    def test_clean_content_normalizes_whitespace(self, prompt_manager):
        """공백 정리 테스트"""
        content = "민법   제543조에    따르면..."
        cleaned = prompt_manager._clean_content(content)
        
        assert "  " not in cleaned
        assert "민법 제543조에 따르면" in cleaned
    
    def test_clean_content_empty_string(self, prompt_manager):
        """빈 문자열 처리 테스트"""
        assert prompt_manager._clean_content("") == ""
        assert prompt_manager._clean_content(None) == ""


class TestNormalizeDocumentFields:
    """_normalize_document_fields 메서드 테스트"""
    
    def test_normalize_document_fields_basic(self, prompt_manager, sample_document):
        """기본 문서 정규화 테스트"""
        normalized = prompt_manager._normalize_document_fields(sample_document)
        
        assert normalized is not None
        assert normalized["law_name"] == "민법"
        assert normalized["article_no"] == "543"
        assert normalized["relevance_score"] == 0.85
        assert "민법 제543조" in normalized["content"]
    
    def test_normalize_document_fields_with_metadata_cleaning(self, prompt_manager, sample_document_with_metadata):
        """메타데이터 정리 포함 문서 정규화 테스트"""
        normalized = prompt_manager._normalize_document_fields(sample_document_with_metadata)
        
        assert normalized is not None
        assert normalized["law_name"] == "민법"
        assert normalized["article_no"] == "543"
        # content에서 메타데이터가 제거되었는지 확인
        assert "'query':" not in normalized["content"]
        assert "'cross_encoder_score':" not in normalized["content"]
    
    def test_normalize_document_fields_case_document(self, prompt_manager, sample_case_document):
        """판례 문서 정규화 테스트"""
        normalized = prompt_manager._normalize_document_fields(sample_case_document)
        
        assert normalized is not None
        assert normalized["court"] == "대법원"
        assert normalized["case_number"] == "2019다12345"
        assert normalized["case_name"] == "손해배상"
        assert normalized["relevance_score"] == 0.78
    
    def test_normalize_document_fields_creates_title(self, prompt_manager, sample_document):
        """문서 제목 생성 테스트"""
        normalized = prompt_manager._normalize_document_fields(sample_document)
        
        assert normalized["title"] == "민법 제543조"
    
    def test_normalize_document_fields_removes_empty_fields(self, prompt_manager):
        """빈 필드 제거 테스트"""
        doc = {
            "content": "테스트 내용",
            "law_name": "",
            "article_no": "",
            "relevance_score": 0.5
        }
        normalized = prompt_manager._normalize_document_fields(doc)
        
        # 빈 문자열 필드는 제거되어야 함
        assert normalized is not None
        # None 값과 빈 문자열이 제거되었는지 확인
        assert "law_name" not in normalized or normalized.get("law_name") != ""
    
    def test_normalize_document_fields_invalid_input(self, prompt_manager):
        """잘못된 입력 처리 테스트"""
        assert prompt_manager._normalize_document_fields(None) is None
        assert prompt_manager._normalize_document_fields("string") is None
        assert prompt_manager._normalize_document_fields([]) is None
    
    def test_normalize_document_fields_short_content(self, prompt_manager):
        """짧은 content 처리 테스트"""
        doc = {
            "content": "짧음",
            "law_name": "민법",
            "article_no": "543"
        }
        normalized = prompt_manager._normalize_document_fields(doc)
        
        # 법률 정보가 있으면 짧은 content도 허용
        assert normalized is not None


class TestOptimizeDocumentsForPrompt:
    """_optimize_documents_for_prompt 메서드 테스트"""
    
    def test_optimize_documents_removes_duplicates(self, prompt_manager):
        """중복 문서 제거 테스트"""
        docs = [
            {
                "content": "민법 제543조에 따르면 계약 해지권은...",
                "relevance_score": 0.85
            },
            {
                "content": "민법 제543조에 따르면 계약 해지권은...",  # 동일한 내용
                "relevance_score": 0.80
            },
            {
                "content": "대법원 판결에 따르면...",
                "relevance_score": 0.78
            }
        ]
        
        optimized = prompt_manager._optimize_documents_for_prompt(docs, "계약 해지")
        
        # 중복이 제거되어야 함
        assert len(optimized) == 2
    
    def test_optimize_documents_sorts_by_relevance(self, prompt_manager):
        """관련성 점수 기준 정렬 테스트"""
        docs = [
            {"content": "문서 1", "relevance_score": 0.70},
            {"content": "문서 2", "relevance_score": 0.90},
            {"content": "문서 3", "relevance_score": 0.80}
        ]
        
        optimized = prompt_manager._optimize_documents_for_prompt(docs, "테스트")
        
        assert len(optimized) == 3
        assert optimized[0]["relevance_score"] == 0.90
        assert optimized[1]["relevance_score"] == 0.80
        assert optimized[2]["relevance_score"] == 0.70
    
    def test_optimize_documents_limits_to_8(self, prompt_manager):
        """최대 8개 문서 제한 테스트"""
        docs = [
            {"content": f"문서 {i}", "relevance_score": 0.5 + i * 0.05}
            for i in range(15)
        ]
        
        optimized = prompt_manager._optimize_documents_for_prompt(docs, "테스트")
        
        assert len(optimized) == 8
    
    def test_optimize_documents_filters_short_content(self, prompt_manager):
        """짧은 content 필터링 테스트"""
        docs = [
            {"content": "짧음", "relevance_score": 0.85},
            {"content": "충분히 긴 문서 내용입니다. 최소 10자 이상입니다.", "relevance_score": 0.80}
        ]
        
        optimized = prompt_manager._optimize_documents_for_prompt(docs, "테스트")
        
        # 짧은 content는 제거되어야 함
        assert len(optimized) == 1
        assert len(optimized[0]["content"]) >= 10
    
    def test_optimize_documents_empty_list(self, prompt_manager):
        """빈 리스트 처리 테스트"""
        assert prompt_manager._optimize_documents_for_prompt([], "테스트") == []
        assert prompt_manager._optimize_documents_for_prompt(None, "테스트") == []


class TestSmartTruncateDocument:
    """_smart_truncate_document 메서드 테스트"""
    
    def test_smart_truncate_short_content(self, prompt_manager):
        """짧은 content는 축약하지 않음"""
        content = "짧은 내용"
        result = prompt_manager._smart_truncate_document(content, 100, "테스트")
        
        assert result == content
    
    def test_smart_truncate_long_content(self, prompt_manager):
        """긴 content 축약 테스트"""
        content = "민법 제543조에 따르면 계약 해지권은 당사자 일방이나 쌍방이 해지 또는 해제의 권리가 있는 때에는 그 해지 또는 해제는 상대방에 대한 의사표시로 한다. " * 10
        result = prompt_manager._smart_truncate_document(content, 100, "계약 해지")
        
        assert len(result) <= 100
        assert "..." in result or len(result) < len(content)
    
    def test_smart_truncate_prioritizes_relevant_sentences(self, prompt_manager):
        """관련 문장 우선 선택 테스트"""
        content = "계약 해지에 대한 내용입니다. " * 5 + "관련 없는 내용입니다. " * 5
        result = prompt_manager._smart_truncate_document(content, 50, "계약 해지")
        
        # 관련 키워드가 포함된 문장이 우선 선택되어야 함
        assert "계약 해지" in result


class TestBuildFinalPrompt:
    """_build_final_prompt 메서드 테스트"""
    
    def test_build_final_prompt_basic(self, prompt_manager, sample_context):
        """기본 프롬프트 구성 테스트"""
        base_prompt = "당신은 법률 전문가입니다."
        query = "계약 해지에 대해 알려주세요"
        
        result = prompt_manager._build_final_prompt(
            base_prompt,
            query,
            sample_context,
            QuestionType.LEGAL_ADVICE
        )
        
        assert isinstance(result, str)
        assert query in result
        assert "검색된 법률 문서" in result or "문서" in result
    
    def test_build_final_prompt_includes_documents(self, prompt_manager, sample_context):
        """문서 포함 테스트"""
        base_prompt = "당신은 법률 전문가입니다."
        query = "계약 해지"
        
        result = prompt_manager._build_final_prompt(
            base_prompt,
            query,
            sample_context,
            QuestionType.LEGAL_ADVICE
        )
        
        # 문서 내용이 포함되어야 함
        assert "민법" in result or "대법원" in result
    
    def test_build_final_prompt_no_documents(self, prompt_manager):
        """문서가 없을 때 처리 테스트"""
        base_prompt = "당신은 법률 전문가입니다."
        query = "계약 해지"
        context = {"document_count": 0}
        
        result = prompt_manager._build_final_prompt(
            base_prompt,
            query,
            context,
            QuestionType.LEGAL_ADVICE
        )
        
        assert isinstance(result, str)
        assert query in result
    
    def test_build_final_prompt_token_limit(self, prompt_manager, sample_context):
        """토큰 제한 테스트"""
        base_prompt = "당신은 법률 전문가입니다. " * 1000
        query = "계약 해지"
        
        result = prompt_manager._build_final_prompt(
            base_prompt,
            query,
            sample_context,
            QuestionType.LEGAL_ADVICE
        )
        
        # 토큰 수가 제한 내에 있어야 함
        tokens = prompt_manager._estimate_tokens(result)
        assert tokens < 1_048_576  # Gemini 2.5 Flash 최대 토큰


class TestGetOptimizedPrompt:
    """get_optimized_prompt 메서드 테스트"""
    
    def test_get_optimized_prompt_basic(self, prompt_manager, sample_context):
        """기본 최적화 프롬프트 생성 테스트"""
        query = "계약 해지에 대해 알려주세요"
        
        result = prompt_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            context=sample_context
        )
        
        assert isinstance(result, str)
        assert query in result
        assert len(result) > 0
    
    def test_get_optimized_prompt_with_domain(self, prompt_manager, sample_context):
        """도메인 포함 프롬프트 생성 테스트"""
        query = "계약 해지"
        
        result = prompt_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.CIVIL_LAW,
            context=sample_context
        )
        
        assert isinstance(result, str)
        assert query in result
    
    def test_get_optimized_prompt_with_model_type(self, prompt_manager, sample_context):
        """모델 타입 지정 프롬프트 생성 테스트"""
        query = "계약 해지"
        
        result = prompt_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            model_type=ModelType.GEMINI,
            context=sample_context
        )
        
        assert isinstance(result, str)
        assert query in result
    
    def test_get_optimized_prompt_fallback(self, prompt_manager):
        """폴백 프롬프트 테스트"""
        query = "테스트 질문"
        
        # 예외 발생 시 폴백 프롬프트 반환
        with patch.object(prompt_manager, '_build_final_prompt', side_effect=Exception("Test error")):
            result = prompt_manager.get_optimized_prompt(
                query=query,
                question_type=QuestionType.LEGAL_ADVICE,
                context={}
            )
            
            assert isinstance(result, str)
            assert query in result


class TestEstimateTokens:
    """_estimate_tokens 메서드 테스트"""
    
    def test_estimate_tokens_korean(self, prompt_manager):
        """한국어 토큰 수 추정 테스트"""
        text = "민법 제543조에 따르면 계약 해지권은 당사자 일방이나 쌍방이 해지 또는 해제의 권리가 있는 때에는 그 해지 또는 해제는 상대방에 대한 의사표시로 한다."
        tokens = prompt_manager._estimate_tokens(text)
        
        # 한국어는 1토큰 = 2.5자로 추정
        expected_min = len(text) // 3  # 보수적 추정
        expected_max = len(text) // 2  # 낙관적 추정
        
        assert expected_min <= tokens <= expected_max
    
    def test_estimate_tokens_empty(self, prompt_manager):
        """빈 문자열 토큰 수 추정 테스트"""
        assert prompt_manager._estimate_tokens("") == 0
        assert prompt_manager._estimate_tokens(None) == 0
    
    def test_estimate_tokens_mixed(self, prompt_manager):
        """한국어와 영어 혼합 토큰 수 추정 테스트"""
        text = "민법 제543조 Civil Code Article 543"
        tokens = prompt_manager._estimate_tokens(text)
        
        assert tokens > 0


class TestRemoveDuplicateDocumentSections:
    """_remove_duplicate_document_sections 메서드 테스트"""
    
    def test_remove_duplicate_sections(self, prompt_manager):
        """중복 문서 섹션 제거 테스트"""
        prompt = """
## 검색된 법률 문서
문서 1 내용

## 검색된 법률 문서
문서 2 내용

## 사용자 질문
질문 내용
"""
        result = prompt_manager._remove_duplicate_document_sections(prompt)
        
        # 첫 번째 섹션만 남아야 함
        assert result.count("## 검색된 법률 문서") == 1
    
    def test_remove_duplicate_sections_no_duplicates(self, prompt_manager):
        """중복이 없을 때 처리 테스트"""
        prompt = """
## 검색된 법률 문서
문서 내용

## 사용자 질문
질문 내용
"""
        result = prompt_manager._remove_duplicate_document_sections(prompt)
        
        assert result.count("## 검색된 법률 문서") == 1


class TestIntegration:
    """통합 테스트"""
    
    def test_full_pipeline(self, prompt_manager):
        """전체 파이프라인 테스트"""
        # 1. 문서 정규화
        doc = {
            "content": "{'query': '계약 해지', 'score': 0.85} 민법 제543조에 따르면...",
            "law_name": "민법",
            "article_no": "543",
            "relevance_score": 0.85
        }
        normalized = prompt_manager._normalize_document_fields(doc)
        assert normalized is not None
        
        # 2. 문서 최적화
        optimized = prompt_manager._optimize_documents_for_prompt([normalized], "계약 해지")
        assert len(optimized) == 1
        
        # 3. 프롬프트 생성
        context = {
            "structured_documents": {
                "documents": [normalized]
            },
            "document_count": 1
        }
        prompt = prompt_manager.get_optimized_prompt(
            query="계약 해지",
            question_type=QuestionType.LEGAL_ADVICE,
            context=context
        )
        
        assert isinstance(prompt, str)
        assert "계약 해지" in prompt
        assert "민법" in prompt or "문서" in prompt

