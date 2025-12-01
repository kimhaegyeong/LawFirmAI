# -*- coding: utf-8 -*-
"""
쿼리 확장 검색 기능 테스트

새로 구현한 LangGraph 노드 → 서브노드 → 태스크 구조 개선 사항 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (하위 폴더로 이동하여 parent 하나 추가)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


def test_semantic_search_engine_query_expansion():
    """SemanticSearchEngineV2의 쿼리 확장 검색 테스트"""
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        from lawfirm_langgraph.core.utils.config import Config
        
        logger.info("=" * 80)
        logger.info("테스트 1: SemanticSearchEngineV2 쿼리 확장 검색")
        logger.info("=" * 80)
        
        config = Config()
        db_path = config.database_path
        
        # 검색 엔진 초기화
        search_engine = SemanticSearchEngineV2(db_path=db_path)
        logger.info(f"✅ SemanticSearchEngineV2 초기화 완료: {db_path}")
        
        # 테스트 쿼리
        test_query = "손해배상 청구"
        expanded_keywords = ["불법행위", "과실", "고의", "손해", "배상"]
        
        logger.info(f"\n📝 테스트 쿼리: {test_query}")
        logger.info(f"📝 확장 키워드: {expanded_keywords}")
        
        # 1. 기존 검색 방식
        logger.info("\n1️⃣ 기존 검색 방식 테스트...")
        results_old = search_engine.search(
            query=test_query,
            k=5,
            similarity_threshold=0.5
        )
        logger.info(f"   결과: {len(results_old)}개 문서")
        if results_old:
            logger.info(f"   상위 3개 점수: {[r.get('relevance_score', 0.0) for r in results_old[:3]]}")
        
        # 2. 쿼리 확장 검색 방식
        logger.info("\n2️⃣ 쿼리 확장 검색 방식 테스트...")
        results_new = search_engine.search_with_query_expansion(
            query=test_query,
            k=5,
            similarity_threshold=0.5,
            expanded_keywords=expanded_keywords,
            use_query_variations=True
        )
        logger.info(f"   결과: {len(results_new)}개 문서")
        if results_new:
            logger.info(f"   상위 3개 점수: {[r.get('relevance_score', 0.0) for r in results_new[:3]]}")
            logger.info(f"   쿼리 변형 타입: {[r.get('query_variation', 'unknown') for r in results_new[:3]]}")
        
        # 3. 쿼리 변형 생성 테스트
        logger.info("\n3️⃣ 쿼리 변형 생성 테스트...")
        variations = search_engine._generate_simple_query_variations(test_query, expanded_keywords)
        logger.info(f"   생성된 변형 수: {len(variations)}")
        for i, var in enumerate(variations, 1):
            logger.info(f"   변형 {i}: {var['type']} - '{var['query'][:50]}...' (가중치: {var['weight']})")
        
        # 4. 핵심 키워드 추출 테스트
        logger.info("\n4️⃣ 핵심 키워드 추출 테스트...")
        core_keywords = search_engine._extract_core_keywords_simple(test_query)
        logger.info(f"   추출된 핵심 키워드: {core_keywords}")
        
        logger.info("\n✅ SemanticSearchEngineV2 쿼리 확장 검색 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_query_expansion_subnode():
    """쿼리 확장 서브노드 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 2: 쿼리 확장 서브노드")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "query_type": "legal_question",
            "legal_field": "civil",
            "optimized_queries": {
                "expanded_keywords": ["불법행위", "과실", "고의", "손해", "배상"]
            },
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 테스트 쿼리: {test_state['query']}")
        logger.info(f"📝 확장 키워드: {test_state['optimized_queries']['expanded_keywords']}")
        
        # 쿼리 확장 서브노드 실행
        logger.info("\n🔄 쿼리 확장 서브노드 실행 중...")
        result_state = workflow.query_expansion_subnode(test_state)
        
        # 결과 확인
        expanded_queries = result_state.get("expanded_queries", {})
        logger.info(f"\n✅ 쿼리 확장 완료:")
        logger.info(f"   - 변형 수: {len(expanded_queries.get('variations', []))}")
        logger.info(f"   - 연관 키워드 수: {len(expanded_queries.get('related_keywords', []))}")
        logger.info(f"   - 정규화된 쿼리 수: {len(expanded_queries.get('normalized', []))}")
        logger.info(f"   - 전체 쿼리 수: {len(expanded_queries.get('all_queries', []))}")
        
        if expanded_queries.get('variations'):
            logger.info("\n   생성된 쿼리 변형:")
            for i, var in enumerate(expanded_queries['variations'][:5], 1):
                logger.info(f"   {i}. {var['type']}: '{var['query'][:60]}...' (가중치: {var['weight']})")
        
        if expanded_queries.get('related_keywords'):
            logger.info(f"\n   연관 키워드: {expanded_queries['related_keywords'][:10]}")
        
        logger.info("\n✅ 쿼리 확장 서브노드 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_semantic_search_variations_subnode():
    """의미적 검색 변형 서브노드 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 3: 의미적 검색 변형 서브노드")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성 (쿼리 확장 결과 포함)
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "query_type": "legal_question",
            "legal_field": "civil",
            "expanded_queries": {
                "original": "손해배상 청구",
                "variations": [
                    {"query": "손해배상 청구", "type": "original", "weight": 1.0, "priority": 1},
                    {"query": "손해배상 청구 불법행위 과실 고의", "type": "keyword_expanded", "weight": 0.9, "priority": 2},
                    {"query": "손해배상", "type": "core_keywords", "weight": 0.85, "priority": 2}
                ],
                "all_queries": ["손해배상 청구", "손해배상 청구 불법행위 과실 고의", "손해배상"]
            },
            "search_params": {
                "semantic_k": 5,
                "similarity_threshold": 0.5
            },
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 테스트 쿼리: {test_state['query']}")
        logger.info(f"📝 쿼리 변형 수: {len(test_state['expanded_queries']['variations'])}")
        
        # 의미적 검색 변형 서브노드 실행
        logger.info("\n🔄 의미적 검색 변형 서브노드 실행 중...")
        result_state = workflow.semantic_search_variations_subnode(test_state)
        
        # 결과 확인
        semantic_results = result_state.get("semantic_results", [])
        semantic_count = result_state.get("semantic_count", 0)
        
        logger.info(f"\n✅ 의미적 검색 완료:")
        logger.info(f"   - 결과 수: {semantic_count}")
        logger.info(f"   - 고유 결과 수: {len(semantic_results)}")
        
        if semantic_results:
            logger.info("\n   상위 5개 결과:")
            for i, result in enumerate(semantic_results[:5], 1):
                query_type = result.get('query_type', 'unknown')
                relevance = result.get('relevance_score', 0.0)
                source = result.get('source', 'Unknown')[:30]
                logger.info(f"   {i}. [{query_type}] 점수: {relevance:.3f}, 소스: {source}...")
        
        logger.info("\n✅ 의미적 검색 변형 서브노드 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_keyword_search_subnode():
    """키워드 검색 서브노드 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 4: 키워드 검색 서브노드")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "query_type": "legal_question",
            "legal_field": "civil",
            "expanded_queries": {
                "original": "손해배상 청구",
                "all_queries": ["손해배상 청구", "손해배상"],
                "related_keywords": ["불법행위", "과실", "고의"]
            },
            "search_params": {
                "keyword_limit": 5
            },
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 테스트 쿼리: {test_state['query']}")
        logger.info(f"📝 키워드 쿼리 수: {len(test_state['expanded_queries']['all_queries'])}")
        
        # 키워드 검색 서브노드 실행
        logger.info("\n🔄 키워드 검색 서브노드 실행 중...")
        result_state = workflow.keyword_search_subnode(test_state)
        
        # 결과 확인
        keyword_results = result_state.get("keyword_results", [])
        keyword_count = result_state.get("keyword_count", 0)
        
        logger.info(f"\n✅ 키워드 검색 완료:")
        logger.info(f"   - 결과 수: {keyword_count}")
        logger.info(f"   - 고유 결과 수: {len(keyword_results)}")
        
        if keyword_results:
            logger.info("\n   상위 5개 결과:")
            for i, result in enumerate(keyword_results[:5], 1):
                source_type = result.get('source_type', 'unknown')
                relevance = result.get('relevance_score', 0.0)
                source = result.get('source', 'Unknown')[:30]
                logger.info(f"   {i}. [{source_type}] 점수: {relevance:.3f}, 소스: {source}...")
        
        logger.info("\n✅ 키워드 검색 서브노드 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_result_merger_subnode():
    """결과 통합 서브노드 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 5: 결과 통합 서브노드")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성 (의미적 검색 및 키워드 검색 결과 포함)
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "semantic_results": [
                {"id": "1", "content": "손해배상에 관한 법률", "relevance_score": 0.9, "metadata": {"chunk_id": "1"}},
                {"id": "2", "content": "불법행위 책임", "relevance_score": 0.8, "metadata": {"chunk_id": "2"}},
            ],
            "keyword_results": [
                {"id": "1", "content": "손해배상에 관한 법률", "relevance_score": 0.85, "metadata": {"chunk_id": "1"}},
                {"id": "3", "content": "민법 제750조", "relevance_score": 0.75, "metadata": {"chunk_id": "3"}},
            ],
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 의미적 검색 결과: {len(test_state['semantic_results'])}개")
        logger.info(f"📝 키워드 검색 결과: {len(test_state['keyword_results'])}개")
        
        # 결과 통합 서브노드 실행
        logger.info("\n🔄 결과 통합 서브노드 실행 중...")
        result_state = workflow.result_merger_subnode(test_state)
        
        # 결과 확인
        merged_documents = result_state.get("merged_documents", [])
        retrieved_docs = result_state.get("retrieved_docs", [])
        
        logger.info(f"\n✅ 결과 통합 완료:")
        logger.info(f"   - 통합된 문서 수: {len(merged_documents)}")
        logger.info(f"   - retrieved_docs 수: {len(retrieved_docs)}")
        
        if merged_documents:
            logger.info("\n   통합된 상위 5개 결과:")
            for i, doc in enumerate(merged_documents[:5], 1):
                search_method = doc.get('search_method', 'unknown')
                final_score = doc.get('final_weighted_score', doc.get('relevance_score', 0.0))
                logger.info(f"   {i}. [{search_method}] 최종 점수: {final_score:.3f}")
        
        logger.info("\n✅ 결과 통합 서브노드 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_expanded_queries_missing():
    """expanded_queries가 없는 경우 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 6: expanded_queries가 없는 경우")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성 (expanded_queries 없음)
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "query_type": "legal_question",
            "legal_field": "civil",
            "search_params": {
                "semantic_k": 5,
                "similarity_threshold": 0.5
            },
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 테스트 쿼리: {test_state['query']}")
        logger.info("📝 expanded_queries: 없음 (폴백 로직 테스트)")
        
        # _get_expanded_queries 헬퍼 메서드 테스트
        logger.info("\n🔄 _get_expanded_queries 헬퍼 메서드 테스트...")
        expanded_queries = workflow._get_expanded_queries(test_state, default_query=test_state['query'])
        logger.info(f"✅ expanded_queries 생성됨:")
        logger.info(f"   - original: {expanded_queries.get('original', 'N/A')}")
        logger.info(f"   - all_queries: {expanded_queries.get('all_queries', [])}")
        logger.info(f"   - variations: {len(expanded_queries.get('variations', []))}개")
        
        # _validate_expanded_queries 헬퍼 메서드 테스트
        logger.info("\n🔄 _validate_expanded_queries 헬퍼 메서드 테스트...")
        validated_queries = workflow._validate_expanded_queries(expanded_queries, test_state['query'])
        logger.info(f"✅ expanded_queries 검증 완료:")
        logger.info(f"   - 필수 필드 확인: original={bool(validated_queries.get('original'))}, "
                   f"all_queries={bool(validated_queries.get('all_queries'))}, "
                   f"variations={'variations' in validated_queries}, "
                   f"related_keywords={'related_keywords' in validated_queries}")
        
        # 의미적 검색 변형 서브노드 실행 (expanded_queries 없음)
        logger.info("\n🔄 의미적 검색 변형 서브노드 실행 (expanded_queries 없음)...")
        result_state = workflow.semantic_search_variations_subnode(test_state)
        
        # 결과 확인
        semantic_results = result_state.get("semantic_results", [])
        semantic_count = result_state.get("semantic_count", 0)
        
        logger.info(f"\n✅ 의미적 검색 완료 (폴백 로직):")
        logger.info(f"   - 결과 수: {semantic_count}")
        logger.info(f"   - 고유 결과 수: {len(semantic_results)}")
        
        # 키워드 검색 서브노드 실행 (expanded_queries 없음)
        logger.info("\n🔄 키워드 검색 서브노드 실행 (expanded_queries 없음)...")
        result_state = workflow.keyword_search_subnode(test_state)
        
        keyword_results = result_state.get("keyword_results", [])
        keyword_count = result_state.get("keyword_count", 0)
        
        logger.info(f"\n✅ 키워드 검색 완료 (폴백 로직):")
        logger.info(f"   - 결과 수: {keyword_count}")
        logger.info(f"   - 고유 결과 수: {len(keyword_results)}")
        
        logger.info("\n✅ expanded_queries가 없는 경우 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_parallel_search_failure():
    """병렬 처리 실패 시나리오 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 7: 병렬 처리 실패 시나리오")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 State 생성
        test_state: LegalWorkflowState = {
            "input": {
                "query": "손해배상 청구",
                "session_id": "test_session"
            },
            "query": "손해배상 청구",
            "query_type": "legal_question",
            "legal_field": "civil",
            "expanded_queries": {
                "original": "손해배상 청구",
                "all_queries": ["손해배상 청구"],
                "variations": []
            },
            "search_params": {
                "semantic_k": 5,
                "similarity_threshold": 0.5,
                "keyword_limit": 5
            },
            "search": {},
            "common": {
                "processing_time": 0.0,
                "tokens_used": 0
            }
        }
        
        logger.info(f"\n📝 테스트 쿼리: {test_state['query']}")
        logger.info("📝 병렬 처리 실패 시나리오 테스트 (폴백 로직 확인)")
        
        # execute_searches_parallel 노드 실행
        logger.info("\n🔄 execute_searches_parallel 노드 실행 중...")
        result_state = workflow.execute_searches_parallel(test_state)
        
        # 결과 확인
        semantic_results = result_state.get("semantic_results", [])
        keyword_results = result_state.get("keyword_results", [])
        semantic_count = result_state.get("semantic_count", 0)
        keyword_count = result_state.get("keyword_count", 0)
        merged_documents = result_state.get("merged_documents", [])
        retrieved_docs = result_state.get("retrieved_docs", [])
        
        logger.info(f"\n✅ 병렬 처리 완료:")
        logger.info(f"   - 의미적 검색 결과: {semantic_count}개")
        logger.info(f"   - 키워드 검색 결과: {keyword_count}개")
        logger.info(f"   - 통합된 문서: {len(merged_documents)}개")
        logger.info(f"   - retrieved_docs: {len(retrieved_docs)}개")
        
        # 폴백 로직 확인
        if semantic_count == 0 and keyword_count == 0:
            logger.warning("⚠️ 두 검색 모두 결과가 없음 (폴백 로직 확인 필요)")
        elif semantic_count == 0:
            logger.info("ℹ️ 의미적 검색 실패, 키워드 검색 결과만 사용 (폴백 로직 작동)")
        elif keyword_count == 0:
            logger.info("ℹ️ 키워드 검색 실패, 의미적 검색 결과만 사용 (폴백 로직 작동)")
        else:
            logger.info("✅ 두 검색 모두 성공")
        
        logger.info("\n✅ 병렬 처리 실패 시나리오 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def test_state_validation_failure():
    """State 검증 실패 케이스 테스트"""
    try:
        from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            from lawfirm_langgraph.core.utils.langgraph_config import LangGraphConfig
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 8: State 검증 실패 케이스")
        logger.info("=" * 80)
        
        # 워크플로우 초기화
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        logger.info("✅ EnhancedLegalQuestionWorkflow 초기화 완료")
        
        # 테스트 케이스 1: expanded_queries가 비어있는 딕셔너리
        logger.info("\n📝 테스트 케이스 1: expanded_queries가 비어있는 딕셔너리")
        test_state_1: LegalWorkflowState = {
            "query": "손해배상 청구",
            "expanded_queries": {},
            "search": {},
            "common": {"processing_time": 0.0, "tokens_used": 0}
        }
        
        expanded_queries_1 = workflow._get_expanded_queries(test_state_1, default_query="손해배상 청구")
        validated_queries_1 = workflow._validate_expanded_queries(expanded_queries_1, "손해배상 청구")
        logger.info(f"✅ 검증 완료: original={validated_queries_1.get('original')}, "
                   f"all_queries={len(validated_queries_1.get('all_queries', []))}개")
        
        # 테스트 케이스 2: expanded_queries에 필수 필드가 없음
        logger.info("\n📝 테스트 케이스 2: expanded_queries에 필수 필드가 없음")
        test_state_2: LegalWorkflowState = {
            "query": "손해배상 청구",
            "expanded_queries": {
                "variations": [{"query": "테스트", "type": "test"}]
                # original, all_queries 등 필수 필드 없음
            },
            "search": {},
            "common": {"processing_time": 0.0, "tokens_used": 0}
        }
        
        expanded_queries_2 = workflow._get_expanded_queries(test_state_2, default_query="손해배상 청구")
        validated_queries_2 = workflow._validate_expanded_queries(expanded_queries_2, "손해배상 청구")
        logger.info(f"✅ 검증 완료: original={validated_queries_2.get('original')}, "
                   f"all_queries={len(validated_queries_2.get('all_queries', []))}개, "
                   f"variations={len(validated_queries_2.get('variations', []))}개")
        
        # 테스트 케이스 3: expanded_queries가 None
        logger.info("\n📝 테스트 케이스 3: expanded_queries가 None")
        test_state_3: LegalWorkflowState = {
            "query": "손해배상 청구",
            "expanded_queries": None,
            "search": {},
            "common": {"processing_time": 0.0, "tokens_used": 0}
        }
        
        expanded_queries_3 = workflow._get_expanded_queries(test_state_3, default_query="손해배상 청구")
        validated_queries_3 = workflow._validate_expanded_queries(expanded_queries_3, "손해배상 청구")
        logger.info(f"✅ 검증 완료: original={validated_queries_3.get('original')}, "
                   f"all_queries={len(validated_queries_3.get('all_queries', []))}개")
        
        # 테스트 케이스 4: expanded_queries가 문자열 (잘못된 타입)
        logger.info("\n📝 테스트 케이스 4: expanded_queries가 문자열 (잘못된 타입)")
        test_state_4: LegalWorkflowState = {
            "query": "손해배상 청구",
            "expanded_queries": "invalid_type",
            "search": {},
            "common": {"processing_time": 0.0, "tokens_used": 0}
        }
        
        expanded_queries_4 = workflow._get_expanded_queries(test_state_4, default_query="손해배상 청구")
        validated_queries_4 = workflow._validate_expanded_queries(expanded_queries_4, "손해배상 청구")
        logger.info(f"✅ 검증 완료: original={validated_queries_4.get('original')}, "
                   f"all_queries={len(validated_queries_4.get('all_queries', []))}개")
        
        logger.info("\n✅ State 검증 실패 케이스 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("쿼리 확장 검색 기능 테스트 시작")
    logger.info("=" * 80)
    
    results = []
    
    # 테스트 1: SemanticSearchEngineV2 쿼리 확장 검색
    results.append(("SemanticSearchEngineV2 쿼리 확장 검색", test_semantic_search_engine_query_expansion()))
    
    # 테스트 2: 쿼리 확장 서브노드
    results.append(("쿼리 확장 서브노드", test_query_expansion_subnode()))
    
    # 테스트 3: 의미적 검색 변형 서브노드
    results.append(("의미적 검색 변형 서브노드", test_semantic_search_variations_subnode()))
    
    # 테스트 4: 키워드 검색 서브노드
    results.append(("키워드 검색 서브노드", test_keyword_search_subnode()))
    
    # 테스트 5: 결과 통합 서브노드
    results.append(("결과 통합 서브노드", test_result_merger_subnode()))
    
    # 테스트 6: expanded_queries가 없는 경우
    results.append(("expanded_queries가 없는 경우", test_expanded_queries_missing()))
    
    # 테스트 7: 병렬 처리 실패 시나리오
    results.append(("병렬 처리 실패 시나리오", test_parallel_search_failure()))
    
    # 테스트 8: State 검증 실패 케이스
    results.append(("State 검증 실패 케이스", test_state_validation_failure()))
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

