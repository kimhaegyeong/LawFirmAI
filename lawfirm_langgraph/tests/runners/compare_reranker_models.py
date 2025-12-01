# -*- coding: utf-8 -*-
"""
Cross-Encoder 모델 비교 테스트 스크립트

두 개의 Cross-Encoder 모델을 비교하여 성능을 평가합니다.

Usage:
    python lawfirm_langgraph/tests/runners/compare_reranker_models.py "질의 내용"
    python lawfirm_langgraph/tests/runners/compare_reranker_models.py  # 기본 질의 사용
"""

import sys
import os
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# UTF-8 인코딩 설정 (Windows 호환)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
runners_dir = script_dir.parent
tests_dir = runners_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

# 로거 설정
from lawfirm_langgraph.core.utils.logger import get_logger
logger = get_logger(__name__)

# 테스트할 모델 목록
MODELS_TO_COMPARE = [
    "Dongjin-kr/ko-reranker",
    "dragonkue/bge-reranker-v2-m3-ko"
]

# 기본 테스트 질의
DEFAULT_QUERY = "계약 해지 사유에 대해 알려주세요"


def compare_models_on_query(query: str, test_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    두 모델을 동일한 쿼리와 문서에 대해 비교
    
    Args:
        query: 테스트 쿼리
        test_documents: 테스트 문서 리스트
    
    Returns:
        비교 결과 딕셔너리
    """
    from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
    
    results = {
        "query": query,
        "num_documents": len(test_documents),
        "models": {}
    }
    
    for model_name in MODELS_TO_COMPARE:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 Testing model: {model_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # 모델 로드 시간 측정
            load_start = time.time()
            ranker = ResultRanker(
                use_cross_encoder=True,
                cross_encoder_model_name=model_name
            )
            
            # 모델 강제 로드
            if ranker._ensure_cross_encoder_loaded():
                load_time = time.time() - load_start
                logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
            else:
                logger.error(f"❌ Failed to load model: {model_name}")
                results["models"][model_name] = {
                    "status": "failed",
                    "error": "Model loading failed"
                }
                continue
            
            # 각 문서에 대해 점수 계산
            scores = []
            score_start = time.time()
            
            for i, doc in enumerate(test_documents):
                doc_text = doc.get("text") or doc.get("content", "")
                if not doc_text:
                    continue
                
                # 텍스트 전처리
                processed_text = ranker._preprocess_text_for_cross_encoder(doc_text, max_length=512)
                
                if not processed_text:
                    continue
                
                # Cross-Encoder 점수 계산
                try:
                    from sentence_transformers import CrossEncoder
                    pairs = [[query, processed_text]]
                    doc_scores = ranker.cross_encoder.predict(pairs, batch_size=1, show_progress_bar=False)
                    raw_score = float(doc_scores[0])
                    
                    scores.append({
                        "doc_index": i,
                        "doc_type": doc.get("type", "unknown"),
                        "raw_score": raw_score,
                        "text_preview": processed_text[:100] + "..." if len(processed_text) > 100 else processed_text,
                        "original_relevance_score": doc.get("relevance_score", 0.0),
                        "original_rank_score": doc.get("rank_score", 0.0)
                    })
                    
                    logger.debug(
                        f"  Document {i+1}/{len(test_documents)}: "
                        f"type={doc.get('type', 'unknown')}, "
                        f"score={raw_score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"  Failed to score document {i+1}: {e}")
                    continue
            
            score_time = time.time() - score_start
            
            # 통계 계산
            if scores:
                raw_scores = [s["raw_score"] for s in scores]
                avg_score = sum(raw_scores) / len(raw_scores)
                max_score = max(raw_scores)
                min_score = min(raw_scores)
                
                # 판례 문서 점수 통계
                precedent_scores = [s["raw_score"] for s in scores if "precedent" in s.get("doc_type", "").lower()]
                precedent_avg = sum(precedent_scores) / len(precedent_scores) if precedent_scores else 0.0
                
                # 법령 문서 점수 통계
                statute_scores = [s["raw_score"] for s in scores if "statute" in s.get("doc_type", "").lower()]
                statute_avg = sum(statute_scores) / len(statute_scores) if statute_scores else 0.0
                
                results["models"][model_name] = {
                    "status": "success",
                    "load_time": load_time,
                    "score_time": score_time,
                    "num_scored": len(scores),
                    "scores": scores,
                    "statistics": {
                        "avg_score": avg_score,
                        "max_score": max_score,
                        "min_score": min_score,
                        "precedent_avg": precedent_avg,
                        "num_precedents": len(precedent_scores),
                        "statute_avg": statute_avg,
                        "num_statutes": len(statute_scores)
                    }
                }
                
                logger.info(f"\n📊 Statistics for {model_name}:")
                logger.info(f"  Average score: {avg_score:.4f}")
                logger.info(f"  Max score: {max_score:.4f}")
                logger.info(f"  Min score: {min_score:.4f}")
                logger.info(f"  Precedent average: {precedent_avg:.4f} ({len(precedent_scores)} documents)")
                logger.info(f"  Scoring time: {score_time:.2f} seconds")
            else:
                results["models"][model_name] = {
                    "status": "failed",
                    "error": "No documents scored"
                }
                
        except Exception as e:
            logger.error(f"❌ Error testing model {model_name}: {e}", exc_info=True)
            results["models"][model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    return results


def print_comparison_summary(results: Dict[str, Any]):
    """비교 결과 요약 출력"""
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Query: {results['query']}")
    logger.info(f"Number of documents: {results['num_documents']}\n")
    
    # 각 모델별 결과 출력
    for model_name in MODELS_TO_COMPARE:
        if model_name not in results["models"]:
            continue
        
        model_result = results["models"][model_name]
        
        if model_result["status"] == "success":
            stats = model_result["statistics"]
            logger.info(f"\n🔍 {model_name}:")
            logger.info(f"  Status: ✅ Success")
            logger.info(f"  Load time: {model_result['load_time']:.2f}s")
            logger.info(f"  Score time: {model_result['score_time']:.2f}s")
            logger.info(f"  Average score: {stats['avg_score']:.4f}")
            logger.info(f"  Max score: {stats['max_score']:.4f}")
            logger.info(f"  Min score: {stats['min_score']:.4f}")
            logger.info(f"  Statute average: {stats['statute_avg']:.4f} ({stats['num_statutes']} documents)")
            logger.info(f"  Precedent average: {stats['precedent_avg']:.4f} ({stats['num_precedents']} documents)")
        else:
            logger.info(f"\n❌ {model_name}:")
            logger.info(f"  Status: Failed")
            logger.info(f"  Error: {model_result.get('error', 'Unknown error')}")
    
    # 비교 분석
    successful_models = {
        name: result for name, result in results["models"].items()
        if result["status"] == "success"
    }
    
    if len(successful_models) >= 2:
        logger.info(f"\n{'='*80}")
        logger.info("📈 COMPARISON ANALYSIS")
        logger.info(f"{'='*80}\n")
        
        model_names = list(successful_models.keys())
        model1_name = model_names[0]
        model2_name = model_names[1]
        
        model1_stats = successful_models[model1_name]["statistics"]
        model2_stats = successful_models[model2_name]["statistics"]
        
        logger.info(f"Average Score Comparison:")
        logger.info(f"  {model1_name}: {model1_stats['avg_score']:.4f}")
        logger.info(f"  {model2_name}: {model2_stats['avg_score']:.4f}")
        diff = model2_stats['avg_score'] - model1_stats['avg_score']
        logger.info(f"  Difference: {diff:+.4f} ({diff/model1_stats['avg_score']*100:+.1f}%)")
        
        logger.info(f"\nStatute Score Comparison:")
        logger.info(f"  {model1_name}: {model1_stats['statute_avg']:.4f}")
        logger.info(f"  {model2_name}: {model2_stats['statute_avg']:.4f}")
        statute_diff = model2_stats['statute_avg'] - model1_stats['statute_avg']
        if model1_stats['statute_avg'] > 0:
            logger.info(f"  Difference: {statute_diff:+.4f} ({statute_diff/model1_stats['statute_avg']*100:+.1f}%)")
        
        logger.info(f"\nPrecedent Score Comparison:")
        logger.info(f"  {model1_name}: {model1_stats['precedent_avg']:.4f}")
        logger.info(f"  {model2_name}: {model2_stats['precedent_avg']:.4f}")
        precedent_diff = model2_stats['precedent_avg'] - model1_stats['precedent_avg']
        if model1_stats['precedent_avg'] > 0:
            logger.info(f"  Difference: {precedent_diff:+.4f} ({precedent_diff/model1_stats['precedent_avg']*100:+.1f}%)")
        
        logger.info(f"\nPerformance Comparison:")
        logger.info(f"  {model1_name}: Load={successful_models[model1_name]['load_time']:.2f}s, Score={successful_models[model1_name]['score_time']:.2f}s")
        logger.info(f"  {model2_name}: Load={successful_models[model2_name]['load_time']:.2f}s, Score={successful_models[model2_name]['score_time']:.2f}s")


async def get_test_documents_from_query(query: str) -> List[Dict[str, Any]]:
    """
    실제 검색을 통해 테스트 문서 가져오기 (검색 커넥터 직접 사용)
    
    Args:
        query: 검색 쿼리
    
    Returns:
        테스트 문서 리스트
    """
    try:
        from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
        
        logger.info(f"🔍 Fetching test documents for query: {query}")
        
        connector = LegalDataConnectorV2()
        
        # 🔥 수정: limit 파라미터 사용 (top_k가 아님)
        # 법령 검색
        statute_results = connector.search_statutes_fts(query, limit=5)
        
        # 판례 검색
        precedent_results = connector.search_cases_fts(query, limit=5)
        
        # 검색 결과를 문서 형식으로 변환
        test_documents = []
        
        # 법령 문서 추가
        for result in statute_results:
            doc = {
                "text": result.get("text") or result.get("content", ""),
                "type": "statute_article",
                "metadata": result.get("metadata", {}),
                "relevance_score": result.get("relevance_score", result.get("rank_score", 0.0)),
                "rank_score": result.get("rank_score", 0.0)
            }
            if doc["text"]:
                test_documents.append(doc)
        
        # 판례 문서 추가
        for result in precedent_results:
            doc = {
                "text": result.get("text") or result.get("content", ""),
                "type": "precedent_content",
                "metadata": result.get("metadata", {}),
                "relevance_score": result.get("relevance_score", result.get("rank_score", 0.0)),
                "rank_score": result.get("rank_score", 0.0)
            }
            if doc["text"]:
                test_documents.append(doc)
        
        logger.info(f"✅ Retrieved {len(test_documents)} test documents ({len(statute_results)} statutes, {len(precedent_results)} precedents)")
        return test_documents
        
    except Exception as e:
        logger.error(f"Failed to fetch test documents: {e}", exc_info=True)
        # 폴백: 샘플 문서 사용
        logger.warning("Using fallback sample documents")
        return [
            {
                "text": "【신 청 인】 <br/>【피신청인】 주식회사 신한은행외 1인 (소송대리인 법무법인 율촌외 5인)<br/>【주    문】<br/>1. 신청인이 피신청인 주식회사 신한은행을 위한 담보",
                "type": "precedent_content",
                "relevance_score": 0.0,
                "rank_score": 0.0
            },
            {
                "text": "계약의 해지는 당사자 일방의 의사표시로 계약을 소급하여 소멸시키는 행위를 말한다. 계약 해지의 사유로는 계약 위반, 불가능, 목적 달성 불가 등이 있다.",
                "type": "statute_article",
                "relevance_score": 0.0,
                "rank_score": 0.0
            },
            {
                "text": "계약 해지의 효과는 계약이 소급하여 소멸하는 것이며, 이미 이행된 급부에 대하여는 원상회복의무가 발생한다.",
                "type": "statute_article",
                "relevance_score": 0.0,
                "rank_score": 0.0
            }
        ]


async def main():
    """메인 함수"""
    # 질의 가져오기
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    
    logger.info(f"\n{'='*80}")
    logger.info("🚀 Cross-Encoder Model Comparison Test")
    logger.info(f"{'='*80}\n")
    logger.info(f"Query: {query}\n")
    
    # 테스트 문서 가져오기
    test_documents = await get_test_documents_from_query(query)
    
    if not test_documents:
        logger.error("No test documents available")
        return
    
    logger.info(f"Using {len(test_documents)} test documents\n")
    
    # 모델 비교 실행
    results = compare_models_on_query(query, test_documents)
    
    # 결과 요약 출력
    print_comparison_summary(results)
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Comparison test completed")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

