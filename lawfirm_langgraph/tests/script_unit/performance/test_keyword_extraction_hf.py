# -*- coding: utf-8 -*-
"""
HuggingFace 모델 기반 키워드 추출 테스트

Usage:
    python lawfirm_langgraph/tests/script_unit/performance/test_keyword_extraction_hf.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
performance_dir = script_dir.parent
unit_dir = performance_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

def test_keyword_extraction():
    """키워드 추출 테스트"""
    print("=" * 80)
    print("HuggingFace 모델 기반 키워드 추출 테스트")
    print("=" * 80)
    
    try:
        from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from core.utils.langgraph_config import LangGraphConfig
        
        # 설정 로드
        try:
            config = LangGraphConfig.from_env()
        except:
            from core.utils.config import Config
            config = Config()
        
        # 워크플로우 초기화
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        # 테스트 쿼리
        test_query = "계약 해지 사유에 대해 알려주세요"
        state = {
            "query": test_query,
            "query_type": "legal_advice",
            "legal_field": "민사법"
        }
        
        print(f"\n테스트 쿼리: {test_query}")
        print(f"HybridQueryProcessor 사용 가능: {workflow.hybrid_query_processor is not None}")
        
        # 키워드 추출 실행
        result = workflow.expand_keywords(state)
        
        # 결과 확인
        extracted_keywords = result.get("extracted_keywords", [])
        if not extracted_keywords:
            # 다른 위치에서 확인
            if "search" in result and isinstance(result["search"], dict):
                extracted_keywords = result["search"].get("extracted_keywords", [])
        
        print(f"\n{'='*80}")
        print(f"✅ 키워드 추출 완료: {len(extracted_keywords)}개")
        print(f"{'='*80}")
        if extracted_keywords:
            print(f"\n추출된 키워드:")
            for i, kw in enumerate(extracted_keywords[:15], 1):
                print(f"  {i}. {kw}")
        else:
            print("\n⚠️ 추출된 키워드가 없습니다.")
        
        if len(extracted_keywords) > 0:
            print(f"\n✅ 테스트 성공: {len(extracted_keywords)}개 키워드 추출됨")
            return True
        else:
            print(f"\n❌ 테스트 실패: 키워드가 추출되지 않음")
            return False
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("🚀 HuggingFace 모델 기반 키워드 추출 테스트 시작\n")
    
    success = test_keyword_extraction()
    
    if success:
        print("\n✅ 모든 테스트 완료")
        sys.exit(0)
    else:
        print("\n❌ 테스트 실패")
        sys.exit(1)

