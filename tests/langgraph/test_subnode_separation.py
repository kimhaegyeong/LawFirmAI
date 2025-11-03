# -*- coding: utf-8 -*-
"""
?œë¸Œ?¸ë“œ ë¶„ë¦¬ ?ŒìŠ¤??(expand_keywords + prepare_search_query)
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


async def test_expand_keywords_node():
    """expand_keywords ?¸ë“œ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??1: expand_keywords ?¸ë“œ ?™ì‘ ?•ì¸")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "ê³„ì•½ ?´ì? ?”ê±´"
    print(f"\nì§ˆë¬¸: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_expand_keywords")
    elapsed = time.time() - start
    
    # processing_steps ?•ì¸
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # expand_keywords ?¸ë“œ ?¤í–‰ ?•ì¸ (???“ì? ë²”ìœ„ë¡?ê²€??
    has_keyword_expansion = any(
        "?¤ì›Œ???•ì¥" in step or 
        "?¤ì›Œ?? in step and "?•ì¥" in step or
        "expand_keywords" in step.lower()
        for step in step_texts
    )
    
    # extracted_keywords ?•ì¸
    extracted_keywords = result.get("extracted_keywords", [])
    
    print(f"\n[ê²°ê³¼]")
    print(f"  ?œê°„: {elapsed:.2f}ì´?)
    print(f"  ?¤ì›Œ???•ì¥ ?¨ê³„ ?¬í•¨: {has_keyword_expansion}")
    print(f"  ì¶”ì¶œ???¤ì›Œ???? {len(extracted_keywords)}ê°?)
    print(f"  ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)
    
    if extracted_keywords:
        print(f"  ?¤ì›Œ???ˆì‹œ: {extracted_keywords[:5]}")
    
    success = has_keyword_expansion and len(extracted_keywords) > 0
    
    if success:
        print("  ??[PASS] expand_keywords ?¸ë“œ ?•ìƒ ?‘ë™")
    else:
        print("  ??[FAIL] expand_keywords ?¸ë“œ ?•ì¸ ?¤íŒ¨")
        print(f"        ?¤ì›Œ???•ì¥ ?¬í•¨: {has_keyword_expansion}, ?¤ì›Œ???? {len(extracted_keywords)}")
    
    return success


async def test_prepare_search_query_node():
    """prepare_search_query ?¸ë“œ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??2: prepare_search_query ?¸ë“œ ?™ì‘ ?•ì¸")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "ë¯¼ë²• ??11ì¡°ì˜ ?´ìš©???Œë ¤ì£¼ì„¸??
    print(f"\nì§ˆë¬¸: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_prepare_query")
    elapsed = time.time() - start
    
    # processing_steps ?•ì¸
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # prepare_search_query ?¸ë“œ ?¤í–‰ ?•ì¸ (???“ì? ë²”ìœ„ë¡?ê²€??
    has_search_query_prep = any(
        "ê²€??ì¿¼ë¦¬ ì¤€ë¹? in step or 
        "ì¿¼ë¦¬ ì¤€ë¹? in step or
        "search_query" in step.lower() or
        "ìµœì ?”ëœ ì¿¼ë¦¬" in step
        for step in step_texts
    )
    
    # optimized_queries ?•ì¸
    optimized_queries = result.get("optimized_queries", {})
    search_query = result.get("search_query", "")
    search_params = result.get("search_params", {})
    
    print(f"\n[ê²°ê³¼]")
    print(f"  ?œê°„: {elapsed:.2f}ì´?)
    print(f"  ê²€??ì¿¼ë¦¬ ì¤€ë¹??¨ê³„ ?¬í•¨: {has_search_query_prep}")
    print(f"  ìµœì ?”ëœ ì¿¼ë¦¬ ?ì„±: {bool(optimized_queries)}")
    print(f"  ê²€??ì¿¼ë¦¬: {search_query[:50] if search_query else 'N/A'}...")
    print(f"  ê²€???Œë¼ë¯¸í„°: {bool(search_params)}")
    print(f"  ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)
    
    if optimized_queries:
        semantic_query = optimized_queries.get("semantic_query", "")
        print(f"  ?˜ë???ì¿¼ë¦¬: {semantic_query[:50] if semantic_query else 'N/A'}...")
    
    success = has_search_query_prep and bool(optimized_queries) and bool(search_params)
    
    if success:
        print("  ??[PASS] prepare_search_query ?¸ë“œ ?•ìƒ ?‘ë™")
    else:
        print("  ??[FAIL] prepare_search_query ?¸ë“œ ?•ì¸ ?¤íŒ¨")
        print(f"        ì¿¼ë¦¬ ì¤€ë¹??¬í•¨: {has_search_query_prep}, "
              f"ìµœì ?”ëœ ì¿¼ë¦¬: {bool(optimized_queries)}, "
              f"ê²€???Œë¼ë¯¸í„°: {bool(search_params)}")
    
    return success


async def test_unified_classification():
    """?µí•© LLM ë¶„ë¥˜ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??3: ?µí•© LLM ë¶„ë¥˜ (ì§ˆë¬¸ ? í˜• + ë³µì¡?? ?™ì‘ ?•ì¸")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    test_cases = [
        ("?ˆë…•?˜ì„¸??, "simple"),
        ("ë¯¼ë²• ??11ì¡?, "moderate"),
        ("ê³„ì•½ ?´ì??€ ?´ì œ??ì°¨ì´", "complex"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_complexity in test_cases:
        print(f"\n?“ ì§ˆë¬¸: {query}")
        print(f"  ?ˆìƒ ë³µì¡?? {expected_complexity}")
        
        try:
            start = time.time()
            result = await workflow_service.process_query(query, f"test_session_{passed}")
            elapsed = time.time() - start
            
            actual_complexity = result.get("query_complexity", "unknown")
            needs_search = result.get("needs_search", True)
            query_type = result.get("query_type", "unknown")
            
            print(f"  ?¤ì œ ë³µì¡?? {actual_complexity}")
            print(f"  ì§ˆë¬¸ ? í˜•: {query_type}")
            print(f"  ê²€???„ìš”: {needs_search}")
            print(f"  ?‘ë‹µ ?œê°„: {elapsed:.2f}ì´?)
            
            # ë³µì¡???¼ì¹˜ ?•ì¸
            if actual_complexity == expected_complexity:
                print("  ??[PASS] ë³µì¡???¼ì¹˜")
                passed += 1
            else:
                print(f"  ? ï¸  ë³µì¡??ë¶ˆì¼ì¹?(?ˆìƒ: {expected_complexity}, ?¤ì œ: {actual_complexity})")
                failed += 1
            
        except Exception as e:
            print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
            logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")
            failed += 1
    
    print(f"\n?“Š ê²°ê³¼: {passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")
    return failed == 0


async def test_subnode_sequence():
    """?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??4: ?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?•ì¸ (expand_keywords ??prepare_search_query)")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "ê³„ì•½ ?´ì? ?”ê±´ê³??ˆì°¨"
    print(f"\nì§ˆë¬¸: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_sequence")
    elapsed = time.time() - start
    
    # processing_steps ?•ì¸
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # ?¨ê³„ ?œì„œ ?•ì¸ (???“ì? ë²”ìœ„ë¡?ê²€??
    keyword_expansion_idx = -1
    search_query_prep_idx = -1
    
    for i, step in enumerate(step_texts):
        step_lower = step.lower()
        if "?¤ì›Œ?? in step and ("?•ì¥" in step or "expansion" in step_lower):
            keyword_expansion_idx = i
        if ("ê²€??ì¿¼ë¦¬" in step or "ì¿¼ë¦¬ ì¤€ë¹? in step or 
            "search_query" in step_lower or "ìµœì ?”ëœ ì¿¼ë¦¬" in step):
            search_query_prep_idx = i
    
    print(f"\n[ê²°ê³¼]")
    print(f"  ?œê°„: {elapsed:.2f}ì´?)
    print(f"  ?¤ì›Œ???•ì¥ ?¨ê³„ ?¸ë±?? {keyword_expansion_idx}")
    print(f"  ê²€??ì¿¼ë¦¬ ì¤€ë¹??¨ê³„ ?¸ë±?? {search_query_prep_idx}")
    print(f"  ì´?ì²˜ë¦¬ ?¨ê³„ ?? {len(processing_steps)}ê°?)
    
    # ?œì„œ ?•ì¸: ?¤ì›Œ???•ì¥??ê²€??ì¿¼ë¦¬ ì¤€ë¹„ë³´??ë¨¼ì? ?¤í–‰?˜ì–´????
    correct_sequence = (
        keyword_expansion_idx >= 0 and 
        search_query_prep_idx >= 0 and 
        keyword_expansion_idx < search_query_prep_idx
    )
    
    # ê²°ê³¼ ?•ì¸
    extracted_keywords = result.get("extracted_keywords", [])
    optimized_queries = result.get("optimized_queries", {})
    search_query = result.get("search_query", "")
    
    has_keywords = len(extracted_keywords) > 0
    has_optimized = bool(optimized_queries) and bool(search_query)
    
    print(f"  ?¤ì›Œ???•ì¥ ê²°ê³¼: {has_keywords} (?¤ì›Œ??{len(extracted_keywords)}ê°?")
    print(f"  ì¿¼ë¦¬ ìµœì ??ê²°ê³¼: {has_optimized}")
    
    success = correct_sequence and has_keywords and has_optimized
    
    if success:
        print("  ??[PASS] ?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?•ìƒ")
    else:
        print("  ??[FAIL] ?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?•ì¸ ?¤íŒ¨")
        print(f"        ?¬ë°”ë¥??œì„œ: {correct_sequence}, "
              f"?¤ì›Œ???ˆìŒ: {has_keywords}, "
              f"ìµœì ?”ë¨: {has_optimized}")
    
    return success


async def test_end_to_end_workflow():
    """?„ì²´ ?Œí¬?Œë¡œ???”ë“œ-???”ë“œ ?ŒìŠ¤??""
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??5: ?„ì²´ ?Œí¬?Œë¡œ???”ë“œ-???”ë“œ ?ŒìŠ¤??)
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    test_queries = [
        "?ˆë…•?˜ì„¸??,
        "ë¯¼ë²• ??11ì¡°ì˜ ?´ìš©???Œë ¤ì£¼ì„¸??,
        "ê³„ì•½ ?´ì??€ ?´ì œ??ì°¨ì´??ë¬´ì—‡?¸ê???",
    ]
    
    passed = 0
    failed = 0
    
    for query in test_queries:
        print(f"\n?“ ì§ˆë¬¸: {query}")
        
        try:
            start = time.time()
            result = await workflow_service.process_query(query, f"test_session_e2e_{passed}")
            elapsed = time.time() - start
            
            answer = result.get("answer", "")
            query_complexity = result.get("query_complexity", "unknown")
            needs_search = result.get("needs_search", True)
            extracted_keywords = result.get("extracted_keywords", [])
            optimized_queries = result.get("optimized_queries", {})
            
            print(f"  ?±ï¸  ?‘ë‹µ ?œê°„: {elapsed:.2f}ì´?)
            print(f"  ?“Š ë³µì¡?? {query_complexity}")
            print(f"  ?” ê²€???„ìš”: {needs_search}")
            print(f"  ?“ ?µë? ê¸¸ì´: {len(answer)}??)
            print(f"  ?”‘ ?¤ì›Œ???? {len(extracted_keywords)}ê°?)
            print(f"  ?” ìµœì ?”ëœ ì¿¼ë¦¬: {bool(optimized_queries)}")
            
            # ê¸°ë³¸ ê²€ì¦?
            has_answer = len(answer) > 0
            has_complexity = query_complexity != "unknown"
            
            if has_answer and has_complexity:
                print("  ??[PASS] ?„ì²´ ?Œí¬?Œë¡œ???•ìƒ ?‘ë™")
                passed += 1
            else:
                print("  ??[FAIL] ?„ì²´ ?Œí¬?Œë¡œ??ê²€ì¦??¤íŒ¨")
                failed += 1
                
        except Exception as e:
            print(f"  ???¤ë¥˜ ë°œìƒ: {e}")
            logger.exception(f"?ŒìŠ¤???¤íŒ¨: {query}")
            failed += 1
    
    print(f"\n?“Š ê²°ê³¼: {passed}ê°??µê³¼, {failed}ê°??¤íŒ¨")
    return failed == 0


async def main():
    """ëª¨ë“  ?ŒìŠ¤???¤í–‰"""
    print("=" * 80)
    print("?œë¸Œ?¸ë“œ ë¶„ë¦¬ ë°??µí•© LLM ë¶„ë¥˜ ?ŒìŠ¤??)
    print("=" * 80)
    
    results = {}
    
    # ?ŒìŠ¤??1: expand_keywords ?¸ë“œ
    try:
        results["expand_keywords"] = await test_expand_keywords_node()
    except Exception as e:
        print(f"??expand_keywords ?ŒìŠ¤???¤ë¥˜: {e}")
        logger.exception("expand_keywords ?ŒìŠ¤???¤íŒ¨")
        results["expand_keywords"] = False
    
    # ?ŒìŠ¤??2: prepare_search_query ?¸ë“œ
    try:
        results["prepare_search_query"] = await test_prepare_search_query_node()
    except Exception as e:
        print(f"??prepare_search_query ?ŒìŠ¤???¤ë¥˜: {e}")
        logger.exception("prepare_search_query ?ŒìŠ¤???¤íŒ¨")
        results["prepare_search_query"] = False
    
    # ?ŒìŠ¤??3: ?µí•© LLM ë¶„ë¥˜
    try:
        results["unified_classification"] = await test_unified_classification()
    except Exception as e:
        print(f"???µí•© LLM ë¶„ë¥˜ ?ŒìŠ¤???¤ë¥˜: {e}")
        logger.exception("?µí•© LLM ë¶„ë¥˜ ?ŒìŠ¤???¤íŒ¨")
        results["unified_classification"] = False
    
    # ?ŒìŠ¤??4: ?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰
    try:
        results["subnode_sequence"] = await test_subnode_sequence()
    except Exception as e:
        print(f"???œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?ŒìŠ¤???¤ë¥˜: {e}")
        logger.exception("?œë¸Œ?¸ë“œ ?œì°¨ ?¤í–‰ ?ŒìŠ¤???¤íŒ¨")
        results["subnode_sequence"] = False
    
    # ?ŒìŠ¤??5: ?”ë“œ-???”ë“œ
    try:
        results["end_to_end"] = await test_end_to_end_workflow()
    except Exception as e:
        print(f"???”ë“œ-???”ë“œ ?ŒìŠ¤???¤ë¥˜: {e}")
        logger.exception("?”ë“œ-???”ë“œ ?ŒìŠ¤???¤íŒ¨")
        results["end_to_end"] = False
    
    # ìµœì¢… ê²°ê³¼
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??ìµœì¢… ê²°ê³¼")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "??PASS" if passed else "??FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n?“Š ì´ê³„: {total_passed}/{total_tests} ?ŒìŠ¤???µê³¼")
    
    if total_passed == total_tests:
        print("??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
        return 0
    else:
        print("? ï¸ ?¼ë? ?ŒìŠ¤???¤íŒ¨")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

