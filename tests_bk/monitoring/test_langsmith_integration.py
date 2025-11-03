# -*- coding: utf-8 -*-
"""
LangSmith ?µí•© ?ŒìŠ¤??
ê°œì„ ??LangGraph ?Œí¬?Œë¡œ?°ë? LangSmithë¡?ëª¨ë‹ˆ?°ë§?˜ëŠ” ?ŒìŠ¤??
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ?˜ê²½ë³€???Œì¼ ë¡œë“œ ?•ì¸ (langgraph_config.py?ì„œ ?´ë? ì²˜ë¦¬??
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"???˜ê²½ë³€???Œì¼ ë¡œë“œ ?„ë£Œ: {env_path}")
    else:
        print(f"??.env ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {env_path}")
except ImportError:
    print("??python-dotenvê°€ ?¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ?? pip install python-dotenvë¡??¤ì¹˜?˜ì„¸??")
except Exception as e:
    print(f"???˜ê²½ë³€???Œì¼ ë¡œë“œ ì¤??¤ë¥˜ ë°œìƒ: {e}")

# ?„ìˆ˜ ?¨í‚¤ì§€ ?¤ì¹˜ ?•ì¸
missing_packages = []

def check_package(package_name, import_name=None, install_name=None):
    """?¨í‚¤ì§€ ?¤ì¹˜ ?•ì¸"""
    if import_name is None:
        import_name = package_name
    if install_name is None:
        install_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        missing_packages.append(install_name)
        print(f"??{package_name}ê°€ ?¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ??")
        print(f"  ?¤ì¹˜ ëª…ë ¹: pip install {install_name}")
        return False

# ?„ìˆ˜ ?¨í‚¤ì§€ ?•ì¸
print("\n?˜ê²½ ?¤ì • ?•ì¸ ì¤?..")
check_package("pydantic-settings", "pydantic_settings", "pydantic-settings")
check_package("numpy", "numpy", "numpy")
check_package("faiss", "faiss", "faiss-cpu")
check_package("structlog", "structlog", "structlog")
check_package("google-generativeai", "google.generativeai", "google-generativeai")

if missing_packages:
    print(f"\n??ì´?{len(missing_packages)}ê°??¨í‚¤ì§€ê°€ ?„ë½?˜ì—ˆ?µë‹ˆ??")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\n???¨í‚¤ì§€?¤ì„ ?¤ì¹˜?????ŒìŠ¤?¸ë? ?¤ì‹œ ?¤í–‰?˜ì„¸??\n")
    # sys.exit(1)  # ì£¼ì„ ì²˜ë¦¬: ?¤ì¹˜ ?ˆí•´???ŒìŠ¤??ì§„í–‰ ê°€?¥í•˜ê²?
else:
    print("??ëª¨ë“  ?„ìˆ˜ ?¨í‚¤ì§€ê°€ ?¤ì¹˜?˜ì–´ ?ˆìŠµ?ˆë‹¤.\n")

# LangGraph ê´€??import (?˜ê²½ë³€??ë¡œë“œ ??
from source.agents.workflow_service import (
    LangGraphWorkflowService,  # noqa: E402
)
from source.utils.langgraph_config import LangGraphConfig  # noqa: E402

# ?ŒìŠ¤???˜ê²½ ?¤ì •
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGGRAPH_CHECKPOINT_STORAGE"] = "memory"  # ë¹ ë¥¸ ?ŒìŠ¤?¸ë? ?„í•´ ë©”ëª¨ë¦??¬ìš©

# LangSmith ?œì„±??
# .env ?Œì¼?ì„œ ?¤ì •??ê°€?¸ì???LangChain ?˜ê²½ë³€?˜ë¡œ ë³€??
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "") or os.getenv("LANGCHAIN_API_KEY", "")
langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false") or os.getenv("LANGCHAIN_TRACING_V2", "false")
langsmith_project = os.getenv("LANGSMITH_PROJECT", "LawFirmAI-Test") or os.getenv("LANGCHAIN_PROJECT", "LawFirmAI-Test")

# LangChain ?˜ê²½ë³€?˜ë¡œ ?¤ì • (LangChain SDKê°€ ?¬ìš©??
os.environ["LANGCHAIN_TRACING_V2"] = "true" if langsmith_tracing.lower() in ["true", "1", "yes"] else "false"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# LangSmith ?˜ê²½ë³€?˜ë„ ?¤ì • (?˜ìœ„ ?¸í™˜??
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
if langsmith_project:
    os.environ["LANGSMITH_PROJECT"] = langsmith_project


class LangSmithIntegrationTest:
    """LangSmith ?µí•© ?ŒìŠ¤???´ë˜??""

    def __init__(self):
        self.config = LangGraphConfig()
        self.service = LangGraphWorkflowService(self.config)
        self.test_start_time = None

    def calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """?µë? ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°"""
        if not result:
            return 0.0

        score = 0.0
        max_score = 100

        # ?µë? ì¡´ì¬ (20??
        answer = result.get('answer', '')
        if answer:
            score += 10
            if len(answer) >= 50:
                score += 10

        # ? ë¢°??(30??
        confidence = result.get('confidence', 0.0)
        score += confidence * 30

        # ?ŒìŠ¤ ?œê³µ (25??
        sources_count = len(result.get('sources', []))
        if sources_count > 0:
            score += min(25, sources_count * 5)

        # ë²•ë¥  ì°¸ì¡° (15??
        legal_refs_count = len(result.get('legal_references', []))
        if legal_refs_count > 0:
            score += min(15, legal_refs_count * 5)

        # ?ëŸ¬ ?†ìŒ (10??
        errors_count = len(result.get('errors', []))
        if errors_count == 0:
            score += 10

        return round(score / max_score, 2)

    def validate_answer_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """?µë? ?ˆì§ˆ ê²€ì¦?""
        if not result:
            return {
                'valid': False,
                'has_answer': False,
                'overall_score': 0.0
            }

        quality = {
            'valid': True,
            'has_answer': bool(result.get('answer')),
            'answer_length_sufficient': len(result.get('answer', '')) >= 50,
            'has_sources': len(result.get('sources', [])) > 0,
            'has_legal_references': len(result.get('legal_references', [])) > 0,
            'confidence_threshold': result.get('confidence', 0) >= 0.5,
            'no_errors': len(result.get('errors', [])) == 0,
            'processing_time_reasonable': result.get('processing_time', 0) < 60
        }

        quality['overall_score'] = self.calculate_quality_score(result)
        quality['valid'] = quality['overall_score'] >= 0.5

        return quality

    async def save_results(self, results: List[Dict[str, Any]]) -> str:
        """?ŒìŠ¤??ê²°ê³¼ë¥?JSON ?Œì¼ë¡??€??""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = project_root / 'tests' / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        filename = result_dir / f'langsmith_test_{timestamp}.json'

        # ê²°ê³¼ ?”ì•½ ?ì„±
        summary = {
            'timestamp': timestamp,
            'total_queries': len(results),
            'successful_queries': len([r for r in results if r.get('result')]),
            'failed_queries': len([r for r in results if not r.get('result')]),
            'langsmith_enabled': os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            'test_duration': time.time() - self.test_start_time if self.test_start_time else 0,
            'results': results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return str(filename)

    async def run_single_query(self, query: str, session_id: str, query_number: int) -> Optional[Dict[str, Any]]:
        """?¨ì¼ ì§ˆì˜ ?¤í–‰"""
        print(f"\n{'='*80}")
        print(f"ì§ˆì˜ #{query_number}: {query}")
        print(f"{'='*80}")

        try:
            # LangGraph ?Œí¬?Œë¡œ???¤í–‰
            result = await self.service.process_query(
                query=query,
                session_id=session_id
            )

            # ?ˆì§ˆ ê²€ì¦?
            quality = self.validate_answer_quality(result)

            # ê²°ê³¼ ì¶œë ¥
            print("\n??ì²˜ë¦¬ ?„ë£Œ")
            print(f"  ?µë? ê¸¸ì´: {len(result.get('answer', ''))}??)
            print(f"  ? ë¢°?? {result.get('confidence', 0):.2%}")
            print(f"  ?ŒìŠ¤ ?? {len(result.get('sources', []))}ê°?)
            print(f"  ë²•ë¥  ì°¸ì¡° ?? {len(result.get('legal_references', []))}ê°?)
            print(f"  ì²˜ë¦¬ ?¨ê³„ ?? {len(result.get('processing_steps', []))}ê°?)
            print(f"  ì²˜ë¦¬ ?œê°„: {result.get('processing_time', 0):.2f}ì´?)
            print(f"  ?ˆì§ˆ ?ìˆ˜: {quality.get('overall_score', 0):.2%}")

            # ?¤ì›Œ???•ì¥ ?•ë³´ (metadata?ì„œ)
            metadata = result.get('metadata', {})
            if 'ai_keyword_expansion' in metadata:
                expansion = metadata['ai_keyword_expansion']
                print(f"  AI ?¤ì›Œ???•ì¥: {expansion.get('method', 'N/A')}")
                print(f"    - ?ë³¸ ?¤ì›Œ?? {len(expansion.get('original_keywords', []))}ê°?)
                print(f"    - ?•ì¥ ?¤ì›Œ?? {len(expansion.get('expanded_keywords', []))}ê°?)
                print(f"    - ? ë¢°?? {expansion.get('confidence', 0):.2%}")

            # ì²˜ë¦¬ ?¨ê³„ ì¶œë ¥
            steps = result.get('processing_steps', [])
            if steps:
                print("\n  ì²˜ë¦¬ ?¨ê³„:")
                for i, step in enumerate(steps[-5:], 1):  # ë§ˆì?ë§?5ê°œë§Œ
                    print(f"    {i}. {step}")

            return result

        except Exception as e:
            print(f"\n??ì²˜ë¦¬ ?¤íŒ¨: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_all_tests(self):
        """?„ì²´ ?ŒìŠ¤???¤í–‰"""
        self.test_start_time = time.time()

        print("=" * 80)
        print("LangSmith ?µí•© ?ŒìŠ¤???œì‘")
        print("ê°œì„ ??LangGraph ?Œí¬?Œë¡œ???ŒìŠ¤??)
        print("=" * 80)

        # ?¤ì • ê²€ì¦?
        validation_errors = self.config.validate()
        if validation_errors:
            print("\n???¤ì • ê²€ì¦??¤ë¥˜:")
            for error in validation_errors:
                print(f"  - {error}")
            print("?ŒìŠ¤?¸ë? ê³„ì† ì§„í–‰?˜ì?ë§??¼ë? ê¸°ëŠ¥???œí•œ?????ˆìŠµ?ˆë‹¤.")

        print("=" * 80)

        # ?ŒìŠ¤??ì¼€?´ìŠ¤ ?•ì˜ (?•ë???
        test_cases = [
            {
                "query": "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???Œë ¤ì£¼ì„¸??,
                "description": "ê¸°ë³¸ ë²•ë¥  ì¡°ì–¸ ì§ˆë¬¸"
            },
            {
                "query": "ê³„ì•½ ?„ë°˜ ??ë²•ì  ì¡°ì¹˜ ë°©ë²•",
                "description": "ê³„ì•½ ê´€??ì§ˆë¬¸"
            },
            {
                "query": "ë¯¼ì‚¬?Œì†¡?ì„œ ?¹ì†Œ?˜ê¸° ?„í•œ ì¦ê±° ?˜ì§‘ ë°©ë²•",
                "description": "ë¯¼ì‚¬?Œì†¡ ?ˆì°¨ ì§ˆë¬¸"
            },
            {
                "query": "ê³„ì•½?œì— ?°ë¥´ë©?ë°°ì†¡ ì§€?????´ë–»ê²??€?‘í•´???˜ë‚˜??",
                "description": "êµ¬ì²´???¬ì•ˆ ì§ˆë¬¸"
            },
            {
                "query": "?´ì „???Œê°œ?´ì£¼???í•´ë°°ìƒ ì²?µ¬?ì„œ ê³¼ì‹¤ë¹„ìœ¨?€ ?´ë–»ê²?ê²°ì •?˜ë‚˜??",
                "description": "ë©€?°í„´ ì§ˆë¬¸ (?´ì „ ì§ˆë¬¸ ì°¸ì¡°)"
            },
            {
                "query": "ë¯¼ë²• ??50ì¡??í•´ë°°ìƒ??ë²”ìœ„??",
                "description": "?¹ì • ë²•ì¡°ë¬??´ì„ ì§ˆë¬¸"
            },
            {
                "query": "?´í–‰ë¶ˆëŠ¥ê³??´í–‰ë¶ˆê??¥ì˜ ì°¨ì´",
                "description": "ë²•ë¥  ?©ì–´ ë¹„êµ ì§ˆë¬¸"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            # ?€?„ìŠ¤?¬í”„ë¥??¬í•¨???¸ì…˜ IDë¡?ê²©ë¦¬ ê°•í™”
            session_id = f"test_{int(time.time())}_{i:03d}"

            print(f"\n\n{'#'*80}")
            print(f"?ŒìŠ¤??ì¼€?´ìŠ¤ #{i}/{len(test_cases)}")
            print(f"?¤ëª…: {test_case['description']}")
            print(f"?¸ì…˜: {session_id}")
            print(f"{'#'*80}")

            try:
                result = await self.run_single_query(
                    query=test_case['query'],
                    session_id=session_id,
                    query_number=i
                )

                # ?ˆì§ˆ ê²€ì¦?ì¶”ê?
                if result:
                    quality = self.validate_answer_quality(result)
                    results.append({
                        'case': i,
                        'query': test_case['query'],
                        'description': test_case['description'],
                        'session_id': session_id,
                        'result': result,
                        'quality': quality,
                        'success': True
                    })
                else:
                    results.append({
                        'case': i,
                        'query': test_case['query'],
                        'description': test_case['description'],
                        'session_id': session_id,
                        'result': None,
                        'success': False,
                        'error': 'Result is None'
                    })
            except Exception as e:
                print(f"\n???ŒìŠ¤??ì¼€?´ìŠ¤ #{i} ?¤í–‰ ?¤íŒ¨: {e}")
                results.append({
                    'case': i,
                    'query': test_case['query'],
                    'description': test_case['description'],
                    'session_id': session_id,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        # ?µê³„ ì¶œë ¥
        print("\n" + "=" * 80)
        print("?ŒìŠ¤??ê²°ê³¼ ?µê³„")
        print("=" * 80)

        total_queries = len(test_cases)
        successful_results = [r for r in results if r.get('success')]
        failed_results = [r for r in results if not r.get('success')]

        print(f"\nì´?ì§ˆì˜ ?? {total_queries}")
        print(f"?±ê³µ??ì§ˆì˜: {len(successful_results)}")
        print(f"?¤íŒ¨??ì§ˆì˜: {len(failed_results)}")

        if successful_results:
            # ?‰ê·  ê°?ê³„ì‚°
            total_confidence = sum(r['result'].get('confidence', 0) for r in successful_results if r.get('result'))
            total_docs = sum(len(r['result'].get('sources', [])) for r in successful_results if r.get('result'))
            total_legal_refs = sum(len(r['result'].get('legal_references', [])) for r in successful_results if r.get('result'))
            total_steps = sum(len(r['result'].get('processing_steps', [])) for r in successful_results if r.get('result'))
            total_time = sum(r['result'].get('processing_time', 0) for r in successful_results if r.get('result'))
            total_quality = sum(r.get('quality', {}).get('overall_score', 0) for r in successful_results)

            success_count = len(successful_results)

            print(f"\n?‰ê·  ? ë¢°?? {total_confidence/success_count:.2%}")
            print(f"?‰ê·  ?ˆì§ˆ ?ìˆ˜: {total_quality/success_count:.2%}")
            print(f"?‰ê·  ?ŒìŠ¤ ?? {total_docs/success_count:.1f}ê°?)
            print(f"?‰ê·  ë²•ë¥  ì°¸ì¡° ?? {total_legal_refs/success_count:.1f}ê°?)
            print(f"?‰ê·  ì²˜ë¦¬ ?¨ê³„ ?? {total_steps/success_count:.1f}ê°?)
            print(f"?‰ê·  ì²˜ë¦¬ ?œê°„: {total_time/success_count:.2f}ì´?)

            # ?ˆì§ˆ ?µê³„
            valid_qualities = [r.get('quality', {}).get('overall_score', 0) for r in successful_results if r.get('quality')]
            if valid_qualities:
                min_quality = min(valid_qualities)
                max_quality = max(valid_qualities)
                print(f"\n?ˆì§ˆ ?ìˆ˜ ë²”ìœ„: {min_quality:.2%} ~ {max_quality:.2%}")
                high_quality_count = len([q for q in valid_qualities if q >= 0.7])
                print(f"ê³ í’ˆì§??µë? (??0%): {high_quality_count}ê°?)

            # AI ?¤ì›Œ???•ì¥ ?µê³„
            ai_expansions = []
            for r in successful_results:
                if r.get('result') and 'metadata' in r['result']:
                    metadata = r['result']['metadata']
                    if 'ai_keyword_expansion' in metadata:
                        ai_expansions.append(metadata['ai_keyword_expansion'])

            if ai_expansions:
                print(f"\nAI ?¤ì›Œ???•ì¥ ?¤í–‰: {len(ai_expansions)}??)
                gemini_count = len([e for e in ai_expansions if e.get('method') == 'gemini_ai'])
                fallback_count = len([e for e in ai_expansions if e.get('method') == 'fallback'])
                print(f"  - Gemini AI: {gemini_count}??)
                print(f"  - Fallback: {fallback_count}??)

        # ê²°ê³¼ ?€??
        try:
            filename = await self.save_results(results)
            print(f"\n??ê²°ê³¼ ?€?¥ë¨: {filename}")
        except Exception as e:
            print(f"\n??ê²°ê³¼ ?€???¤íŒ¨: {e}")

        print("\n" + "=" * 80)
        print("LangSmith ëª¨ë‹ˆ?°ë§ ?•ì¸")
        print("=" * 80)
        print("\nLangSmith ?€?œë³´?œì—???¤ìŒ ?•ë³´ë¥??•ì¸?????ˆìŠµ?ˆë‹¤:")
        print("  - ê°??¸ë“œ???¤í–‰ ?œê°„")
        print("  - ?¸ë“œ ê°??°ì´???ë¦„")
        print("  - AI ?¤ì›Œ???•ì¥ ê³¼ì •")
        print("  - ?ëŸ¬ ë°?ê²½ê³  ë©”ì‹œì§€")
        print("  - ? í° ?¬ìš©??)
        print("  - ë¹„ìš© ì¶”ì ")
        print("\nLangSmith URL: https://smith.langchain.com (?´ë¼?°ë“œ ?¤ì •??ê²½ìš°)")

        return results


async def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    test_runner = LangSmithIntegrationTest()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 80)
    print("?ŒìŠ¤???„ë£Œ!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # LangSmith ?¤ì • ?•ì¸ ë°?ì¶œë ¥
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "LawFirmAI-Test")

    print("\n" + "=" * 80)
    print("LangSmith ?¤ì • ?•ì¸")
    print("=" * 80)

    if langchain_api_key and langchain_tracing.lower() == "true":
        print("??LangSmith ?œì„±?”ë¨")
        print(f"  API Key: {langchain_api_key[:20]}...{langchain_api_key[-10:]} (ë¶€ë¶??œì‹œ)")
        print(f"  Project: {langchain_project}")
        print(f"  Tracing: {langchain_tracing}")
    else:
        print("??LangSmith ?¤ì • ê²½ê³ :")
        print(f"  API Key: {'?¤ì •?? if langchain_api_key else '???¤ì •?˜ì? ?ŠìŒ'}")
        print(f"  Tracing: {langchain_tracing}")
        print(f"  Project: {langchain_project}")
        print("\n?¤ì • ë°©ë²•:")
        print("  .env ?Œì¼??ì¶”ê?:")
        print("    LANGSMITH_API_KEY=your-api-key")
        print("    LANGSMITH_TRACING=true")
        print("    LANGSMITH_PROJECT=your-project-name")
        print("\nLangSmith ?†ì´???ŒìŠ¤?¸ëŠ” ì§„í–‰?©ë‹ˆ??")

    print("=" * 80 + "\n")

    # ?ŒìŠ¤???¤í–‰
    asyncio.run(main())
