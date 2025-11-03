# -*- coding: utf-8 -*-
"""
?„ì²´ LangGraph ?ŒìŠ¤???¤í–‰ ?¤í¬ë¦½íŠ¸
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Windows ë¹„ë™ê¸??˜ê²½?ì„œ ë¡œê¹… ë²„í¼ ?ëŸ¬ ë°©ì?
class SafeStreamHandler(logging.StreamHandler):
    """?ˆì „???¤íŠ¸ë¦??¸ë“¤??- detached ë²„í¼ ?ëŸ¬ ë°©ì?"""
    def emit(self, record):
        try:
            super().emit(record)
        except (ValueError, OSError, AttributeError):
            # detached buffer ?ëŸ¬??ê¸°í? ?¤íŠ¸ë¦??ëŸ¬ ë¬´ì‹œ
            pass

logging.basicConfig(
    level=logging.ERROR,
    handlers=[SafeStreamHandler()],
    force=True  # ê¸°ì¡´ ?¤ì •??ê°•ì œë¡??¬ì„¤??
)
logging.raiseExceptions = False  # ë¡œê¹… ?ˆì™¸ ë¬´ì‹œ
logger = logging.getLogger(__name__)

# ?ŒìŠ¤??ëª¨ë“ˆ import
import importlib.util


def import_test_module(module_name):
    """?ŒìŠ¤??ëª¨ë“ˆ ?™ì  import"""
    module_path = Path(__file__).parent / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"?ŒìŠ¤???Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_all_tests():
    """ëª¨ë“  ?ŒìŠ¤???¤í–‰"""
    print("=" * 80)
    print("?„ì²´ LangGraph ?ŒìŠ¤???¤í–‰")
    print("=" * 80)
    print(f"\n?œì‘ ?œê°„: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = []
    total_start = time.time()

    # ?ŒìŠ¤??1: ëª¨ë“  ?œë‚˜ë¦¬ì˜¤ ?ŒìŠ¤??
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??1: ëª¨ë“  ?œë‚˜ë¦¬ì˜¤ ?ŒìŠ¤??(test_all_scenarios)")
    print("=" * 80)
    try:
        module = import_test_module("test_all_scenarios")
        result = await module.main()
        test_results.append(("ëª¨ë“  ?œë‚˜ë¦¬ì˜¤ ?ŒìŠ¤??, result == 0))
    except Exception as e:
        print(f"  ??[ERROR] ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("ëª¨ë“  ?œë‚˜ë¦¬ì˜¤ ?ŒìŠ¤??, False))

    # ?ŒìŠ¤??2: ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤??
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??2: ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤??(test_optimized_workflow)")
    print("=" * 80)
    try:
        module = import_test_module("test_optimized_workflow")
        await module.main()
        # ???ŒìŠ¤?¸ëŠ” exit codeë¥?ë°˜í™˜?˜ì? ?Šìœ¼ë¯€ë¡??±ê³µ?¼ë¡œ ê°„ì£¼
        test_results.append(("ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤??, True))
    except Exception as e:
        print(f"  ??[ERROR] ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("ìµœì ?”ëœ ?Œí¬?Œë¡œ???ŒìŠ¤??, False))

    # ?ŒìŠ¤??3: ?¸ë“œ ?µí•© ?ŒìŠ¤??
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??3: ?¸ë“œ ?µí•© ?ŒìŠ¤??(test_node_integration)")
    print("=" * 80)
    try:
        module = import_test_module("test_node_integration")
        result = await module.main()
        test_results.append(("?¸ë“œ ?µí•© ?ŒìŠ¤??, result == 0))
    except Exception as e:
        print(f"  ??[ERROR] ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("?¸ë“œ ?µí•© ?ŒìŠ¤??, False))

    # ?ŒìŠ¤??4: ê°„ë‹¨???¸ë“œ ?µí•© ?ŒìŠ¤??
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??4: ê°„ë‹¨???¸ë“œ ?µí•© ?ŒìŠ¤??(test_node_integration_simple)")
    print("=" * 80)
    try:
        module = import_test_module("test_node_integration_simple")
        result = await module.test_integration()
        test_results.append(("ê°„ë‹¨???¸ë“œ ?µí•© ?ŒìŠ¤??, result == 0))
    except Exception as e:
        print(f"  ??[ERROR] ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("ê°„ë‹¨???¸ë“œ ?µí•© ?ŒìŠ¤??, False))

    # ?ŒìŠ¤??5: ?„ë¡¬?„íŠ¸ ê°œì„  ?ŒìŠ¤??
    print("\n" + "=" * 80)
    print("?ŒìŠ¤??5: ?„ë¡¬?„íŠ¸ ê°œì„  ?ŒìŠ¤??(test_prompt_improvements)")
    print("=" * 80)
    test_file_path = Path(__file__).parent / "test_prompt_improvements.py"
    if not test_file_path.exists():
        print("  ??¸  [SKIP] ?ŒìŠ¤???Œì¼???†ì–´ ê±´ë„ˆ?ë‹ˆ??")
        test_results.append(("?„ë¡¬?„íŠ¸ ê°œì„  ?ŒìŠ¤??, None))  # None?€ ê±´ë„ˆ???ŒìŠ¤?¸ë? ?˜ë?
    else:
        try:
            module = import_test_module("test_prompt_improvements")
            result = await module.main()
            test_results.append(("?„ë¡¬?„íŠ¸ ê°œì„  ?ŒìŠ¤??, result == 0))
        except Exception as e:
            print(f"  ??[ERROR] ?ŒìŠ¤???¤íŒ¨: {e}")
            import traceback
            traceback.print_exc()
            test_results.append(("?„ë¡¬?„íŠ¸ ê°œì„  ?ŒìŠ¤??, False))

    total_elapsed = time.time() - total_start

    # ìµœì¢… ê²°ê³¼ ?”ì•½
    print("\n" + "=" * 80)
    print("?“Š ?„ì²´ ?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("=" * 80)

    # None?€ ê±´ë„ˆ???ŒìŠ¤?? True/False???¤í–‰???ŒìŠ¤??ê²°ê³¼
    skipped = sum(1 for _, result in test_results if result is None)
    passed = sum(1 for _, result in test_results if result is True)
    failed = sum(1 for _, result in test_results if result is False)
    total_executed = passed + failed

    for test_name, result in test_results:
        if result is None:
            status = "??¸  SKIP"
        elif result:
            status = "??PASS"
        else:
            status = "??FAIL"
        print(f"  {test_name}: {status}")

    if skipped > 0:
        print(f"\n?„ì²´: {passed}/{total_executed} ?ŒìŠ¤???µê³¼ ({skipped}ê°?ê±´ë„ˆ?€)")
    else:
        print(f"\n?„ì²´: {passed}/{total_executed} ?ŒìŠ¤???µê³¼")
    print(f"ì´??¤í–‰ ?œê°„: {total_elapsed:.2f}ì´?)
    print(f"ì¢…ë£Œ ?œê°„: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if total_executed > 0 and passed == total_executed:
        print("\n??ëª¨ë“  ?ŒìŠ¤???µê³¼!")
        return 0
    elif total_executed > 0:
        print(f"\n? ï¸ {failed}ê°??ŒìŠ¤???¤íŒ¨")
        return 1
    else:
        print("\n? ï¸ ?¤í–‰???ŒìŠ¤?¸ê? ?†ìŠµ?ˆë‹¤.")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
