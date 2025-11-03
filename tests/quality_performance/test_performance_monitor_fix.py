#!/usr/bin/env python3
"""
PerformanceMonitor log_request ?¤ë¥˜ ?ŒìŠ¤??
"""

import sys
import os
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?sys.path??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_performance_monitor():
    """PerformanceMonitor ?ŒìŠ¤??""
    print("=== PerformanceMonitor log_request ?ŒìŠ¤??===")
    
    try:
        # gradio/app.py??PerformanceMonitor ?´ë˜???ŒìŠ¤??
        from gradio.app import PerformanceMonitor
        
        print("PerformanceMonitor ?´ë˜???„í¬???±ê³µ")
        
        # ?¸ìŠ¤?´ìŠ¤ ?ì„±
        monitor = PerformanceMonitor()
        print("PerformanceMonitor ?¸ìŠ¤?´ìŠ¤ ?ì„± ?±ê³µ")
        
        # log_request ë©”ì„œ???ŒìŠ¤??(ê¸°ë³¸ ë§¤ê°œë³€??
        monitor.log_request(response_time=1.5, success=True)
        print("log_request(response_time, success) ?¸ì¶œ ?±ê³µ")
        
        # log_request ë©”ì„œ???ŒìŠ¤??(operation ë§¤ê°œë³€???¬í•¨)
        monitor.log_request(response_time=2.0, success=True, operation="test_operation")
        print("log_request(response_time, success, operation) ?¸ì¶œ ?±ê³µ")
        
        # ?µê³„ ì¡°íšŒ
        stats = monitor.get_stats()
        print(f"?µê³„ ì¡°íšŒ ?±ê³µ: {stats}")
        
        print("ëª¨ë“  PerformanceMonitor ?ŒìŠ¤???µê³¼!")
        return True
        
    except Exception as e:
        print(f"PerformanceMonitor ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_service_integration():
    """ChatService?€ PerformanceMonitor ?µí•© ?ŒìŠ¤??""
    print("\n=== ChatService ?µí•© ?ŒìŠ¤??===")
    
    try:
        from source.utils.config import Config
        from source.services.chat_service import ChatService
        import asyncio
        
        print("ChatService ì´ˆê¸°??ì¤?..")
        config = Config()
        chat_service = ChatService(config)
        
        # ê°„ë‹¨??ë©”ì‹œì§€ ì²˜ë¦¬ ?ŒìŠ¤??
        test_message = "?ˆë…•?˜ì„¸??
        session_id = "test_session_performance"
        user_id = "test_user_performance"
        
        print(f"?ŒìŠ¤??ë©”ì‹œì§€: {test_message}")
        
        # ë¹„ë™ê¸??¨ìˆ˜ ?¤í–‰
        async def run_test():
            result = await chat_service.process_message(test_message, None, session_id, user_id)
            return result
        
        result = asyncio.run(run_test())
        
        if result and isinstance(result, dict):
            print("ChatService ë©”ì‹œì§€ ì²˜ë¦¬ ?±ê³µ")
            print(f"?‘ë‹µ ?? {list(result.keys())}")
            return True
        else:
            print("ChatService ë©”ì‹œì§€ ì²˜ë¦¬ ?¤íŒ¨")
            return False
            
    except Exception as e:
        print(f"ChatService ?µí•© ?ŒìŠ¤???¤íŒ¨: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PerformanceMonitor ?¤ë¥˜ ?˜ì • ?ŒìŠ¤???œì‘...")
    
    # PerformanceMonitor ?ŒìŠ¤??
    monitor_success = test_performance_monitor()
    
    # ChatService ?µí•© ?ŒìŠ¤??
    integration_success = test_chat_service_integration()
    
    # ìµœì¢… ê²°ê³¼
    print(f"\n=== ìµœì¢… ê²°ê³¼ ===")
    print(f"PerformanceMonitor ?ŒìŠ¤?? {'?µê³¼' if monitor_success else '?¤íŒ¨'}")
    print(f"ChatService ?µí•© ?ŒìŠ¤?? {'?µê³¼' if integration_success else '?¤íŒ¨'}")
    
    if monitor_success and integration_success:
        print("ëª¨ë“  ?ŒìŠ¤???µê³¼! log_request ?¤ë¥˜ê°€ ?´ê²°?˜ì—ˆ?µë‹ˆ??")
    else:
        print("?¼ë? ?ŒìŠ¤???¤íŒ¨. ì¶”ê? ?˜ì •???„ìš”?©ë‹ˆ??")
