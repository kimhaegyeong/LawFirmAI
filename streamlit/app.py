# -*- coding: utf-8 -*-
"""
LawFirmAI - Streamlit ? í”Œë¦¬ì??´ì…˜
LangGraph ê¸°ë°˜ ê°„ì†Œ?”ëœ ë²„ì „
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangGraph ?œì„±???¤ì •
os.environ["USE_LANGGRAPH"] = "true"

import streamlit as st

# LangGraph ?Œí¬?Œë¡œ???œë¹„?¤ë§Œ ?¬ìš©
from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/streamlit_app.log')
    ]
)
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit ?„ìš© LawFirmAI ? í”Œë¦¬ì??´ì…˜ - LangGraph ê¸°ë°˜"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.workflow = None
        self.is_initialized = False
        self.initialization_error = None
        self.current_session_id = f"streamlit_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_user_id = "streamlit_user"

    def initialize_components(self) -> bool:
        """ì»´í¬?ŒíŠ¸ ì´ˆê¸°??""
        try:
            self.logger.info("Initializing LawFirmAI with LangGraph...")
            start_time = time.time()

            # LangGraph ?¤ì • ë¡œë“œ
            self.config = LangGraphConfig.from_env()
            self.logger.info("LangGraph config loaded")

            # LangGraph ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
            self.workflow = LangGraphWorkflowService(self.config)
            self.logger.info("LangGraph workflow service initialized")

            initialization_time = time.time() - start_time
            self.logger.info(f"Components initialized successfully in {initialization_time:.2f} seconds")

            self.is_initialized = True
            return True

        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    def process_query(self, query: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """ì§ˆì˜ ì²˜ë¦¬ - LangGraph ?¬ìš©"""
        if not self.is_initialized:
            return {
                "answer": "?œìŠ¤?œì´ ?„ì§ ì´ˆê¸°?”ë˜ì§€ ?Šì•˜?µë‹ˆ?? ? ì‹œ ???¤ì‹œ ?œë„?´ì£¼?¸ìš”.",
                "error": "System not initialized",
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"}
            }

        # ?¸ì…˜ ID ?¤ì •
        if not session_id:
            session_id = self.current_session_id
        if not user_id:
            user_id = self.current_user_id

        start_time = time.time()

        try:
            # LangGraphë¥??µí•´ ì§ˆë¬¸ ì²˜ë¦¬
            if self.workflow:
                result = asyncio.run(self.workflow.process_query(query, session_id))
            else:
                raise RuntimeError("Workflow not initialized")

            response_time = time.time() - start_time

            return {
                "answer": result.get("answer", "ì£„ì†¡?©ë‹ˆ?? ?µë????ì„±?????†ìŠµ?ˆë‹¤."),
                "confidence": {
                    "confidence": result.get("confidence", 0.0),
                    "reliability_level": "HIGH" if result.get("confidence", 0) > 0.7 else "MEDIUM" if result.get("confidence", 0) > 0.4 else "LOW"
                },
                "processing_time": response_time,
                "question_type": result.get("query_type", "general_question"),
                "session_id": result.get("session_id", session_id),
                "user_id": user_id,
                "legal_references": result.get("legal_references", []),
                "processing_steps": result.get("processing_steps", []),
                "metadata": result.get("metadata", {}),
                "errors": result.get("errors", [])
            }

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Error processing query: {e}")

            return {
                "answer": "ì£„ì†¡?©ë‹ˆ?? ì²˜ë¦¬ ì¤??¤ë¥˜ê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤. ?¤ì‹œ ?œë„?´ì£¼?¸ìš”.",
                "error": str(e),
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                "processing_time": response_time
            }


@st.cache_resource
def initialize_app():
    """Streamlit ìºì‹œë¥??¬ìš©????ì´ˆê¸°??""
    app = StreamlitApp()
    if not app.initialize_components():
        logger.error("Failed to initialize components")
    return app


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    st.set_page_config(
        page_title="LawFirmAI - ë²•ë¥  AI ?´ì‹œ?¤í„´??,
        page_icon="?–ï¸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("?–ï¸ LawFirmAI - ë²•ë¥  AI ?´ì‹œ?¤í„´??)
    st.markdown("### LangGraph ê¸°ë°˜ ë²•ë¥  ì§ˆì˜?‘ë‹µ ?œìŠ¤??)

    # ?œìŠ¤??ì´ˆê¸°??
    if 'app' not in st.session_state:
        with st.spinner('?œìŠ¤?œì„ ì´ˆê¸°?”í•˜??ì¤?..'):
            st.session_state.app = initialize_app()

    app = st.session_state.app

    # ?¬ì´?œë°”
    with st.sidebar:
        st.header("?™ï¸ ?¤ì •")
        st.subheader("?œìŠ¤???•ë³´")
        if app.is_initialized:
            st.success("???œìŠ¤?œì´ ?•ìƒ?ìœ¼ë¡?ì´ˆê¸°?”ë˜?ˆìŠµ?ˆë‹¤.")
            st.info(f"?¸ì…˜ ID: {app.current_session_id[:20]}...")
            st.info(f"?¬ìš©??ID: {app.current_user_id}")
        else:
            st.error("???œìŠ¤??ì´ˆê¸°???¤íŒ¨")
            if app.initialization_error:
                st.error(f"?¤ë¥˜: {app.initialization_error}")

    # ì±„íŒ… ?¸í„°?˜ì´??
    st.markdown("### ë²•ë¥  ê´€??ì§ˆë¬¸???•í™•?˜ê³  ? ë¢°?????ˆëŠ” ?µë????œê³µ?©ë‹ˆ??")

    # ì±„íŒ… ?ˆìŠ¤? ë¦¬ ì´ˆê¸°??
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # ì±„íŒ… ?œì‹œ
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

                # ? ë¢°???•ë³´ ?œì‹œ
                if "confidence" in msg:
                    conf_info = msg["confidence"]
                    st.caption(
                        f"? ë¢°?? {conf_info.get('confidence', 0):.1%} | "
                        f"?˜ì?: {conf_info.get('reliability_level', 'Unknown')} | "
                        f"ì²˜ë¦¬ ?œê°„: {msg.get('processing_time', 0):.2f}ì´?
                    )

    # ì§ˆë¬¸ ?…ë ¥
    user_input = st.chat_input("ë²•ë¥  ê´€??ì§ˆë¬¸???…ë ¥?˜ì„¸??..")

    if user_input:
        # ?¬ìš©??ë©”ì‹œì§€ ì¶”ê?
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    # ?µë? ?ì„±
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_msg = st.session_state.chat_history[-1]["content"]

        with st.spinner('?µë????ì„±?˜ëŠ” ì¤?..'):
            result = app.process_query(last_msg)

            answer = result.get("answer", "ì£„ì†¡?©ë‹ˆ?? ?µë????ì„±?????†ìŠµ?ˆë‹¤.")

            # ?µë? ì¶”ê?
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "confidence": result.get("confidence", {}),
                "processing_time": result.get("processing_time", 0)
            })

            st.rerun()


if __name__ == "__main__":
    main()
