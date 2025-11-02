# -*- coding: utf-8 -*-
"""
LawFirmAI - Streamlit 애플리케이션
LangGraph 기반 간소화된 버전
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangGraph 활성화 설정
os.environ["USE_LANGGRAPH"] = "true"

import streamlit as st

# LangGraph 워크플로우 서비스만 사용
from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig

# 로깅 설정
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
    """Streamlit 전용 LawFirmAI 애플리케이션 - LangGraph 기반"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.workflow = None
        self.is_initialized = False
        self.initialization_error = None
        self.current_session_id = f"streamlit_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_user_id = "streamlit_user"

    def initialize_components(self) -> bool:
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing LawFirmAI with LangGraph...")
            start_time = time.time()

            # LangGraph 설정 로드
            self.config = LangGraphConfig.from_env()
            self.logger.info("LangGraph config loaded")

            # LangGraph 워크플로우 서비스 초기화
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
        """질의 처리 - LangGraph 사용"""
        if not self.is_initialized:
            return {
                "answer": "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "error": "System not initialized",
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"}
            }

        # 세션 ID 설정
        if not session_id:
            session_id = self.current_session_id
        if not user_id:
            user_id = self.current_user_id

        start_time = time.time()

        try:
            # LangGraph를 통해 질문 처리
            if self.workflow:
                result = asyncio.run(self.workflow.process_query(query, session_id))
            else:
                raise RuntimeError("Workflow not initialized")

            response_time = time.time() - start_time

            return {
                "answer": result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다."),
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
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "error": str(e),
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                "processing_time": response_time
            }


@st.cache_resource
def initialize_app():
    """Streamlit 캐시를 사용한 앱 초기화"""
    app = StreamlitApp()
    if not app.initialize_components():
        logger.error("Failed to initialize components")
    return app


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="LawFirmAI - 법률 AI 어시스턴트",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚖️ LawFirmAI - 법률 AI 어시스턴트")
    st.markdown("### LangGraph 기반 법률 질의응답 시스템")

    # 시스템 초기화
    if 'app' not in st.session_state:
        with st.spinner('시스템을 초기화하는 중...'):
            st.session_state.app = initialize_app()

    app = st.session_state.app

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        st.subheader("시스템 정보")
        if app.is_initialized:
            st.success("✅ 시스템이 정상적으로 초기화되었습니다.")
            st.info(f"세션 ID: {app.current_session_id[:20]}...")
            st.info(f"사용자 ID: {app.current_user_id}")
        else:
            st.error("❌ 시스템 초기화 실패")
            if app.initialization_error:
                st.error(f"오류: {app.initialization_error}")

    # 채팅 인터페이스
    st.markdown("### 법률 관련 질문에 정확하고 신뢰할 수 있는 답변을 제공합니다.")

    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 채팅 표시
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

                # 신뢰도 정보 표시
                if "confidence" in msg:
                    conf_info = msg["confidence"]
                    st.caption(
                        f"신뢰도: {conf_info.get('confidence', 0):.1%} | "
                        f"수준: {conf_info.get('reliability_level', 'Unknown')} | "
                        f"처리 시간: {msg.get('processing_time', 0):.2f}초"
                    )

    # 질문 입력
    user_input = st.chat_input("법률 관련 질문을 입력하세요...")

    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    # 답변 생성
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_msg = st.session_state.chat_history[-1]["content"]

        with st.spinner('답변을 생성하는 중...'):
            result = app.process_query(last_msg)

            answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")

            # 답변 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "confidence": result.get("confidence", {}),
                "processing_time": result.get("processing_time", 0)
            })

            st.rerun()


if __name__ == "__main__":
    main()
