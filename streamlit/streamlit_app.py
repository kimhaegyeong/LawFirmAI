#!/usr/bin/env python3
"""
Streamlit 법률 챗봇 애플리케이션
"""

import streamlit as st
import json
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import threading
import psutil
import gc

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'source'))

from source.services.enhanced_chat_service import EnhancedChatService
from source.utils.config import Config

class StreamlitLegalChatbot:
    """Streamlit 법률 챗봇"""
    
    def __init__(self):
        """초기화 - 지연 로딩 방식"""
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.chat_service = None  # 초기에는 None으로 설정
        self.history_file = "data/conversation_history.json"
        self._initialization_started = False  # 초기화 시작 플래그
        
        # 빠른 질문 템플릿
        self.quick_questions = [
            "계약서 작성 시 주의사항",
            "임대차 계약 분쟁 해결",
            "교통사고 처리 절차",
            "퇴직금 계산 방법",
            "명예훼손 기준과 대응방법",
            "부동산 매매 계약서 검토",
            "근로계약서 필수 조항",
            "소비자 분쟁 해결 절차"
        ]
        
        self._ensure_history_file()
        # _initialize_chat_service() 제거 - 지연 로딩으로 변경
        
        # 메모리 모니터링 초기화
        self._memory_stats = {
            'initial_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'peak_memory': 0,
            'current_memory': 0
        }
    
    def _ensure_history_file(self):
        """히스토리 파일 존재 확인"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def get_chat_service(self):
        """필요할 때만 채팅 서비스 초기화 - 지연 로딩"""
        if self.chat_service is None and not self._initialization_started:
            self._initialization_started = True
            
            # 초기화 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 데이터베이스 연결 중...")
                progress_bar.progress(20)
                
                status_text.text("🔄 AI 모델 로딩 중...")
                progress_bar.progress(50)
                
                status_text.text("🔄 벡터 인덱스 로딩 중...")
                progress_bar.progress(80)
                
                self.chat_service = EnhancedChatService(self.config)
                
                status_text.text("✅ AI 서비스 초기화 완료!")
                progress_bar.progress(100)
                
                # 완료 메시지 잠시 표시
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                self.logger.info("EnhancedChatService 초기화 완료")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                self.logger.error(f"채팅 서비스 초기화 실패: {e}")
                st.error(f"AI 서비스 초기화에 실패했습니다: {str(e)}")
                st.info("페이지를 새로고침하거나 잠시 후 다시 시도해주세요.")
                self._initialization_started = False
                return None
        
        return self.chat_service
    
    def _update_memory_stats(self):
        """메모리 사용량 업데이트"""
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self._memory_stats['current_memory'] = current_memory
            self._memory_stats['peak_memory'] = max(
                self._memory_stats['peak_memory'], 
                current_memory
            )
        except Exception as e:
            self.logger.warning(f"메모리 통계 업데이트 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """메모리 사용량 정보 반환"""
        self._update_memory_stats()
        return {
            'initial_memory_mb': self._memory_stats['initial_memory'],
            'current_memory_mb': self._memory_stats['current_memory'],
            'peak_memory_mb': self._memory_stats['peak_memory'],
            'memory_increase_mb': self._memory_stats['current_memory'] - self._memory_stats['initial_memory']
        }
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            # 가비지 컬렉션 실행
            collected = gc.collect()
            
            # 메모리 통계 업데이트
            self._update_memory_stats()
            
            self.logger.info(f"메모리 정리 완료: {collected}개 객체 수집")
            return collected
        except Exception as e:
            self.logger.error(f"메모리 정리 실패: {e}")
            return 0
    
    def _generate_session_id(self):
        """새 세션 ID 생성"""
        return f"session_{int(datetime.now().timestamp())}"
    
    def _generate_conversation_title(self, query: str) -> str:
        """대화 제목 자동 생성"""
        try:
            # 간단한 키워드 기반 제목 생성
            keywords = {
                "계약": "계약 관련 상담",
                "임대차": "임대차 계약 상담", 
                "교통사고": "교통사고 처리 상담",
                "퇴직금": "퇴직금 계산 상담",
                "명예훼손": "명예훼손 관련 상담",
                "부동산": "부동산 관련 상담",
                "근로": "근로계약 상담",
                "소비자": "소비자 분쟁 상담"
            }
            
            for keyword, title in keywords.items():
                if keyword in query:
                    return title
            
            return "법률 상담"
        except Exception as e:
            self.logger.error(f"제목 생성 오류: {e}")
            return "법률 상담"
    
    def _process_message_sync(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """동기적으로 메시지 처리"""
        try:
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.chat_service.process_message(message, user_id)
                    )
                finally:
                    loop.close()
            
            # 별도 스레드에서 비동기 함수 실행
            result = None
            exception = None
            
            def target():
                nonlocal result, exception
                try:
                    result = run_async()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=30)  # 30초 타임아웃
            
            if thread.is_alive():
                raise TimeoutError("메시지 처리 시간 초과")
            
            if exception:
                raise exception
            
            return result
            
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")
            return {
                "response": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "question_type": "unknown",
                "generation_method": "error",
                "is_restricted": False
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """쿼리 처리 및 응답 생성 - 지연 로딩 적용"""
        if not query.strip():
            return {"response": "질문을 입력해주세요.", "confidence": 0.0}
        
        # 채팅 서비스가 없으면 초기화
        chat_service = self.get_chat_service()
        if chat_service is None:
            return {
                "response": "AI 서비스를 초기화할 수 없습니다. 잠시 후 다시 시도해주세요.",
                "confidence": 0.0,
                "sources": [],
                "question_type": "unknown",
                "generation_method": "error",
                "is_restricted": False
            }
        
        try:
            # 채팅 서비스로 메시지 처리
            result = self._process_message_sync(query, "default")
            return result
            
        except Exception as e:
            self.logger.error(f"쿼리 처리 오류: {e}")
            return {
                "response": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "question_type": "unknown",
                "generation_method": "error",
                "is_restricted": False
            }
    
    def save_conversation(self, messages: List[Dict[str, str]], title: str) -> str:
        """대화 내용 저장"""
        try:
            if not messages:
                return "저장할 대화 내용이 없습니다."
            
            # 기존 히스토리 로드
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    all_history = json.load(f)
            except FileNotFoundError:
                all_history = []
            
            # 현재 대화 데이터 생성
            conversation_data = {
                "id": self._generate_session_id(),
                "title": title,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "messages": messages
            }
            
            all_history.append(conversation_data)
            
            # 파일에 저장
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(all_history, f, ensure_ascii=False, indent=2)
            
            return f"대화가 저장되었습니다: {title}"
            
        except Exception as e:
            self.logger.error(f"대화 저장 오류: {e}")
            return f"저장 중 오류가 발생했습니다: {str(e)}"
    
    def load_conversation_history(self) -> List[Dict[str, Any]]:
        """대화 히스토리 로드"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
            
            # 최근 10개 대화만 반환
            return all_history[-10:] if len(all_history) > 10 else all_history
            
        except Exception as e:
            self.logger.error(f"히스토리 로드 오류: {e}")
            return []

def main():
    """메인 Streamlit 애플리케이션"""
    
    # 페이지 설정
    st.set_page_config(
        page_title="AI 법률 상담 챗봇",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 스타일 추가
    st.markdown("""
    <style>
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .sidebar .sidebar-content {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* 채팅 메시지 스타일링 */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* 헤더 스타일링 */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* 버튼 스타일링 */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 입력창 스타일링 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화 - 빠른 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = StreamlitLegalChatbot()  # 이제 빠르게 초기화됨
    if "conversation_title" not in st.session_state:
        st.session_state.conversation_title = "새로운 상담"
    if "ai_initialized" not in st.session_state:
        st.session_state.ai_initialized = False
    if "stream_speed" not in st.session_state:
        st.session_state.stream_speed = 0.02
    if "new_message_added" not in st.session_state:
        st.session_state.new_message_added = False
    if "streaming_interrupted" not in st.session_state:
        st.session_state.streaming_interrupted = False
    
    chatbot = st.session_state.chatbot
    
    # AI 초기화 상태 표시
    if not st.session_state.ai_initialized:
        st.info("💡 **AI 서비스 준비 완료!** 질문을 입력하거나 빠른 질문을 클릭하면 AI가 초기화됩니다.")
    
    # 헤더 영역
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #1f77b4; margin: 0;">⚖️</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #1f77b4; margin: 0; font-size: 2.5rem;">AI 법률 상담 챗봇</h1>
            <p style="color: #666; margin: 5px 0; font-size: 1.1rem;">전문적인 법률 정보를 제공합니다</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("❓ 도움말", help="사용법과 주의사항을 확인하세요"):
            st.info("""
            **사용법:**
            1. 법률 관련 질문을 입력하세요
            2. 빠른 질문 버튼을 활용하세요
            3. 대화 내용을 저장할 수 있습니다
            
            **주의사항:**
            - 일반적인 법률 정보만 제공됩니다
            - 전문적인 법률 자문은 변호사와 상담하세요
            """)
    
    # 구분선
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 메뉴")
        
        # 새 대화 버튼
        if st.button("➕ 새 대화", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.conversation_title = "새로운 상담"
            st.session_state.streaming_interrupted = False  # 중단 상태 리셋
            st.rerun()
        
        # 빠른 질문 섹션
        with st.expander("⚡ 빠른 질문", expanded=True):
            for question in chatbot.quick_questions[:4]:
                if st.button(question, use_container_width=True, key=f"quick_{question}"):
                    # AI 서비스 초기화 확인
                    chat_service = chatbot.get_chat_service()
                    if chat_service is None:
                        st.error("AI 서비스를 초기화할 수 없습니다.")
                        return
                    
                    # 첫 번째 질문인 경우 제목 생성
                    if not st.session_state.messages:
                        st.session_state.conversation_title = chatbot._generate_conversation_title(question)
                    
                    # 메시지 추가
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # AI 응답 생성
                    with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
                        result = chatbot.process_query(question)
                        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                        st.session_state.ai_initialized = True
                        st.session_state.new_message_added = True
                    
                    st.rerun()
        
        # 대화 히스토리 섹션
        with st.expander("💬 최근 대화", expanded=False):
            history = chatbot.load_conversation_history()
            
            if history:
                # 최근 5개만 표시
                for conv in reversed(history[-5:]):
                    title = conv.get("title", "제목 없음")
                    start_time = conv.get("start_time", "")
                    if start_time:
                        try:
                            dt = datetime.fromisoformat(start_time)
                            time_str = dt.strftime("%m/%d %H:%M")
                        except:
                            time_str = start_time[:10]
                    else:
                        time_str = "시간 없음"
                    
                    if st.button(f"{title} ({time_str})", use_container_width=True, key=f"history_{conv.get('id', '')}"):
                        # 히스토리 불러오기
                        messages = conv.get("messages", [])
                        st.session_state.messages = messages
                        st.session_state.conversation_title = title
                        st.rerun()
            else:
                st.caption("저장된 대화가 없습니다.")
        
        # 참고 정보 섹션
        with st.expander("📖 참고 정보", expanded=False):
            st.markdown("""
            **유용한 링크:**
            - 📖 법률 용어 사전
            - 🏛️ 법률 상담 기관
            - 📞 긴급 상담 전화: 132
            
            **법률 상담 기관:**
            - 법원도서관 법률상담
            - 대한변호사협회 상담
            - 한국법무법인 상담센터
            """)
        
        # 메모리 정보 섹션
        with st.expander("💾 메모리 정보", expanded=False):
            memory_info = chatbot.get_memory_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("현재 메모리", f"{memory_info['current_memory_mb']:.1f} MB")
                st.metric("증가량", f"+{memory_info['memory_increase_mb']:.1f} MB")
            
            with col2:
                st.metric("최대 메모리", f"{memory_info['peak_memory_mb']:.1f} MB")
                
                if st.button("🧹 메모리 정리", use_container_width=True):
                    collected = chatbot.cleanup_memory()
                    st.success(f"{collected}개 객체 정리 완료")
                    st.rerun()
        
        # 스트리밍 설정 섹션
        with st.expander("⚙️ 표시 설정", expanded=False):
            st.markdown("**📝 응답 표시 속도**")
            stream_speed = st.slider(
                "스트리밍 속도 (초/글자)", 
                min_value=0.01, 
                max_value=0.1, 
                value=0.02, 
                step=0.01,
                help="낮을수록 빠르게 표시됩니다"
            )
            
            # 세션 상태에 저장
            st.session_state.stream_speed = stream_speed
        
        # 면책 조항
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;">
            <strong>⚠️ 면책 조항</strong><br>
            이 AI는 일반적인 법률 정보만 제공하며,<br>
            전문적인 법률 자문을 대신할 수 없습니다.
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 채팅 영역
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader(f"💬 {st.session_state.conversation_title}")
    
    with col2:
        if st.button("💾 저장", use_container_width=True):
            if st.session_state.messages:
                result = chatbot.save_conversation(st.session_state.messages, st.session_state.conversation_title)
                st.success(result)
            else:
                st.warning("저장할 대화 내용이 없습니다.")
    
    # 채팅 메시지 표시
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # 메시지 내용
            if (message["role"] == "assistant" and 
                i == len(st.session_state.messages) - 1 and 
                st.session_state.new_message_added):
                # 마지막 AI 응답에만 스트리밍 효과 적용 (새로 추가된 경우만)
                response_placeholder = st.empty()
                control_placeholder = st.empty()
                full_response = ""
                
                # 중단 버튼 표시
                with control_placeholder.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("⏹️ 응답 중단", key=f"interrupt_{i}"):
                            st.session_state.streaming_interrupted = True
                
                # 사용자 설정 속도 사용
                stream_speed = st.session_state.get('stream_speed', 0.02)
                
                for char_idx, char in enumerate(message["content"]):
                    # 중단 플래그 확인
                    if st.session_state.get('streaming_interrupted', False):
                        full_response += "... (사용자에 의해 중단됨)"
                        break
                        
                    full_response += char
                    response_placeholder.markdown(full_response + "▌")
                    
                    # 중단 버튼 업데이트 (5글자마다)
                    if char_idx % 5 == 0:
                        with control_placeholder.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                if st.button("⏹️ 응답 중단", key=f"interrupt_{i}_{char_idx}"):
                                    st.session_state.streaming_interrupted = True
                    
                    import time
                    time.sleep(stream_speed)
                
                # 컨트롤 제거
                control_placeholder.empty()
                
                # 마지막에 커서 제거
                response_placeholder.markdown(full_response)
                
                # 스트리밍 완료 후 플래그 리셋
                st.session_state.new_message_added = False
                st.session_state.streaming_interrupted = False
            else:
                # 일반 메시지는 즉시 표시
                st.markdown(message["content"])
            
            # 타임스탬프 추가 (마지막 메시지에만)
            if i == len(st.session_state.messages) - 1:
                current_time = datetime.now().strftime("%H:%M")
                st.caption(f"🕐 {current_time}")
    
    # 채팅 입력
    if prompt := st.chat_input("법률 관련 질문을 입력하세요..."):
        # AI 서비스 초기화 확인
        chat_service = chatbot.get_chat_service()
        if chat_service is None:
            st.error("AI 서비스를 초기화할 수 없습니다.")
            return
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 첫 번째 질문인 경우 제목 생성
        if len(st.session_state.messages) == 1:
            st.session_state.conversation_title = chatbot._generate_conversation_title(prompt)
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
                    result = chatbot.process_query(prompt)
                    st.session_state.ai_initialized = True
                
                # 에러 처리
                if result.get("generation_method") == "error":
                    st.error("❌ 응답 생성 중 오류가 발생했습니다.")
                    st.info("다시 시도해주시거나 다른 질문을 입력해주세요.")
                    response = "죄송합니다. 현재 서비스에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요."
                else:
                    # 응답 표시
                    response = result.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                    
                    # 참고 데이터가 없는 경우 특별 표시
                    if result.get("no_sources", False):
                        st.warning("⚠️ 참고 데이터를 찾을 수 없습니다")
                        st.markdown(response)
                        
                        # 검색 제안 표시
                        if "suggestion" in result:
                            st.markdown("### 💡 검색 제안")
                            for suggestion in result["suggestion"]:
                                st.markdown(f"• {suggestion}")
                    else:
                        st.markdown(response)
                    
                    # 응답 품질 정보 표시
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        if result.get("confidence", 0) > 0:
                            confidence = result['confidence']
                            if confidence >= 0.8:
                                st.success(f"신뢰도: {confidence:.1%}")
                            elif confidence >= 0.6:
                                st.warning(f"신뢰도: {confidence:.1%}")
                            else:
                                st.error(f"신뢰도: {confidence:.1%}")
                        elif result.get("no_sources", False):
                            st.error("신뢰도: 0% (참고 데이터 없음)")
                    
                    with col2:
                        if result.get("question_type"):
                            st.info(f"유형: {result['question_type']}")
                        elif result.get("no_sources", False):
                            st.info("유형: 참고 데이터 없음")
                    
                    with col3:
                        current_time = datetime.now().strftime("%H:%M")
                        st.caption(f"🕐 {current_time}")
                    
                    # 참고 자료 표시 (참고 데이터가 있는 경우에만)
                    if result.get("sources") and not result.get("no_sources", False):
                        with st.expander("📚 참고 자료", expanded=False):
                            for source in result["sources"]:
                                st.markdown(f"- {source}")
                    elif not result.get("no_sources", False):
                        st.info("ℹ️ 이 답변에는 참고 데이터가 포함되어 있지 않습니다.")
                    
                    # 생성 방법 표시
                    if result.get("generation_method"):
                        method_emoji = {
                            "rag": "🔍",
                            "llm": "🧠", 
                            "hybrid": "⚡",
                            "error": "❌",
                            "no_sources": "⚠️"
                        }
                        emoji = method_emoji.get(result["generation_method"], "🤖")
                        st.caption(f"{emoji} {result['generation_method'].upper()} 방식으로 생성됨")
                
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                response = "죄송합니다. 현재 서비스에 일시적인 문제가 있습니다."
        
        # AI 응답을 메시지에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.new_message_added = True
    
    # 푸터 - 항상 표시되도록 채팅 입력 블록 밖으로 이동
    st.markdown("---")
    
    # 상태 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        memory_info = chatbot.get_memory_info()
        memory_status = "🟢 정상" if memory_info['current_memory_mb'] < 1000 else "🟡 주의" if memory_info['current_memory_mb'] < 2000 else "🔴 위험"
        
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 12px;">
            <strong>📊 서비스 상태</strong><br>
            <span style="color: #28a745;">●</span> 정상 운영<br>
            <strong>💾 메모리:</strong> {memory_info['current_memory_mb']:.0f}MB {memory_status}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            <strong>🔒 개인정보</strong><br>
            대화 내용은 저장되지 않습니다
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            <strong>⚖️ AI 법률 상담 챗봇 v1.0</strong><br>
            Powered by Streamlit
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        main()
    except Exception as e:
        st.error(f"애플리케이션 실행 오류: {e}")
        st.exception(e)
