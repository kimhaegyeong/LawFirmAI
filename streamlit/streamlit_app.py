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

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'source'))

from source.services.enhanced_chat_service import EnhancedChatService
from source.utils.config import Config

class StreamlitLegalChatbot:
    """Streamlit 법률 챗봇"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.chat_service = None
        self.history_file = "data/conversation_history.json"
        
        # 법률 분야 목록
        self.legal_fields = [
            "전체", "민사법", "형사법", "가족법", "노동법", 
            "부동산법", "소비자법", "상법", "행정법", "기타"
        ]
        
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
        self._initialize_chat_service()
    
    def _ensure_history_file(self):
        """히스토리 파일 존재 확인"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _initialize_chat_service(self):
        """채팅 서비스 초기화"""
        try:
            self.chat_service = EnhancedChatService(self.config)
            self.logger.info("EnhancedChatService 초기화 완료")
        except Exception as e:
            self.logger.error(f"채팅 서비스 초기화 실패: {e}")
            raise
    
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
    
    def process_query(self, query: str, legal_field: str = "전체") -> Dict[str, Any]:
        """쿼리 처리 및 응답 생성"""
        if not query.strip():
            return {"response": "질문을 입력해주세요.", "confidence": 0.0}
        
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
                "legal_field": "전체",
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
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = StreamlitLegalChatbot()
    if "conversation_title" not in st.session_state:
        st.session_state.conversation_title = "새로운 상담"
    
    chatbot = st.session_state.chatbot
    
    # 헤더
    st.title("🤖 AI 법률 상담 챗봇")
    st.markdown("법률 관련 질문에 답변해드립니다")
    st.caption("일상적인 법률 질문을 입력하세요. 전문적인 법률 자문은 변호사와 상담하세요.")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 새 대화 버튼
        if st.button("➕ 새 대화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_title = "새로운 상담"
            st.rerun()
        
        # 법률 분야 선택
        legal_field = st.selectbox(
            "📚 법률 분야",
            options=chatbot.legal_fields,
            index=0
        )
        
        # 빠른 질문
        st.subheader("⚡ 빠른 질문")
        for question in chatbot.quick_questions[:4]:
            if st.button(question, use_container_width=True, key=f"quick_{question}"):
                # 빠른 질문 처리
                result = chatbot.process_query(question, legal_field)
                
                # 첫 번째 질문인 경우 제목 생성
                if not st.session_state.messages:
                    st.session_state.conversation_title = chatbot._generate_conversation_title(question)
                
                # 메시지 추가
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                st.rerun()
        
        # 대화 히스토리
        st.subheader("💬 최근 대화")
        history = chatbot.load_conversation_history()
        
        if history:
            for conv in reversed(history):  # 최신순으로 표시
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
        
        # 참고 정보
        st.subheader("📖 참고 정보")
        st.markdown("""
        - 📖 [법률 용어 사전](#)
        - 🏛️ [법률 상담 기관](#)
        - 📞 [긴급 상담 전화](#)
        """)
        
        # 면책 조항
        st.warning("""
        **⚠️ 면책 조항**  
        이 AI는 일반적인 법률 정보만 제공하며,  
        전문적인 법률 자문을 대신할 수 없습니다.
        """)
    
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 채팅 입력
    if prompt := st.chat_input("법률 관련 질문을 입력하세요..."):
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
            with st.spinner("답변을 생성하고 있습니다..."):
                result = chatbot.process_query(prompt, legal_field)
                
                # 응답 표시
                response = result.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                st.markdown(response)
                
                # 추가 정보 표시
                if result.get("sources"):
                    with st.expander("📚 참고 자료"):
                        for source in result["sources"]:
                            st.markdown(f"- {source}")
                
                if result.get("confidence", 0) > 0:
                    st.caption(f"신뢰도: {result['confidence']:.2f}")
        
        # AI 응답을 메시지에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        AI 법률 상담 챗봇 v1.0 | Powered by Streamlit
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
