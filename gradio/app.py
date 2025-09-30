#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LawFirmAI - Gradio Web Interface
HuggingFace Spaces 배포용 메인 애플리케이션
"""

import os
import sys
import logging
from pathlib import Path

# Add source directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "source"))

import gradio as gr
from services.chat_service import ChatService
from utils.config import Config
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize chat service
        chat_service = ChatService(config)
        
        def process_chat(message, history):
            """채팅 처리 함수"""
            try:
                if not message.strip():
                    return "질문을 입력해주세요."
                
                # Process message through chat service
                response = chat_service.process_message(message)
                
                return response.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                
            except Exception as e:
                logger.error(f"Chat processing error: {e}")
                return f"오류가 발생했습니다: {str(e)}"
        
        def clear_history():
            """대화 기록 삭제"""
            return [], []
        
        # Create Gradio interface
        with gr.Blocks(
            title="법률 AI 어시스턴트",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # ⚖️ 법률 AI 어시스턴트
                
                법률 관련 질문에 답변해드리는 AI 어시스턴트입니다.
                판례, 법령, Q&A 데이터베이스를 기반으로 정확한 정보를 제공합니다.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="대화",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="질문을 입력하세요",
                            placeholder="예: 계약서에서 주의해야 할 조항은 무엇인가요?",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("전송", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("대화 초기화", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ## 📚 주요 기능
                        
                        - **판례 검색**: 법원 판례 검색 및 분석
                        - **법령 해설**: 법령 조문 해석 및 설명
                        - **계약서 분석**: 계약서 검토 및 위험 요소 분석
                        - **Q&A**: 자주 묻는 법률 질문 답변
                        
                        ## 💡 사용 팁
                        
                        - 구체적인 질문을 해주세요
                        - 관련 법령이나 판례를 언급해주세요
                        - 복잡한 질문은 단계별로 나누어 질문하세요
                        """
                    )
            
            # Event handlers
            msg.submit(process_chat, [msg, chatbot], [chatbot, msg])
            submit_btn.click(process_chat, [msg, chatbot], [chatbot, msg])
            clear_btn.click(clear_history, outputs=[chatbot, msg])
        
        return interface
        
    except Exception as e:
        logger.error(f"Failed to create Gradio interface: {e}")
        raise

def main():
    """메인 실행 함수"""
    try:
        logger.info("Starting LawFirmAI Gradio application...")
        
        # Create and launch interface
        interface = create_gradio_interface()
        
        # Launch with HuggingFace Spaces configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
