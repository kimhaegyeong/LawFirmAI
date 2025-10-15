#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LawFirmAI - ML Enhanced Gradio Web Interface
HuggingFace Spaces 배포용 메인 애플리케이션 (ML 강화 버전)
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add source directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "source"))

import gradio as gr
from services.chat_service import ChatService
from services.rag_service import MLEnhancedRAGService
from services.search_service import MLEnhancedSearchService
from data.database import DatabaseManager
from data.vector_store import LegalVectorStore
from models.model_manager import LegalModelManager
from utils.config import Config
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def create_ml_enhanced_gradio_interface():
    """ML 강화 Gradio 인터페이스 생성"""
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize ML-enhanced services
        database = DatabaseManager(config.database_path)
        vector_store = LegalVectorStore(
            model_name="BAAI/bge-m3",
            dimension=1024,
            index_type="flat"
        )
        model_manager = LegalModelManager(config)
        
        ml_rag_service = MLEnhancedRAGService(config, model_manager, vector_store, database)
        ml_search_service = MLEnhancedSearchService(config, database, vector_store, model_manager)
        
        # Legacy chat service for compatibility
        chat_service = ChatService(config)
        
        def process_ml_enhanced_chat(message: str, history: List, 
                                   use_ml_enhanced: bool = True,
                                   quality_threshold: float = 0.7,
                                   search_type: str = "hybrid") -> tuple:
            """ML 강화 채팅 처리 함수"""
            try:
                if not message.strip():
                    return "질문을 입력해주세요.", history
                
                if use_ml_enhanced:
                    # ML 강화 RAG 서비스 사용
                    filters = {"quality_threshold": quality_threshold} if quality_threshold > 0 else None
                    response = ml_rag_service.process_query(
                        query=message,
                        top_k=5,
                        filters=filters
                    )
                    
                    # 응답 포맷팅
                    formatted_response = format_ml_enhanced_response(response)
                    
                else:
                    # 레거시 채팅 서비스 사용
                    response = chat_service.process_message(message)
                    formatted_response = response.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                
                # 대화 기록 업데이트
                history.append([message, formatted_response])
                
                return "", history
                
            except Exception as e:
                logger.error(f"ML-enhanced chat processing error: {e}")
                error_msg = f"오류가 발생했습니다: {str(e)}"
                history.append([message, error_msg])
                return "", history
        
        def format_ml_enhanced_response(response: Dict[str, Any]) -> str:
            """ML 강화 응답 포맷팅"""
            try:
                main_response = response.get("response", "죄송합니다. 응답을 생성할 수 없습니다.")
                sources = response.get("sources", [])
                ml_stats = response.get("ml_stats", {})
                
                formatted = f"**답변:**\n{main_response}\n\n"
                
                if sources:
                    formatted += "**참고 자료:**\n"
                    for i, source in enumerate(sources[:3], 1):  # 상위 3개만 표시
                        formatted += f"{i}. {source.get('title', 'Unknown')}"
                        if source.get('article_number'):
                            formatted += f" - {source['article_number']}"
                        if source.get('article_title'):
                            formatted += f"({source['article_title']})"
                        formatted += f" (품질: {source.get('quality_score', 0.0):.2f})\n"
                
                if ml_stats:
                    formatted += f"\n**ML 통계:**\n"
                    formatted += f"- 검색된 문서: {ml_stats.get('total_documents', 0)}개\n"
                    formatted += f"- 본칙 조문: {ml_stats.get('main_articles', 0)}개\n"
                    formatted += f"- 부칙 조문: {ml_stats.get('supplementary_articles', 0)}개\n"
                    formatted += f"- 평균 품질 점수: {ml_stats.get('avg_quality_score', 0.0):.3f}\n"
                
                return formatted
                
            except Exception as e:
                logger.error(f"Error formatting ML-enhanced response: {e}")
                return response.get("response", "응답 포맷팅 중 오류가 발생했습니다.")
        
        def search_documents(query: str, search_type: str = "hybrid", 
                           limit: int = 10) -> str:
            """문서 검색 함수"""
            try:
                if not query.strip():
                    return "검색어를 입력해주세요."
                
                results = ml_search_service.search_documents(
                    query=query,
                    search_type=search_type,
                    limit=limit
                )
                
                if not results:
                    return "검색 결과가 없습니다."
                
                formatted_results = f"**검색 결과 ({len(results)}개):**\n\n"
                
                for i, result in enumerate(results, 1):
                    formatted_results += f"**{i}. {result.get('title', 'Unknown')}**\n"
                    if result.get('article_number'):
                        formatted_results += f"- 조문: {result['article_number']}"
                        if result.get('article_title'):
                            formatted_results += f"({result['article_title']})"
                        formatted_results += "\n"
                    
                    formatted_results += f"- 유사도: {result.get('similarity', 0.0):.3f}\n"
                    formatted_results += f"- 품질 점수: {result.get('quality_score', 0.0):.3f}\n"
                    formatted_results += f"- 조문 유형: {result.get('article_type', 'main')}\n"
                    
                    if result.get('is_supplementary'):
                        formatted_results += "- 부칙 조문\n"
                    
                    content = result.get('content', '')
                    if content:
                        formatted_results += f"- 내용: {content[:200]}{'...' if len(content) > 200 else ''}\n"
                    
                    formatted_results += "\n"
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Document search error: {e}")
                return f"검색 중 오류가 발생했습니다: {str(e)}"
        
        def get_quality_stats() -> str:
            """품질 통계 조회 함수"""
            try:
                # 간단한 품질 통계 쿼리
                stats_query = """
                    SELECT 
                        COUNT(*) as total_articles,
                        SUM(CASE WHEN al.ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_articles,
                        AVG(al.parsing_quality_score) as avg_quality_score,
                        SUM(CASE WHEN aa.article_type = 'main' THEN 1 ELSE 0 END) as main_articles,
                        SUM(CASE WHEN aa.article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_articles
                    FROM assembly_articles aa
                    LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                """
                
                results = database.execute_query(stats_query)
                stats = results[0] if results else {}
                
                formatted_stats = "**📊 품질 통계**\n\n"
                formatted_stats += f"- 총 조문 수: {stats.get('total_articles', 0):,}개\n"
                formatted_stats += f"- ML 강화 조문: {stats.get('ml_enhanced_articles', 0):,}개\n"
                formatted_stats += f"- 평균 품질 점수: {stats.get('avg_quality_score', 0.0):.3f}\n"
                formatted_stats += f"- 본칙 조문: {stats.get('main_articles', 0):,}개\n"
                formatted_stats += f"- 부칙 조문: {stats.get('supplementary_articles', 0):,}개\n"
                
                ml_rate = (stats.get('ml_enhanced_articles', 0) / max(stats.get('total_articles', 1), 1)) * 100
                formatted_stats += f"- ML 강화 비율: {ml_rate:.1f}%\n"
                
                return formatted_stats
                
            except Exception as e:
                logger.error(f"Quality stats error: {e}")
                return f"통계 조회 중 오류가 발생했습니다: {str(e)}"
        
        def clear_history():
            """대화 기록 삭제"""
            return [], []
        
        # Create ML-enhanced Gradio interface
        with gr.Blocks(
            title="법률 AI 어시스턴트 (ML 강화)",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: auto !important;
            }
            .ml-stats {
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # ⚖️ 법률 AI 어시스턴트 (ML 강화 버전)
                
                머신러닝 기반 법률 문서 파싱과 하이브리드 검색을 활용한 고급 법률 AI 어시스턴트입니다.
                """
            )
            
            with gr.Tabs():
                # 채팅 탭
                with gr.Tab("💬 AI 채팅"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="ML 강화 대화",
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
                                stats_btn = gr.Button("품질 통계", variant="secondary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ ML 설정")
                            
                            use_ml_enhanced = gr.Checkbox(
                                label="ML 강화 모드",
                                value=True,
                                info="머신러닝 기반 검색 사용"
                            )
                            
                            quality_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="품질 임계값",
                                info="검색 품질 기준"
                            )
                            
                            search_type = gr.Dropdown(
                                choices=["hybrid", "semantic", "keyword", "supplementary", "high_quality"],
                                value="hybrid",
                                label="검색 유형",
                                info="검색 방법 선택"
                            )
                            
                            gr.Markdown(
                                """
                                ### 📚 주요 기능
                                
                                - **ML 강화 파싱**: 머신러닝 기반 조문 경계 감지
                                - **하이브리드 검색**: 의미적 + 키워드 검색 결합
                                - **품질 필터링**: 파싱 품질 기반 결과 필터링
                                - **부칙 파싱**: 본칙과 부칙 분리 파싱
                                - **실시간 통계**: ML 강화 통계 및 품질 지표
                                """
                            )
                
                # 검색 탭
                with gr.Tab("🔍 문서 검색"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            search_query = gr.Textbox(
                                label="검색어",
                                placeholder="예: 계약서, 손해배상, 부동산",
                                lines=1
                            )
                            
                            with gr.Row():
                                search_type_dropdown = gr.Dropdown(
                                    choices=["hybrid", "semantic", "keyword", "supplementary", "high_quality"],
                                    value="hybrid",
                                    label="검색 유형"
                                )
                                search_limit = gr.Slider(
                                    minimum=5,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="결과 개수"
                                )
                                search_btn = gr.Button("검색", variant="primary")
                        
                        with gr.Column(scale=3):
                            search_results = gr.Markdown(
                                label="검색 결과",
                                value="검색어를 입력하고 검색 버튼을 클릭하세요."
                            )
                
                # 통계 탭
                with gr.Tab("📊 품질 통계"):
                    with gr.Row():
                        with gr.Column():
                            quality_stats_display = gr.Markdown(
                                label="품질 통계",
                                value="품질 통계 버튼을 클릭하여 최신 통계를 확인하세요."
                            )
                            
                            refresh_stats_btn = gr.Button("통계 새로고침", variant="primary")
            
            # Event handlers
            msg.submit(
                process_ml_enhanced_chat, 
                [msg, chatbot, use_ml_enhanced, quality_threshold, search_type], 
                [msg, chatbot]
            )
            submit_btn.click(
                process_ml_enhanced_chat, 
                [msg, chatbot, use_ml_enhanced, quality_threshold, search_type], 
                [msg, chatbot]
            )
            clear_btn.click(clear_history, outputs=[chatbot, msg])
            stats_btn.click(get_quality_stats, outputs=quality_stats_display)
            
            search_btn.click(
                search_documents,
                [search_query, search_type_dropdown, search_limit],
                outputs=search_results
            )
            
            refresh_stats_btn.click(get_quality_stats, outputs=quality_stats_display)
        
        return interface
        
    except Exception as e:
        logger.error(f"Failed to create ML-enhanced Gradio interface: {e}")
        raise

def create_gradio_interface():
    """레거시 호환성을 위한 기본 Gradio 인터페이스"""
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
        logger.info("Starting LawFirmAI ML-Enhanced Gradio application...")
        
        # Create and launch ML-enhanced interface
        interface = create_ml_enhanced_gradio_interface()
        
        # Launch with HuggingFace Spaces configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start ML-enhanced application: {e}")
        # Fallback to legacy interface
        try:
            logger.info("Falling back to legacy interface...")
            interface = create_gradio_interface()
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                debug=os.getenv("DEBUG", "false").lower() == "true",
                show_error=True
            )
        except Exception as fallback_error:
            logger.error(f"Failed to start legacy application: {fallback_error}")
            sys.exit(1)

if __name__ == "__main__":
    main()
