# -*- coding: utf-8 -*-
"""
LawFirmAI - 고급 기능 컴포넌트
지능형 질문 제안 및 문서 분석 인터페이스
"""

import gradio as gr
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SmartSuggestionSystem:
    """지능형 질문 제안 시스템"""
    
    def __init__(self):
        self.suggestion_templates = {
            "일반인": [
                "계약서 작성 시 주의사항은?",
                "이혼 절차는 어떻게 진행하나요?",
                "손해배상 청구 방법은?",
                "임대차 계약서 검토 포인트는?",
                "법적 분쟁 해결 방법은?"
            ],
            "법무팀": [
                "최신 판례 동향은?",
                "법령 개정 사항은?",
                "법률 검토 체크리스트는?",
                "소송 전략 수립 방법은?",
                "계약서 위험 요소 분석은?"
            ],
            "변호사": [
                "복잡한 법률 쟁점 분석은?",
                "판례 비교 분석 방법은?",
                "법률 검색 최적화는?",
                "소송 준비 체크리스트는?",
                "법률 의견서 작성 가이드는?"
            ],
            "법학자": [
                "법령 해석 방법론은?",
                "법학 연구 방법은?",
                "비교법 연구는?",
                "법철학적 접근은?",
                "법률사례 분석은?"
            ]
        }
    
    def create_suggestion_components(self):
        """질문 제안 컴포넌트 생성"""
        suggestion_buttons = []
        
        for i in range(5):
            btn = gr.Button(
                f"제안 질문 {i+1}",
                variant="secondary",
                size="sm",
                visible=False
            )
            suggestion_buttons.append(btn)
        
        refresh_suggestions_btn = gr.Button("새로고침", variant="secondary", size="sm")
        suggestion_count = gr.Slider(1, 5, value=3, step=1, label="제안 개수")
        auto_refresh = gr.Checkbox(label="자동 새로고침", value=True)
        
        return suggestion_buttons, refresh_suggestions_btn, suggestion_count, auto_refresh
    
    def get_suggestions(self, user_type: str, interest_areas: List[str], conversation_history: List) -> List[str]:
        """사용자 유형과 관심사에 따른 질문 제안"""
        try:
            suggestions = self.suggestion_templates.get(user_type, self.suggestion_templates["일반인"]).copy()
            
            # 관심 분야별 제안 추가
            if "민법" in interest_areas:
                suggestions.extend([
                    "민법상 계약 해제 요건은?",
                    "불법행위 성립 요건은?"
                ])
            
            if "형법" in interest_areas:
                suggestions.extend([
                    "형사처벌 요건은?",
                    "형사절차는 어떻게 진행되나요?"
                ])
            
            if "상법" in interest_areas:
                suggestions.extend([
                    "회사법상 주주권리는?",
                    "상법상 책임 제한은?"
                ])
            
            return suggestions[:5]  # 최대 5개 제안
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return ["계약서 검토 요청", "판례 검색", "법령 해설"]

class DocumentAnalysisInterface:
    """문서 분석 인터페이스"""
    
    def __init__(self):
        self.analysis_types = {
            "contract": "계약서 검토",
            "legal": "법률 문서 분석",
            "precedent": "판례 분석",
            "regulation": "법령 분석"
        }
        
        self.analysis_options = [
            "위험 요소 분석",
            "법적 근거 검토",
            "개선 제안",
            "비용 분석",
            "타임라인 분석"
        ]
    
    def create_document_analysis_components(self):
        """문서 분석 컴포넌트 생성"""
        with gr.Tab("📄 문서 분석"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="문서 업로드",
                        file_types=[".pdf", ".docx", ".txt", ".hwp"],
                        file_count="multiple"
                    )
                    
                    analysis_type = gr.Radio(
                        choices=[
                            ("계약서 검토", "contract"),
                            ("법률 문서 분석", "legal"),
                            ("판례 분석", "precedent"),
                            ("법령 분석", "regulation")
                        ],
                        value="contract",
                        label="분석 유형"
                    )
                    
                    analysis_options = gr.CheckboxGroup(
                        choices=self.analysis_options,
                        value=["위험 요소 분석", "법적 근거 검토"],
                        label="분석 옵션"
                    )
                    
                    analysis_detail = gr.Slider(
                        1, 5, value=3, step=1,
                        label="분석 상세도",
                        info="1: 간단, 5: 상세"
                    )
                    
                    analyze_btn = gr.Button("분석 시작", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    analysis_status = gr.Textbox(
                        label="분석 상태",
                        value="대기 중",
                        interactive=False
                    )
                    
                    analysis_summary = gr.Markdown(
                        label="분석 결과",
                        value="문서를 업로드하고 분석을 시작하세요."
                    )
                    
                    risk_analysis = gr.Markdown(label="위험 요소 분석")
                    legal_basis = gr.Markdown(label="법적 근거")
                    improvement_suggestions = gr.Markdown(label="개선 제안")
                    
                    with gr.Row():
                        risk_indicator = gr.HTML(
                            value="<div>위험도: <span>분석 대기</span></div>"
                        )
                        confidence_score = gr.HTML(
                            value="<div>신뢰도: <span>분석 대기</span></div>"
                        )
                    
                    analysis_stats = gr.JSON(
                        label="분석 통계",
                        value={}
                    )
                    
                    with gr.Row():
                        download_report_btn = gr.Button("보고서 다운로드", variant="secondary")
                        share_results_btn = gr.Button("결과 공유", variant="secondary")
        
        return (file_upload, analysis_type, analysis_options, analysis_detail, analyze_btn,
                analysis_status, analysis_summary, risk_analysis, legal_basis,
                improvement_suggestions, risk_indicator, confidence_score,
                analysis_stats, download_report_btn, share_results_btn)
    
    def analyze_document(self, file_path: str, analysis_type: str, options: List[str], detail_level: int) -> Dict[str, Any]:
        """문서 분석 실행"""
        try:
            # 실제 분석 로직은 기존 서비스 활용
            # 여기서는 시뮬레이션된 결과 반환
            
            analysis_result = {
                "summary": f"{analysis_type} 분석이 완료되었습니다.",
                "risks": [
                    "계약 조건 불명확",
                    "책임 제한 조항 부재",
                    "해지 조건 모호"
                ],
                "recommendations": [
                    "계약 조건을 명확히 정의",
                    "책임 제한 조항 추가",
                    "해지 조건 구체화"
                ],
                "legal_basis": [
                    "민법 제543조 (계약의 해제)",
                    "민법 제750조 (불법행위)",
                    "민법 제398조 (손해배상)"
                ],
                "risk_level": "medium",
                "confidence": 0.85,
                "processing_time": 2.5,
                "analysis_stats": {
                    "total_pages": 10,
                    "analyzed_sections": 8,
                    "risk_count": 3,
                    "recommendation_count": 3
                }
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "error": str(e),
                "risk_level": "unknown",
                "confidence": 0.0
            }

class AdvancedFeaturesManager:
    """고급 기능 관리자"""
    
    def __init__(self):
        self.suggestion_system = SmartSuggestionSystem()
        self.document_analysis = DocumentAnalysisInterface()
    
    def create_advanced_features(self):
        """고급 기능 컴포넌트 생성"""
        suggestion_components = self.suggestion_system.create_suggestion_components()
        document_components = self.document_analysis.create_document_analysis_components()
        
        return {
            "suggestions": suggestion_components,
            "document_analysis": document_components
        }

def create_production_advanced_features():
    """프로덕션 고급 기능 컴포넌트 생성"""
    try:
        features_manager = AdvancedFeaturesManager()
        
        return {
            "advanced_features": features_manager.create_advanced_features()
        }
    except Exception as e:
        logger.error(f"Error creating advanced features: {e}")
        return {}