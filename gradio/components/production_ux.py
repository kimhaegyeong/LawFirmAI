# -*- coding: utf-8 -*-
"""
LawFirmAI - 사용자 온보딩 및 에러 처리 시스템
프로덕션 환경을 위한 사용자 친화적 인터페이스 컴포넌트
"""

import gradio as gr
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class UserOnboardingSystem:
    """사용자 온보딩 시스템"""
    
    def __init__(self):
        self.onboarding_data = {}
        self.user_preferences = {}
        
    def create_welcome_modal(self):
        """환영 모달 생성"""
        # Modal 대신 HTML로 대체
        welcome_modal = gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 20px 0;">
            <h2>🎉 LawFirmAI에 오신 것을 환영합니다!</h2>
            <p>법률 전문가를 위한 AI 어시스턴트입니다.</p>
        </div>
        """)
        start_btn = gr.Button("시작하기", variant="primary", size="lg")
        demo_btn = gr.Button("데모 보기", variant="secondary", size="lg")
        
        return welcome_modal, start_btn, demo_btn
    
    def create_profile_setup_modal(self):
        """프로필 설정 모달"""
        profile_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 20px 0;">
            <h3>👤 프로필 설정</h3>
        </div>
        """)
        
        user_type = gr.Radio(
            choices=[
                ("일반인", "일반인"),
                ("법무팀", "법무팀"),
                ("변호사", "변호사"),
                ("법학자", "법학자")
            ],
            value="일반인",
            label="전문성 수준"
        )
        
        interest_areas = gr.CheckboxGroup(
            choices=[
                "민법", "형법", "상법", "근로기준법",
                "부동산", "금융", "지적재산권", "환경법"
            ],
            label="관심 분야",
            value=["민법"]
        )
        
        complete_btn = gr.Button("완료", variant="primary")
        skip_btn = gr.Button("건너뛰기", variant="secondary")
        
        return profile_modal, user_type, interest_areas, complete_btn, skip_btn
    
    def create_tutorial_modal(self):
        """튜토리얼 모달"""
        tutorial_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 20px 0;">
            <h3>📚 사용법 가이드</h3>
            <p>LawFirmAI 사용법을 안내해드립니다.</p>
        </div>
        """)
        
        got_it_btn = gr.Button("알겠습니다", variant="primary")
        need_help_btn = gr.Button("도움말 보기", variant="secondary")
        
        return tutorial_modal, got_it_btn, need_help_btn
    
    def create_demo_modal(self):
        """데모 모달"""
        demo_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 20px 0;">
            <h3>🎬 데모 보기</h3>
            <p>LawFirmAI의 주요 기능을 체험해보세요.</p>
        </div>
        """)
        
        try_now_btn = gr.Button("지금 체험하기", variant="primary")
        close_demo_btn = gr.Button("닫기", variant="secondary")
        
        return demo_modal, try_now_btn, close_demo_btn

class ErrorHandlingSystem:
    """에러 처리 시스템"""
    
    def __init__(self):
        self.error_logs = []
        self.recovery_suggestions = {
            "connection_error": "인터넷 연결을 확인하고 다시 시도해주세요.",
            "model_error": "모델 로딩 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "validation_error": "입력 내용을 확인하고 다시 입력해주세요.",
            "timeout_error": "요청 시간이 초과되었습니다. 다시 시도해주세요."
        }
    
    def create_error_modal(self):
        """에러 모달 생성"""
        error_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #ff6b6b; border-radius: 10px; margin: 20px 0; background-color: #ffe0e0;">
            <h3>⚠️ 오류가 발생했습니다</h3>
            <p>문제를 해결하는 방법을 안내해드립니다.</p>
        </div>
        """)
        
        retry_btn = gr.Button("다시 시도", variant="primary")
        report_btn = gr.Button("오류 신고", variant="secondary")
        
        return error_modal, retry_btn, report_btn
    
    def get_error_message(self, error_type: str, error_details: str = "") -> str:
        """에러 메시지 생성"""
        base_message = self.recovery_suggestions.get(error_type, "알 수 없는 오류가 발생했습니다.")
        
        if error_details:
            return f"{base_message}\n\n상세 정보: {error_details}"
        
        return base_message

class FeedbackSystem:
    """피드백 시스템"""
    
    def __init__(self):
        self.feedback_data = []
        self.feedback_stats = {
            "total_feedback": 0,
            "average_rating": 0.0,
            "common_issues": []
        }
    
    def create_feedback_modal(self):
        """피드백 모달 생성"""
        feedback_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #4ecdc4; border-radius: 10px; margin: 20px 0; background-color: #e8f8f5;">
            <h3>💬 피드백</h3>
            <p>서비스 개선을 위해 피드백을 남겨주세요.</p>
        </div>
        """)
        
        satisfaction = gr.Slider(1, 5, value=3, step=1, label="전체 만족도")
        accuracy = gr.Slider(1, 5, value=3, step=1, label="답변 정확성")
        speed = gr.Slider(1, 5, value=3, step=1, label="응답 속도")
        usability = gr.Slider(1, 5, value=3, step=1, label="사용 편의성")
        
        feedback_text = gr.Textbox(
            label="자유 피드백",
            placeholder="개선사항이나 의견을 자유롭게 입력해주세요",
            lines=5
        )
        
        issue_type = gr.CheckboxGroup(
            choices=[
                "답변이 부정확함",
                "응답이 너무 느림",
                "인터페이스가 복잡함",
                "기능이 부족함",
                "기타"
            ],
            label="문제 유형 (선택사항)"
        )
        
        submit_feedback_btn = gr.Button("피드백 제출", variant="primary")
        cancel_feedback_btn = gr.Button("취소", variant="secondary")
        
        return feedback_modal, satisfaction, accuracy, speed, usability, feedback_text, issue_type, submit_feedback_btn, cancel_feedback_btn
    
    def collect_feedback(self, satisfaction: int, accuracy: int, speed: int, usability: int, 
                        feedback_text: str, issue_types: List[str]) -> str:
        """피드백 수집"""
        try:
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "satisfaction": satisfaction,
                "accuracy": accuracy,
                "speed": speed,
                "usability": usability,
                "feedback_text": feedback_text,
                "issue_types": issue_types
            }
            
            self.feedback_data.append(feedback_data)
            self.feedback_stats["total_feedback"] += 1
            
            # 평균 점수 계산
            avg_rating = (satisfaction + accuracy + speed + usability) / 4
            self.feedback_stats["average_rating"] = avg_rating
            
            logger.info(f"Feedback collected: {feedback_data}")
            
            if avg_rating >= 4:
                return "감사합니다! 피드백이 도움이 됩니다. 😊"
            elif avg_rating >= 3:
                return "피드백을 바탕으로 개선하겠습니다. 👍"
            else:
                return "피드백을 바탕으로 답변을 개선하겠습니다. 더 구체적인 질문을 해주시면 도움이 됩니다. 🔧"
                
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return "피드백 제출 중 오류가 발생했습니다."

class HelpSystem:
    """도움말 시스템"""
    
    def __init__(self):
        self.help_topics = {
            "getting_started": "시작하기",
            "features": "주요 기능",
            "troubleshooting": "문제 해결",
            "contact": "문의하기"
        }
    
    def create_help_modal(self):
        """도움말 모달 생성"""
        help_modal = gr.HTML("""
        <div style="padding: 20px; border: 1px solid #667eea; border-radius: 10px; margin: 20px 0; background-color: #f0f4ff;">
            <h3>❓ 도움말</h3>
            <p>LawFirmAI 사용에 도움이 되는 정보를 제공합니다.</p>
        </div>
        """)
        
        contact_support_btn = gr.Button("문의하기", variant="primary")
        close_help_btn = gr.Button("닫기", variant="secondary")
        
        return help_modal, contact_support_btn, close_help_btn

class ProductionUXEnhancer:
    """프로덕션 UX 향상 시스템"""
    
    def __init__(self):
        self.onboarding = UserOnboardingSystem()
        self.error_handling = ErrorHandlingSystem()
        self.feedback_system = FeedbackSystem()
        self.help_system = HelpSystem()
    
    def create_complete_onboarding_flow(self):
        """완전한 온보딩 플로우 생성"""
        welcome_modal, start_btn, demo_btn = self.onboarding.create_welcome_modal()
        profile_modal, user_type, interest_areas, complete_btn, skip_btn = self.onboarding.create_profile_setup_modal()
        tutorial_modal, got_it_btn, need_help_btn = self.onboarding.create_tutorial_modal()
        demo_modal, try_now_btn, close_demo_btn = self.onboarding.create_demo_modal()
        
        return {
            "welcome": (welcome_modal, start_btn, demo_btn),
            "profile": (profile_modal, user_type, interest_areas, complete_btn, skip_btn),
            "tutorial": (tutorial_modal, got_it_btn, need_help_btn),
            "demo": (demo_modal, try_now_btn, close_demo_btn)
        }
    
    def create_error_handling_components(self):
        """에러 처리 컴포넌트 생성"""
        error_modal, retry_btn, report_btn = self.error_handling.create_error_modal()
        
        return {
            "error_modal": error_modal,
            "retry_btn": retry_btn,
            "report_btn": report_btn
        }
    
    def create_feedback_components(self):
        """피드백 컴포넌트 생성"""
        feedback_modal, satisfaction, accuracy, speed, usability, feedback_text, issue_type, submit_feedback_btn, cancel_feedback_btn = self.feedback_system.create_feedback_modal()
        
        return {
            "feedback_modal": feedback_modal,
            "satisfaction": satisfaction,
            "accuracy": accuracy,
            "speed": speed,
            "usability": usability,
            "feedback_text": feedback_text,
            "issue_type": issue_type,
            "submit_feedback_btn": submit_feedback_btn,
            "cancel_feedback_btn": cancel_feedback_btn
        }
    
    def create_help_components(self):
        """도움말 컴포넌트 생성"""
        help_modal, contact_support_btn, close_help_btn = self.help_system.create_help_modal()
        
        return {
            "help_modal": help_modal,
            "contact_support_btn": contact_support_btn,
            "close_help_btn": close_help_btn
        }

def create_production_ux_components():
    """프로덕션 UX 컴포넌트 생성"""
    try:
        ux_enhancer = ProductionUXEnhancer()
        
        return {
            "onboarding": ux_enhancer.create_complete_onboarding_flow(),
            "error_handling": ux_enhancer.create_error_handling_components(),
            "feedback": ux_enhancer.create_feedback_components(),
            "help": ux_enhancer.create_help_components()
        }
    except Exception as e:
        logger.error(f"Error creating UX components: {e}")
        return {}