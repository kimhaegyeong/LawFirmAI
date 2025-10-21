# -*- coding: utf-8 -*-
"""
LawFirmAI - 사용자 온보딩 시스템
신규 사용자를 위한 친화적인 가이드 및 튜토리얼
"""

import gradio as gr
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class UserOnboardingSystem:
    """사용자 온보딩 시스템"""
    
    def __init__(self):
        self.onboarding_data = {}
        self.user_preferences = {}
        self.tutorial_steps = [
            "welcome",
            "profile_setup", 
            "feature_tour",
            "first_question",
            "complete"
        ]
        self.current_step = 0
    
    def create_welcome_section(self):
        """환영 섹션 생성"""
        welcome_html = """
        <div class="welcome-section">
            <div class="welcome-icon">⚖️</div>
            <h2>LawFirmAI에 오신 것을 환영합니다!</h2>
            <p>법률 전문가와 일반인을 위한 AI 어시스턴트입니다.</p>
            <div class="welcome-features">
                <div class="feature-item">
                    <span class="feature-icon">📄</span>
                    <span>문서 분석 및 검토</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🔍</span>
                    <span>판례 및 법령 검색</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">💼</span>
                    <span>법률 상담 및 조언</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🎯</span>
                    <span>맞춤형 답변 제공</span>
                </div>
            </div>
        </div>
        """
        
        start_btn = gr.Button("시작하기", variant="primary", size="lg")
        skip_btn = gr.Button("건너뛰기", variant="secondary")
        
        return welcome_html, start_btn, skip_btn
    
    def create_profile_setup_section(self):
        """프로필 설정 섹션"""
        profile_html = """
        <div class="profile-section">
            <h3>👤 프로필 설정</h3>
            <p>사용자 유형을 선택하면 더 적합한 답변을 받을 수 있습니다.</p>
        </div>
        """
        
        user_type = gr.Radio(
            choices=[
                ("일반인", "일반적인 법률 질문과 상담"),
                ("법무팀", "기업 법무 업무 및 계약서 검토"),
                ("변호사", "법률 전문가용 고급 기능")
            ],
            value="일반인",
            label="사용자 유형"
        )
        
        interest_areas = gr.CheckboxGroup(
            choices=[
                "민법", "형법", "상법", "근로기준법",
                "부동산", "금융", "지적재산권", "환경법"
            ],
            label="관심 분야 (선택사항)",
            value=[]
        )
        
        complete_btn = gr.Button("완료", variant="primary")
        back_btn = gr.Button("이전", variant="secondary")
        
        return profile_html, user_type, interest_areas, complete_btn, back_btn
    
    def create_feature_tour_section(self):
        """기능 투어 섹션"""
        tour_html = """
        <div class="tour-section">
            <h3>🚀 주요 기능 소개</h3>
            <div class="tour-steps">
                <div class="tour-step">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <h4>질문 입력</h4>
                        <p>법률 관련 질문을 자연어로 입력하세요</p>
                    </div>
                </div>
                <div class="tour-step">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <h4>빠른 질문</h4>
                        <p>자주 묻는 질문 버튼을 활용하세요</p>
                    </div>
                </div>
                <div class="tour-step">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <h4>문서 분석</h4>
                        <p>계약서나 법률 문서를 업로드하여 분석하세요</p>
                    </div>
                </div>
                <div class="tour-step">
                    <div class="step-number">4</div>
                    <div class="step-content">
                        <h4>맞춤 설정</h4>
                        <p>사용자 유형과 관심 분야를 설정하세요</p>
                    </div>
                </div>
            </div>
        </div>
        """
        
        next_btn = gr.Button("다음", variant="primary")
        back_btn = gr.Button("이전", variant="secondary")
        
        return tour_html, next_btn, back_btn
    
    def create_first_question_section(self):
        """첫 질문 섹션"""
        question_html = """
        <div class="first-question-section">
            <h3>💡 첫 질문을 해보세요!</h3>
            <p>아래 예시 질문 중 하나를 선택하거나 직접 질문을 입력해보세요.</p>
            <div class="example-questions">
                <button class="example-question-btn" onclick="setQuestion('계약서 작성 시 주의사항은?')">
                    계약서 작성 시 주의사항은?
                </button>
                <button class="example-question-btn" onclick="setQuestion('이혼 절차는 어떻게 진행하나요?')">
                    이혼 절차는 어떻게 진행하나요?
                </button>
                <button class="example-question-btn" onclick="setQuestion('손해배상 청구 방법은?')">
                    손해배상 청구 방법은?
                </button>
                <button class="example-question-btn" onclick="setQuestion('임대차 계약서 검토 포인트는?')">
                    임대차 계약서 검토 포인트는?
                </button>
            </div>
        </div>
        """
        
        complete_btn = gr.Button("완료", variant="primary")
        back_btn = gr.Button("이전", variant="secondary")
        
        return question_html, complete_btn, back_btn
    
    def create_completion_section(self):
        """완료 섹션"""
        completion_html = """
        <div class="completion-section">
            <div class="completion-icon">🎉</div>
            <h3>설정이 완료되었습니다!</h3>
            <p>이제 LawFirmAI를 자유롭게 사용하실 수 있습니다.</p>
            <div class="completion-tips">
                <h4>💡 사용 팁</h4>
                <ul>
                    <li>구체적이고 명확한 질문을 하시면 더 정확한 답변을 받을 수 있습니다</li>
                    <li>문서를 업로드하여 분석을 요청할 수 있습니다</li>
                    <li>빠른 질문 버튼을 활용해보세요</li>
                    <li>설정에서 사용자 유형을 변경할 수 있습니다</li>
                </ul>
            </div>
        </div>
        """
        
        start_using_btn = gr.Button("사용 시작하기", variant="primary", size="lg")
        
        return completion_html, start_using_btn
    
    def get_onboarding_css(self):
        """온보딩용 CSS 스타일"""
        return """
        .welcome-section {
            text-align: center;
            padding: 40px 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .welcome-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .welcome-section h2 {
            color: #2c3e50;
            margin: 0 0 15px 0;
            font-size: 2rem;
        }
        
        .welcome-section p {
            color: #6c757d;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        
        .welcome-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        .feature-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        
        .profile-section, .tour-section, .first-question-section, .completion-section {
            padding: 30px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .tour-steps {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .tour-step {
            display: flex;
            align-items: flex-start;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        .step-number {
            width: 40px;
            height: 40px;
            background: #007bff;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
        
        .step-content h4 {
            margin: 0 0 5px 0;
            color: #2c3e50;
        }
        
        .step-content p {
            margin: 0;
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .example-questions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        
        .example-question-btn {
            padding: 15px 20px;
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            text-align: left;
        }
        
        .example-question-btn:hover {
            background: #bbdefb;
            transform: translateY(-2px);
        }
        
        .completion-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .completion-section h3 {
            color: #2c3e50;
            margin: 0 0 15px 0;
        }
        
        .completion-tips {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        
        .completion-tips h4 {
            color: #28a745;
            margin: 0 0 15px 0;
        }
        
        .completion-tips ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .completion-tips li {
            margin: 8px 0;
            color: #495057;
        }
        
        @media (max-width: 768px) {
            .welcome-features {
                grid-template-columns: 1fr;
            }
            
            .tour-steps {
                grid-template-columns: 1fr;
            }
            
            .example-questions {
                grid-template-columns: 1fr;
            }
            
            .welcome-section, .profile-section, .tour-section, 
            .first-question-section, .completion-section {
                padding: 20px;
                margin: 10px 0;
            }
        }
        """
    
    def create_onboarding_flow(self):
        """완전한 온보딩 플로우 생성"""
        # CSS 추가
        css = self.get_onboarding_css()
        
        # 각 섹션 생성
        welcome_html, start_btn, skip_btn = self.create_welcome_section()
        profile_html, user_type, interest_areas, complete_btn, back_btn = self.create_profile_setup_section()
        tour_html, next_btn, back_btn2 = self.create_feature_tour_section()
        question_html, complete_btn2, back_btn3 = self.create_first_question_section()
        completion_html, start_using_btn = self.create_completion_section()
        
        return {
            "css": css,
            "welcome": (welcome_html, start_btn, skip_btn),
            "profile": (profile_html, user_type, interest_areas, complete_btn, back_btn),
            "tour": (tour_html, next_btn, back_btn2),
            "question": (question_html, complete_btn2, back_btn3),
            "completion": (completion_html, start_using_btn)
        }

def create_onboarding_components():
    """온보딩 컴포넌트 생성"""
    try:
        onboarding_system = UserOnboardingSystem()
        return onboarding_system.create_onboarding_flow()
    except Exception as e:
        logger.error(f"Error creating onboarding components: {e}")
        return {}
