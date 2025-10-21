# -*- coding: utf-8 -*-
"""
LawFirmAI - 사용자 친화적 에러 처리 시스템
일반 사용자를 위한 명확하고 도움이 되는 에러 메시지
"""

import gradio as gr
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class UserFriendlyErrorHandler:
    """사용자 친화적 에러 처리 시스템"""
    
    def __init__(self):
        self.error_logs = []
        self.recovery_suggestions = {
            "connection_error": {
                "title": "연결 문제",
                "message": "인터넷 연결을 확인하고 다시 시도해주세요.",
                "actions": [
                    "인터넷 연결 상태를 확인하세요",
                    "페이지를 새로고침해보세요",
                    "잠시 후 다시 시도해주세요"
                ],
                "icon": "🌐"
            },
            "model_error": {
                "title": "일시적 오류",
                "message": "AI 모델 처리 중 문제가 발생했습니다.",
                "actions": [
                    "잠시 기다린 후 질문을 다시 입력하세요",
                    "질문을 더 간단하게 작성해보세요",
                    "다른 질문을 시도해보세요"
                ],
                "icon": "🤖"
            },
            "validation_error": {
                "title": "입력 오류",
                "message": "입력하신 내용을 확인해주세요.",
                "actions": [
                    "질문을 다시 확인해주세요",
                    "구체적인 질문을 입력해주세요",
                    "부적절한 내용이 포함되지 않았는지 확인하세요"
                ],
                "icon": "✏️"
            },
            "timeout_error": {
                "title": "응답 시간 초과",
                "message": "요청 처리 시간이 초과되었습니다.",
                "actions": [
                    "질문을 더 간단하게 작성해보세요",
                    "잠시 후 다시 시도해주세요",
                    "다른 질문을 시도해보세요"
                ],
                "icon": "⏰"
            },
            "file_error": {
                "title": "파일 처리 오류",
                "message": "업로드한 파일을 처리할 수 없습니다.",
                "actions": [
                    "파일 형식을 확인해주세요 (PDF, DOCX, TXT 지원)",
                    "파일 크기를 확인해주세요 (10MB 이하)",
                    "파일이 손상되지 않았는지 확인해주세요"
                ],
                "icon": "📄"
            },
            "permission_error": {
                "title": "권한 오류",
                "message": "요청을 처리할 권한이 없습니다.",
                "actions": [
                    "페이지를 새로고침해보세요",
                    "다른 브라우저를 시도해보세요",
                    "잠시 후 다시 시도해주세요"
                ],
                "icon": "🔒"
            },
            "unknown_error": {
                "title": "알 수 없는 오류",
                "message": "예상치 못한 문제가 발생했습니다.",
                "actions": [
                    "페이지를 새로고침해보세요",
                    "잠시 후 다시 시도해주세요",
                    "문제가 지속되면 관리자에게 문의하세요"
                ],
                "icon": "❓"
            }
        }
    
    def create_error_display(self, error_type: str, error_details: str = "") -> str:
        """에러 표시 HTML 생성"""
        error_info = self.recovery_suggestions.get(error_type, self.recovery_suggestions["unknown_error"])
        
        actions_html = ""
        for i, action in enumerate(error_info["actions"], 1):
            actions_html += f"<li>{action}</li>"
        
        error_html = f"""
        <div class="error-container">
            <div class="error-header">
                <span class="error-icon">{error_info['icon']}</span>
                <h3 class="error-title">{error_info['title']}</h3>
            </div>
            <div class="error-message">
                <p>{error_info['message']}</p>
            </div>
            <div class="error-actions">
                <h4>해결 방법:</h4>
                <ul>
                    {actions_html}
                </ul>
            </div>
            {f'<div class="error-details"><strong>상세 정보:</strong> {error_details}</div>' if error_details else ''}
            <div class="error-help">
                <p>문제가 지속되면 <strong>새로고침</strong> 버튼을 클릭하거나 잠시 후 다시 시도해주세요.</p>
            </div>
        </div>
        """
        
        return error_html
    
    def create_error_modal(self):
        """에러 모달 생성"""
        error_modal = gr.HTML("""
        <div class="error-modal" style="display: none;">
            <div class="error-content">
                <div class="error-header">
                    <span class="error-icon">⚠️</span>
                    <h3>오류가 발생했습니다</h3>
                </div>
                <div class="error-body">
                    <p>문제를 해결하는 방법을 안내해드립니다.</p>
                </div>
                <div class="error-footer">
                    <button class="retry-btn">다시 시도</button>
                    <button class="close-btn">닫기</button>
                </div>
            </div>
        </div>
        """)
        
        retry_btn = gr.Button("다시 시도", variant="primary")
        close_btn = gr.Button("닫기", variant="secondary")
        refresh_btn = gr.Button("새로고침", variant="secondary")
        
        return error_modal, retry_btn, close_btn, refresh_btn
    
    def create_loading_state(self):
        """로딩 상태 표시"""
        loading_html = """
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <p class="loading-text">답변을 생성하고 있습니다...</p>
            <div class="loading-tips">
                <p>💡 <strong>팁:</strong> 구체적인 질문을 하시면 더 정확한 답변을 받을 수 있습니다.</p>
            </div>
        </div>
        """
        
        return loading_html
    
    def create_success_message(self, message: str) -> str:
        """성공 메시지 생성"""
        success_html = f"""
        <div class="success-container">
            <div class="success-header">
                <span class="success-icon">✅</span>
                <h3>처리 완료</h3>
            </div>
            <div class="success-message">
                <p>{message}</p>
            </div>
        </div>
        """
        
        return success_html
    
    def get_error_css(self):
        """에러 처리용 CSS 스타일"""
        return """
        .error-container {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .error-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .error-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        
        .error-title {
            color: #e53e3e;
            margin: 0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .error-message {
            margin-bottom: 15px;
        }
        
        .error-message p {
            color: #2d3748;
            margin: 0;
            font-size: 1rem;
        }
        
        .error-actions {
            background: #f7fafc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .error-actions h4 {
            color: #2d3748;
            margin: 0 0 10px 0;
            font-size: 1rem;
            font-weight: 600;
        }
        
        .error-actions ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .error-actions li {
            color: #4a5568;
            margin: 5px 0;
            font-size: 0.9rem;
        }
        
        .error-details {
            background: #edf2f7;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 0.8rem;
            color: #718096;
        }
        
        .error-help {
            background: #e6fffa;
            border: 1px solid #81e6d9;
            border-radius: 8px;
            padding: 12px;
        }
        
        .error-help p {
            margin: 0;
            color: #234e52;
            font-size: 0.9rem;
        }
        
        .loading-container {
            text-align: center;
            padding: 40px 20px;
            background: #f7fafc;
            border-radius: 12px;
            margin: 15px 0;
        }
        
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid #3182ce;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #2d3748;
            font-size: 1.1rem;
            margin: 0 0 15px 0;
        }
        
        .loading-tips {
            background: #e6fffa;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .loading-tips p {
            margin: 0;
            color: #234e52;
            font-size: 0.9rem;
        }
        
        .success-container {
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .success-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .success-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        
        .success-header h3 {
            color: #38a169;
            margin: 0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .success-message p {
            color: #2d3748;
            margin: 0;
            font-size: 1rem;
        }
        
        .retry-btn, .close-btn, .refresh-btn {
            margin: 5px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .retry-btn {
            background: #3182ce;
            color: white;
        }
        
        .retry-btn:hover {
            background: #2c5aa0;
        }
        
        .close-btn, .refresh-btn {
            background: #718096;
            color: white;
        }
        
        .close-btn:hover, .refresh-btn:hover {
            background: #4a5568;
        }
        
        @media (max-width: 768px) {
            .error-container, .loading-container, .success-container {
                padding: 15px;
                margin: 10px 0;
            }
            
            .error-title, .success-header h3 {
                font-size: 1.1rem;
            }
            
            .loading-spinner {
                width: 30px;
                height: 30px;
            }
        }
        """
    
    def log_error(self, error_type: str, error_details: str, user_action: str = ""):
        """에러 로깅"""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_details": error_details,
            "user_action": user_action
        }
        
        self.error_logs.append(error_log)
        logger.error(f"User-friendly error: {error_log}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        if not self.error_logs:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = {}
        for log in self.error_logs:
            error_type = log["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_logs),
            "error_types": error_types,
            "recent_errors": self.error_logs[-5:] if len(self.error_logs) > 5 else self.error_logs
        }

def create_error_handling_components():
    """에러 처리 컴포넌트 생성"""
    try:
        error_handler = UserFriendlyErrorHandler()
        
        error_modal, retry_btn, close_btn, refresh_btn = error_handler.create_error_modal()
        loading_html = error_handler.create_loading_state()
        
        return {
            "error_handler": error_handler,
            "error_modal": error_modal,
            "retry_btn": retry_btn,
            "close_btn": close_btn,
            "refresh_btn": refresh_btn,
            "loading_html": loading_html,
            "css": error_handler.get_error_css()
        }
    except Exception as e:
        logger.error(f"Error creating error handling components: {e}")
        return {}
