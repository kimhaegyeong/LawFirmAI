/**
 * 피드백 관련 타입 정의
 */

export interface FeedbackRequest {
  session_id: string;
  message_id?: string;
  rating: number; // 1-5
  comment?: string;
  feedback_type?: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  session_id: string;
  message_id?: string;
  rating: number;
  comment?: string;
  timestamp: string;
}

