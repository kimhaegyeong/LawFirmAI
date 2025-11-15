/**
 * 피드백 서비스
 */
import { api, extractApiError } from './api';
import type { FeedbackRequest, FeedbackResponse } from '../types/feedback';

/**
 * 피드백 전송
 */
export async function sendFeedback(data: FeedbackRequest): Promise<FeedbackResponse> {
  try {
    const response = await api.post<FeedbackResponse>('/feedback', {
      ...data,
      feedback_type: data.feedback_type || 'general',
    });
    return response.data;
  } catch (error) {
    throw extractApiError(error);
  }
}

/**
 * 긍정/부정 피드백을 숫자로 변환
 */
export function ratingToNumber(rating: 'positive' | 'negative'): number {
  return rating === 'positive' ? 5 : 1;
}

