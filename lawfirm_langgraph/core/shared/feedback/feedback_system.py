# -*- coding: utf-8 -*-
"""
사용자 피드백 수집 시스템
사용자의 피드백을 수집하고 분석하는 시스템입니다.
"""

import json
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = get_logger(__name__)

class FeedbackType(Enum):
    """피드백 유형"""
    RATING = "rating"  # 평점
    TEXT = "text"      # 텍스트 피드백
    BUG_REPORT = "bug_report"  # 버그 리포트
    FEATURE_REQUEST = "feature_request"  # 기능 요청
    GENERAL = "general"  # 일반 피드백

class FeedbackRating(Enum):
    """피드백 평점"""
    VERY_POOR = 1
    POOR = 2
    AVERAGE = 3
    GOOD = 4
    EXCELLENT = 5

@dataclass
class Feedback:
    """피드백 데이터 클래스"""
    id: str
    timestamp: datetime
    session_id: Optional[str]
    user_id: Optional[str]
    feedback_type: FeedbackType
    rating: Optional[FeedbackRating]
    text_content: Optional[str]
    question: Optional[str]
    answer: Optional[str]
    context: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]

class FeedbackCollector:
    """피드백 수집 클래스"""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        """
        피드백 수집기 초기화
        
        Args:
            db_path: 데이터베이스 경로
        """
        self.db_path = db_path
        self.logger = get_logger(__name__)
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        session_id TEXT,
                        user_id TEXT,
                        feedback_type TEXT NOT NULL,
                        rating INTEGER,
                        text_content TEXT,
                        question TEXT,
                        answer TEXT,
                        context TEXT,
                        metadata TEXT
                    )
                """)
                
                # 인덱스 생성
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
                
                conn.commit()
                self.logger.info("Feedback database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize feedback database: {e}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def submit_feedback(self, 
                       feedback_type: FeedbackType,
                       rating: Optional[FeedbackRating] = None,
                       text_content: Optional[str] = None,
                       question: Optional[str] = None,
                       answer: Optional[str] = None,
                       session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """피드백 제출"""
        
        # 피드백 ID 생성
        feedback_id = f"feedback_{int(time.time())}_{hash(str(locals())) % 10000}"
        
        # 피드백 객체 생성
        feedback = Feedback(
            id=feedback_id,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            feedback_type=feedback_type,
            rating=rating,
            text_content=text_content,
            question=question,
            answer=answer,
            context=context,
            metadata=metadata
        )
        
        # 데이터베이스에 저장
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback (
                        id, timestamp, session_id, user_id, feedback_type,
                        rating, text_content, question, answer, context, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id,
                    feedback.timestamp.isoformat(),
                    feedback.session_id,
                    feedback.user_id,
                    feedback.feedback_type.value,
                    feedback.rating.value if feedback.rating else None,
                    feedback.text_content,
                    feedback.question,
                    feedback.answer,
                    json.dumps(feedback.context) if feedback.context else None,
                    json.dumps(feedback.metadata) if feedback.metadata else None
                ))
                conn.commit()
                
            self.logger.info(f"Feedback submitted: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit feedback: {e}")
            raise
    
    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """피드백 조회"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_feedback(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback: {e}")
            return None
    
    def get_feedback_list(self, 
                         limit: int = 100,
                         offset: int = 0,
                         feedback_type: Optional[FeedbackType] = None,
                         session_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Feedback]:
        """피드백 목록 조회"""
        try:
            with self._get_db_connection() as conn:
                query = "SELECT * FROM feedback WHERE 1=1"
                params = []
                
                if feedback_type:
                    query += " AND feedback_type = ?"
                    params.append(feedback_type.value)
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_feedback(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback list: {e}")
            return []
    
    def _row_to_feedback(self, row: sqlite3.Row) -> Feedback:
        """데이터베이스 행을 Feedback 객체로 변환"""
        return Feedback(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            session_id=row["session_id"],
            user_id=row["user_id"],
            feedback_type=FeedbackType(row["feedback_type"]),
            rating=FeedbackRating(row["rating"]) if row["rating"] else None,
            text_content=row["text_content"],
            question=row["question"],
            answer=row["answer"],
            context=json.loads(row["context"]) if row["context"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """피드백 통계 반환"""
        try:
            with self._get_db_connection() as conn:
                start_date = datetime.now() - timedelta(days=days)
                
                # 전체 피드백 수
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE timestamp >= ?",
                    (start_date.isoformat(),)
                )
                total_feedback = cursor.fetchone()[0]
                
                # 피드백 유형별 통계
                cursor = conn.execute("""
                    SELECT feedback_type, COUNT(*) 
                    FROM feedback 
                    WHERE timestamp >= ? 
                    GROUP BY feedback_type
                """, (start_date.isoformat(),))
                feedback_by_type = dict(cursor.fetchall())
                
                # 평점 통계
                cursor = conn.execute("""
                    SELECT rating, COUNT(*) 
                    FROM feedback 
                    WHERE timestamp >= ? AND rating IS NOT NULL
                    GROUP BY rating
                """, (start_date.isoformat(),))
                rating_stats = dict(cursor.fetchall())
                
                # 평균 평점
                cursor = conn.execute("""
                    SELECT AVG(rating) 
                    FROM feedback 
                    WHERE timestamp >= ? AND rating IS NOT NULL
                """, (start_date.isoformat(),))
                avg_rating = cursor.fetchone()[0] or 0
                
                # 일별 피드백 수
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) 
                    FROM feedback 
                    WHERE timestamp >= ? 
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """, (start_date.isoformat(),))
                daily_feedback = dict(cursor.fetchall())
                
                return {
                    "total_feedback": total_feedback,
                    "feedback_by_type": feedback_by_type,
                    "rating_stats": rating_stats,
                    "average_rating": round(avg_rating, 2),
                    "daily_feedback": daily_feedback,
                    "period_days": days
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback stats: {e}")
            return {}
    
    def export_feedback(self, 
                       output_path: str,
                       feedback_type: Optional[FeedbackType] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> bool:
        """피드백 내보내기"""
        try:
            feedback_list = self.get_feedback_list(
                limit=10000,  # 큰 수로 설정하여 모든 피드백 가져오기
                feedback_type=feedback_type,
                start_date=start_date,
                end_date=end_date
            )
            
            # JSON 형태로 내보내기
            export_data = []
            for feedback in feedback_list:
                feedback_dict = asdict(feedback)
                feedback_dict["timestamp"] = feedback.timestamp.isoformat()
                if feedback.rating:
                    feedback_dict["rating"] = feedback.rating.value
                feedback_dict["feedback_type"] = feedback.feedback_type.value
                export_data.append(feedback_dict)
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Feedback exported to {output_path}: {len(export_data)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export feedback: {e}")
            return False

class FeedbackAnalyzer:
    """피드백 분석 클래스"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.logger = get_logger(__name__)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """텍스트 감정 분석 (간단한 키워드 기반)"""
        positive_keywords = ["좋다", "만족", "도움", "유용", "정확", "빠르다", "쉽다"]
        negative_keywords = ["나쁘다", "불만", "도움안됨", "부정확", "느리다", "어렵다", "오류"]
        
        text_lower = text.lower()
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / (positive_count + negative_count)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / (positive_count + negative_count)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_keywords": positive_count,
            "negative_keywords": negative_count
        }
    
    def analyze_feedback_trends(self, days: int = 30) -> Dict[str, Any]:
        """피드백 트렌드 분석"""
        stats = self.feedback_collector.get_feedback_stats(days)
        
        # 평점 트렌드 분석
        rating_trend = "stable"
        if stats.get("average_rating", 0) > 4.0:
            rating_trend = "positive"
        elif stats.get("average_rating", 0) < 3.0:
            rating_trend = "negative"
        
        # 피드백 유형 분석
        feedback_types = stats.get("feedback_by_type", {})
        most_common_type = max(feedback_types.items(), key=lambda x: x[1])[0] if feedback_types else None
        
        # 일별 트렌드 분석
        daily_feedback = stats.get("daily_feedback", {})
        if len(daily_feedback) >= 7:
            recent_days = list(daily_feedback.values())[:7]
            avg_recent = sum(recent_days) / len(recent_days)
            older_days = list(daily_feedback.values())[7:14]
            avg_older = sum(older_days) / len(older_days) if older_days else avg_recent
            
            if avg_recent > avg_older * 1.2:
                volume_trend = "increasing"
            elif avg_recent < avg_older * 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "insufficient_data"
        
        return {
            "rating_trend": rating_trend,
            "volume_trend": volume_trend,
            "most_common_type": most_common_type,
            "average_rating": stats.get("average_rating", 0),
            "total_feedback": stats.get("total_feedback", 0),
            "analysis_period_days": days
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        stats = self.feedback_collector.get_feedback_stats(30)
        
        # 평점 기반 제안
        avg_rating = stats.get("average_rating", 0)
        if avg_rating < 3.0:
            suggestions.append("평점이 낮습니다. 답변 품질을 개선해야 합니다.")
        elif avg_rating < 4.0:
            suggestions.append("평점이 보통입니다. 사용자 경험을 개선할 여지가 있습니다.")
        
        # 피드백 유형 기반 제안
        feedback_types = stats.get("feedback_by_type", {})
        if feedback_types.get("bug_report", 0) > 5:
            suggestions.append("버그 리포트가 많습니다. 시스템 안정성을 개선해야 합니다.")
        
        if feedback_types.get("feature_request", 0) > 10:
            suggestions.append("기능 요청이 많습니다. 새로운 기능 개발을 고려해보세요.")
        
        # 피드백 수 기반 제안
        total_feedback = stats.get("total_feedback", 0)
        if total_feedback < 10:
            suggestions.append("피드백이 부족합니다. 사용자 참여를 유도하는 방법을 고려해보세요.")
        
        return suggestions

# 전역 인스턴스
feedback_collector = FeedbackCollector()
feedback_analyzer = FeedbackAnalyzer(feedback_collector)

def get_feedback_collector() -> FeedbackCollector:
    """피드백 수집기 인스턴스 반환"""
    return feedback_collector

def get_feedback_analyzer() -> FeedbackAnalyzer:
    """피드백 분석기 인스턴스 반환"""
    return feedback_analyzer

def submit_feedback(feedback_type: FeedbackType,
                   rating: Optional[FeedbackRating] = None,
                   text_content: Optional[str] = None,
                   question: Optional[str] = None,
                   answer: Optional[str] = None,
                   session_id: Optional[str] = None,
                   user_id: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
    """피드백 제출 헬퍼 함수"""
    return feedback_collector.submit_feedback(
        feedback_type=feedback_type,
        rating=rating,
        text_content=text_content,
        question=question,
        answer=answer,
        session_id=session_id,
        user_id=user_id,
        context=context,
        metadata=metadata
    )
