# -*- coding: utf-8 -*-
"""
사용자 피드백 시스템
답변 품질에 대한 사용자 피드백 수집 및 분석
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import sqlite3
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import statistics

logger = get_logger(__name__)


class FeedbackCollector:
    """사용자 피드백 수집 및 관리 시스템"""
    
    def __init__(self, feedback_db_path: str = "./data/feedback.db"):
        self.feedback_db_path = feedback_db_path
        self.logger = get_logger(__name__)
        self._initialize_feedback_db()
    
    def _initialize_feedback_db(self):
        """피드백 데이터베이스 초기화"""
        Path(self.feedback_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()
        
        # 피드백 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                query_type TEXT NOT NULL,
                answer TEXT NOT NULL,
                overall_rating INTEGER NOT NULL CHECK(overall_rating >= 1 AND overall_rating <= 5),
                accuracy_rating INTEGER NOT NULL CHECK(accuracy_rating >= 1 AND accuracy_rating <= 5),
                clarity_rating INTEGER NOT NULL CHECK(clarity_rating >= 1 AND clarity_rating <= 5),
                completeness_rating INTEGER NOT NULL CHECK(completeness_rating >= 1 AND completeness_rating <= 5),
                legal_accuracy_rating INTEGER NOT NULL CHECK(legal_accuracy_rating >= 1 AND legal_accuracy_rating <= 5),
                helpful_rating INTEGER NOT NULL CHECK(helpful_rating >= 1 AND helpful_rating <= 5),
                comments TEXT,
                processing_time REAL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 피드백 통계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type TEXT NOT NULL,
                total_feedback INTEGER DEFAULT 0,
                avg_overall_rating REAL DEFAULT 0.0,
                avg_accuracy_rating REAL DEFAULT 0.0,
                avg_clarity_rating REAL DEFAULT 0.0,
                avg_completeness_rating REAL DEFAULT 0.0,
                avg_legal_accuracy_rating REAL DEFAULT 0.0,
                avg_helpful_rating REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON user_feedback(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_query_type ON user_feedback(query_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON user_feedback(created_at)')
        
        conn.commit()
        conn.close()
        self.logger.info("Feedback database initialized")
    
    def submit_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """피드백 제출"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_feedback (
                    session_id, query, query_type, answer,
                    overall_rating, accuracy_rating, clarity_rating,
                    completeness_rating, legal_accuracy_rating, helpful_rating,
                    comments, processing_time, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data.get("session_id", ""),
                feedback_data.get("query", ""),
                feedback_data.get("query_type", ""),
                feedback_data.get("answer", ""),
                feedback_data.get("overall_rating", 3),
                feedback_data.get("accuracy_rating", 3),
                feedback_data.get("clarity_rating", 3),
                feedback_data.get("completeness_rating", 3),
                feedback_data.get("legal_accuracy_rating", 3),
                feedback_data.get("helpful_rating", 3),
                feedback_data.get("comments", ""),
                feedback_data.get("processing_time", 0.0),
                feedback_data.get("confidence", 0.0)
            ))
            
            conn.commit()
            conn.close()
            
            # 통계 업데이트
            self._update_feedback_stats(feedback_data.get("query_type", ""))
            
            self.logger.info(f"Feedback submitted for session: {feedback_data.get('session_id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting feedback: {e}")
            return False
    
    def _update_feedback_stats(self, query_type: str):
        """피드백 통계 업데이트"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            
            # 해당 질문 유형의 모든 피드백 조회
            cursor.execute('''
                SELECT overall_rating, accuracy_rating, clarity_rating,
                       completeness_rating, legal_accuracy_rating, helpful_rating
                FROM user_feedback 
                WHERE query_type = ?
            ''', (query_type,))
            
            ratings = cursor.fetchall()
            
            if ratings:
                # 평균 계산
                overall_avg = statistics.mean([r[0] for r in ratings])
                accuracy_avg = statistics.mean([r[1] for r in ratings])
                clarity_avg = statistics.mean([r[2] for r in ratings])
                completeness_avg = statistics.mean([r[3] for r in ratings])
                legal_accuracy_avg = statistics.mean([r[4] for r in ratings])
                helpful_avg = statistics.mean([r[5] for r in ratings])
                
                # 통계 업데이트 또는 삽입
                cursor.execute('''
                    INSERT OR REPLACE INTO feedback_stats (
                        query_type, total_feedback, avg_overall_rating,
                        avg_accuracy_rating, avg_clarity_rating,
                        avg_completeness_rating, avg_legal_accuracy_rating,
                        avg_helpful_rating, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    query_type, len(ratings), overall_avg,
                    accuracy_avg, clarity_avg, completeness_avg,
                    legal_accuracy_avg, helpful_avg
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating feedback stats: {e}")
    
    def get_feedback_stats(self, query_type: Optional[str] = None) -> Dict[str, Any]:
        """피드백 통계 조회"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if query_type:
                cursor.execute('''
                    SELECT * FROM feedback_stats WHERE query_type = ?
                ''', (query_type,))
                row = cursor.fetchone()
                
                if row:
                    stats = dict(row)
                else:
                    stats = {
                        "query_type": query_type,
                        "total_feedback": 0,
                        "avg_overall_rating": 0.0,
                        "avg_accuracy_rating": 0.0,
                        "avg_clarity_rating": 0.0,
                        "avg_completeness_rating": 0.0,
                        "avg_legal_accuracy_rating": 0.0,
                        "avg_helpful_rating": 0.0
                    }
            else:
                # 전체 통계
                cursor.execute('SELECT COUNT(*) FROM user_feedback')
                total_feedback = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT AVG(overall_rating), AVG(accuracy_rating), AVG(clarity_rating),
                           AVG(completeness_rating), AVG(legal_accuracy_rating), AVG(helpful_rating)
                    FROM user_feedback
                ''')
                row = cursor.fetchone()
                
                stats = {
                    "total_feedback": total_feedback,
                    "avg_overall_rating": row[0] or 0.0,
                    "avg_accuracy_rating": row[1] or 0.0,
                    "avg_clarity_rating": row[2] or 0.0,
                    "avg_completeness_rating": row[3] or 0.0,
                    "avg_legal_accuracy_rating": row[4] or 0.0,
                    "avg_helpful_rating": row[5] or 0.0
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting feedback stats: {e}")
            return {}
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 피드백 조회"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_feedback 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            feedback_list = [dict(row) for row in rows]
            
            conn.close()
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Error getting recent feedback: {e}")
            return []
    
    def get_feedback_by_query_type(self, query_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """질문 유형별 피드백 조회"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_feedback 
                WHERE query_type = ?
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (query_type, limit))
            
            rows = cursor.fetchall()
            feedback_list = [dict(row) for row in rows]
            
            conn.close()
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Error getting feedback by query type: {e}")
            return []
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """피드백 트렌드 분석"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 최근 30일 피드백
            cursor.execute('''
                SELECT query_type, AVG(overall_rating) as avg_rating, COUNT(*) as count
                FROM user_feedback 
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY query_type
                ORDER BY avg_rating DESC
            ''')
            
            recent_stats = [dict(row) for row in cursor.fetchall()]
            
            # 전체 기간 피드백
            cursor.execute('''
                SELECT query_type, AVG(overall_rating) as avg_rating, COUNT(*) as count
                FROM user_feedback 
                GROUP BY query_type
                ORDER BY avg_rating DESC
            ''')
            
            overall_stats = [dict(row) for row in cursor.fetchall()]
            
            # 가장 개선이 필요한 영역
            cursor.execute('''
                SELECT AVG(accuracy_rating) as accuracy,
                       AVG(clarity_rating) as clarity,
                       AVG(completeness_rating) as completeness,
                       AVG(legal_accuracy_rating) as legal_accuracy,
                       AVG(helpful_rating) as helpful
                FROM user_feedback
            ''')
            
            row = cursor.fetchone()
            improvement_areas = {
                "accuracy": row["accuracy"] or 0.0,
                "clarity": row["clarity"] or 0.0,
                "completeness": row["completeness"] or 0.0,
                "legal_accuracy": row["legal_accuracy"] or 0.0,
                "helpful": row["helpful"] or 0.0
            }
            
            # 가장 낮은 점수 영역 찾기
            lowest_area = min(improvement_areas.items(), key=lambda x: x[1])
            
            conn.close()
            
            return {
                "recent_stats": recent_stats,
                "overall_stats": overall_stats,
                "improvement_areas": improvement_areas,
                "lowest_performing_area": lowest_area,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback trends: {e}")
            return {}


class FeedbackAnalyzer:
    """피드백 분석 및 개선 제안 시스템"""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.logger = get_logger(__name__)
    
    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """개선 제안 생성"""
        try:
            trends = self.feedback_collector.analyze_feedback_trends()
            suggestions = []
            
            if not trends:
                return suggestions
            
            # 가장 낮은 성능 영역에 대한 제안
            lowest_area = trends.get("lowest_performing_area", ("", 0.0))
            if lowest_area[1] < 3.0:  # 3점 미만인 경우
                suggestions.append({
                    "area": lowest_area[0],
                    "current_score": lowest_area[1],
                    "suggestion": self._get_improvement_suggestion(lowest_area[0]),
                    "priority": "high"
                })
            
            # 질문 유형별 개선 제안
            for stat in trends.get("recent_stats", []):
                if stat["avg_rating"] < 3.5:  # 3.5점 미만인 경우
                    suggestions.append({
                        "query_type": stat["query_type"],
                        "current_score": stat["avg_rating"],
                        "suggestion": f"{stat['query_type']} 영역의 답변 품질을 개선해야 합니다.",
                        "priority": "medium"
                    })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return []
    
    def _get_improvement_suggestion(self, area: str) -> str:
        """영역별 개선 제안"""
        suggestions = {
            "accuracy": "키워드 매핑 시스템을 강화하고 답변 정확성을 높이기 위한 검증 로직을 추가하세요.",
            "clarity": "답변 구조화 템플릿을 개선하고 단계별 설명을 강화하세요.",
            "completeness": "답변 길이를 늘리고 더 상세한 정보를 포함하도록 프롬프트를 개선하세요.",
            "legal_accuracy": "법률 용어 사전을 확장하고 법조문 인용을 강화하세요.",
            "helpful": "사용자 관점에서 더 실용적인 조언을 제공하도록 답변을 개선하세요."
        }
        
        return suggestions.get(area, "해당 영역의 품질을 개선하기 위한 구체적인 방안을 검토하세요.")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """품질 메트릭 조회"""
        try:
            stats = self.feedback_collector.get_feedback_stats()
            
            # 품질 등급 계산
            overall_score = stats.get("avg_overall_rating", 0.0)
            if overall_score >= 4.0:
                quality_grade = "우수"
            elif overall_score >= 3.0:
                quality_grade = "양호"
            elif overall_score >= 2.0:
                quality_grade = "보통"
            else:
                quality_grade = "개선 필요"
            
            return {
                "overall_score": overall_score,
                "quality_grade": quality_grade,
                "total_feedback": stats.get("total_feedback", 0),
                "detailed_scores": {
                    "accuracy": stats.get("avg_accuracy_rating", 0.0),
                    "clarity": stats.get("avg_clarity_rating", 0.0),
                    "completeness": stats.get("avg_completeness_rating", 0.0),
                    "legal_accuracy": stats.get("avg_legal_accuracy_rating", 0.0),
                    "helpful": stats.get("avg_helpful_rating", 0.0)
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality metrics: {e}")
            return {}
