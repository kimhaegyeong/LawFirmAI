# -*- coding: utf-8 -*-
"""
사용자 프로필 관리자
사용자별 선호도 및 전문성 수준 관리
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..data.conversation_store import ConversationStore

logger = get_logger(__name__)


class ExpertiseLevel(Enum):
    """전문성 수준"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DetailLevel(Enum):
    """답변 상세도"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    DETAILED = "detailed"


@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    expertise_level: ExpertiseLevel
    preferred_detail_level: DetailLevel
    preferred_language: str
    interest_areas: List[str]
    device_info: Optional[Dict[str, Any]] = None
    location_info: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


class UserProfileManager:
    """사용자 프로필 관리자"""
    
    def __init__(self, conversation_store: Optional[ConversationStore] = None):
        """
        사용자 프로필 관리자 초기화
        
        Args:
            conversation_store: 대화 저장소 (None이면 새로 생성)
        """
        self.logger = get_logger(__name__)
        self.conversation_store = conversation_store or ConversationStore()
        
        # 전문성 수준별 키워드 패턴
        self.expertise_keywords = {
            ExpertiseLevel.BEGINNER: [
                "기본", "간단", "쉽게", "처음", "모름", "모르겠", "어떻게", "무엇",
                "기초", "초보", "입문", "설명", "알려주", "가르쳐"
            ],
            ExpertiseLevel.INTERMEDIATE: [
                "절차", "방법", "과정", "단계", "순서", "어떤", "언제", "어디서",
                "조건", "요건", "필요", "준비", "신청", "제출"
            ],
            ExpertiseLevel.ADVANCED: [
                "법리", "법령", "조문", "항", "호", "목", "해석", "적용", "판례",
                "대법원", "고등법원", "지방법원", "법원", "판결"
            ],
            ExpertiseLevel.EXPERT: [
                "법적", "법률", "법학", "법조", "법원", "법무", "법정", "법률가",
                "변호사", "판사", "검사", "법무사", "법학자", "법률전문가"
            ]
        }
        
        # 관심 분야 키워드
        self.interest_area_keywords = {
            "민법": ["계약", "손해배상", "불법행위", "소유권", "상속", "이혼", "가족"],
            "형법": ["범죄", "형벌", "처벌", "벌금", "징역", "사기", "절도", "강도"],
            "상법": ["회사", "주식", "법인", "이사", "주주", "상장", "합병", "분할"],
            "근로기준법": ["근로", "임금", "퇴직금", "해고", "근로시간", "휴가", "노동"],
            "부동산": ["부동산", "등기", "매매", "임대", "전세", "월세", "보증금"],
            "금융": ["대출", "이자", "연체", "담보", "보증", "신용", "카드", "은행"],
            "지적재산권": ["특허", "상표", "저작권", "디자인", "지적재산", "라이선스"],
            "세법": ["세금", "소득세", "법인세", "부가가치세", "세무", "신고", "납부"],
            "환경법": ["환경", "오염", "폐기물", "대기", "수질", "소음", "진동"],
            "의료법": ["의료", "의사", "병원", "진료", "의료사고", "의료진", "환자"]
        }
        
        # 답변 스타일 선호도 패턴
        self.response_style_patterns = {
            "간결함": ["간단", "짧게", "요약", "핵심만", "간단히"],
            "상세함": ["자세히", "구체적으로", "예시", "설명", "상세히"],
            "전문적": ["법적", "법률", "조문", "판례", "법리"],
            "실무적": ["실제", "실무", "방법", "절차", "단계"]
        }
        
        self.logger.info("UserProfileManager initialized")
    
    def create_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        사용자 프로필 생성
        
        Args:
            user_id: 사용자 ID
            profile_data: 프로필 데이터
            
        Returns:
            bool: 생성 성공 여부
        """
        try:
            # 기존 프로필 확인
            existing_profile = self.get_profile(user_id)
            if existing_profile:
                self.logger.warning(f"Profile for user {user_id} already exists")
                return False
            
            # 프로필 데이터 검증 및 변환
            validated_data = self._validate_profile_data(profile_data)
            
            # 데이터베이스에 저장
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO user_profiles 
                (user_id, expertise_level, preferred_detail_level, preferred_language, 
                 interest_areas, device_info, location_info, preferences, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    validated_data.get("expertise_level", ExpertiseLevel.BEGINNER.value),
                    validated_data.get("preferred_detail_level", DetailLevel.MEDIUM.value),
                    validated_data.get("preferred_language", "ko"),
                    json.dumps(validated_data.get("interest_areas", [])),
                    json.dumps(validated_data.get("device_info", {})),
                    json.dumps(validated_data.get("location_info", {})),
                    json.dumps(validated_data.get("preferences", {})),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.info(f"Profile created for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating profile for user {user_id}: {e}")
            return False
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        사용자 프로필 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            Optional[Dict[str, Any]]: 프로필 데이터 (없으면 None)
        """
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    "user_id": row["user_id"],
                    "expertise_level": row["expertise_level"],
                    "preferred_detail_level": row["preferred_detail_level"],
                    "preferred_language": row["preferred_language"],
                    "interest_areas": json.loads(row["interest_areas"]) if row["interest_areas"] else [],
                    "device_info": json.loads(row["device_info"]) if row["device_info"] else {},
                    "location_info": json.loads(row["location_info"]) if row["location_info"] else {},
                    "preferences": json.loads(row["preferences"]) if row["preferences"] else {},
                    "created_at": row["created_at"],
                    "last_updated": row["last_updated"]
                }
                
        except Exception as e:
            self.logger.error(f"Error getting profile for user {user_id}: {e}")
            return None
    
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        사용자 선호도 업데이트
        
        Args:
            user_id: 사용자 ID
            preferences: 업데이트할 선호도
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 기존 프로필 조회
            existing_profile = self.get_profile(user_id)
            if not existing_profile:
                self.logger.warning(f"Profile for user {user_id} not found")
                return False
            
            # 선호도 병합
            current_preferences = existing_profile.get("preferences", {})
            current_preferences.update(preferences)
            
            # 데이터베이스 업데이트
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                UPDATE user_profiles 
                SET preferences = ?, last_updated = ?
                WHERE user_id = ?
                """, (
                    json.dumps(current_preferences),
                    datetime.now().isoformat(),
                    user_id
                ))
                
                conn.commit()
                self.logger.info(f"Preferences updated for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False
    
    def track_expertise_level(self, user_id: str, question_history: List[Dict]) -> str:
        """
        질문 이력을 바탕으로 전문성 수준 추적
        
        Args:
            user_id: 사용자 ID
            question_history: 질문 이력
            
        Returns:
            str: 추정된 전문성 수준
        """
        try:
            if not question_history:
                return ExpertiseLevel.BEGINNER.value
            
            # 질문 텍스트 수집
            question_texts = []
            for q in question_history:
                if isinstance(q, dict):
                    question_texts.append(q.get("user_query", ""))
                else:
                    question_texts.append(str(q))
            
            combined_text = " ".join(question_texts).lower()
            
            # 각 전문성 수준별 점수 계산
            expertise_scores = {}
            for level, keywords in self.expertise_keywords.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                expertise_scores[level] = score
            
            # 가장 높은 점수의 전문성 수준 선택
            max_level = max(expertise_scores.items(), key=lambda x: x[1])
            
            # 점수가 너무 낮으면 초보자로 분류
            if max_level[1] < 2:
                estimated_level = ExpertiseLevel.BEGINNER
            else:
                estimated_level = max_level[0]
            
            # 프로필 업데이트
            self.update_preferences(user_id, {"estimated_expertise_level": estimated_level.value})
            
            self.logger.info(f"Expertise level estimated for user {user_id}: {estimated_level.value}")
            return estimated_level.value
            
        except Exception as e:
            self.logger.error(f"Error tracking expertise level for user {user_id}: {e}")
            return ExpertiseLevel.BEGINNER.value
    
    def get_personalized_context(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        사용자별 개인화된 컨텍스트 제공
        
        Args:
            user_id: 사용자 ID
            query: 질문
            
        Returns:
            Dict[str, Any]: 개인화된 컨텍스트
        """
        try:
            profile = self.get_profile(user_id)
            if not profile:
                # 기본 프로필 생성
                self.create_profile(user_id, {})
                profile = self.get_profile(user_id)
            
            # 관심 분야 분석
            interest_areas = self._analyze_interest_areas(query, profile.get("interest_areas", []))
            
            # 답변 스타일 결정
            response_style = self._determine_response_style(query, profile)
            
            # 전문성 수준에 따른 컨텍스트 조정
            expertise_context = self._get_expertise_context(profile.get("expertise_level", ExpertiseLevel.BEGINNER.value))
            
            return {
                "user_id": user_id,
                "expertise_level": profile.get("expertise_level", ExpertiseLevel.BEGINNER.value),
                "preferred_detail_level": profile.get("preferred_detail_level", DetailLevel.MEDIUM.value),
                "interest_areas": interest_areas,
                "response_style": response_style,
                "expertise_context": expertise_context,
                "preferences": profile.get("preferences", {}),
                "personalization_score": self._calculate_personalization_score(profile, query)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting personalized context for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "expertise_level": ExpertiseLevel.BEGINNER.value,
                "preferred_detail_level": DetailLevel.MEDIUM.value,
                "interest_areas": [],
                "response_style": "medium",
                "expertise_context": {},
                "preferences": {},
                "personalization_score": 0.0
            }
    
    def update_interest_areas(self, user_id: str, query: str) -> bool:
        """
        질문을 바탕으로 관심 분야 업데이트
        
        Args:
            user_id: 사용자 ID
            query: 질문
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            profile = self.get_profile(user_id)
            if not profile:
                return False
            
            # 질문에서 관심 분야 추출
            detected_areas = self._detect_interest_areas_from_query(query)
            
            # 기존 관심 분야와 병합
            current_areas = profile.get("interest_areas", [])
            updated_areas = list(set(current_areas + detected_areas))
            
            # 프로필 업데이트
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                UPDATE user_profiles 
                SET interest_areas = ?, last_updated = ?
                WHERE user_id = ?
                """, (
                    json.dumps(updated_areas),
                    datetime.now().isoformat(),
                    user_id
                ))
                
                conn.commit()
                self.logger.info(f"Interest areas updated for user {user_id}: {updated_areas}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating interest areas for user {user_id}: {e}")
            return False
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 통계 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            Dict[str, Any]: 사용자 통계
        """
        try:
            profile = self.get_profile(user_id)
            if not profile:
                return {}
            
            # 세션 통계 조회
            user_sessions = self.conversation_store.get_user_sessions(user_id, limit=100)
            
            # 질문 유형 분석
            question_types = {}
            total_turns = 0
            
            for session in user_sessions:
                session_data = self.conversation_store.load_session(session["session_id"])
                if session_data:
                    total_turns += len(session_data.get("turns", []))
                    for turn in session_data.get("turns", []):
                        q_type = turn.get("question_type", "unknown")
                        question_types[q_type] = question_types.get(q_type, 0) + 1
            
            return {
                "user_id": user_id,
                "profile_created_at": profile["created_at"],
                "last_updated": profile["last_updated"],
                "expertise_level": profile["expertise_level"],
                "preferred_detail_level": profile["preferred_detail_level"],
                "interest_areas": profile["interest_areas"],
                "total_sessions": len(user_sessions),
                "total_turns": total_turns,
                "question_type_distribution": question_types,
                "avg_turns_per_session": total_turns / len(user_sessions) if user_sessions else 0,
                "most_active_area": max(question_types.items(), key=lambda x: x[1])[0] if question_types else "unknown"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user statistics for user {user_id}: {e}")
            return {}
    
    def _validate_profile_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """프로필 데이터 검증"""
        validated = {}
        
        # 전문성 수준 검증
        expertise_level = profile_data.get("expertise_level", ExpertiseLevel.BEGINNER.value)
        if expertise_level in [level.value for level in ExpertiseLevel]:
            validated["expertise_level"] = expertise_level
        else:
            validated["expertise_level"] = ExpertiseLevel.BEGINNER.value
        
        # 상세도 수준 검증
        detail_level = profile_data.get("preferred_detail_level", DetailLevel.MEDIUM.value)
        if detail_level in [level.value for level in DetailLevel]:
            validated["preferred_detail_level"] = detail_level
        else:
            validated["preferred_detail_level"] = DetailLevel.MEDIUM.value
        
        # 관심 분야 검증
        interest_areas = profile_data.get("interest_areas", [])
        validated["interest_areas"] = [area for area in interest_areas if area in self.interest_area_keywords]
        
        # 기타 데이터
        validated["preferred_language"] = profile_data.get("preferred_language", "ko")
        validated["device_info"] = profile_data.get("device_info", {})
        validated["location_info"] = profile_data.get("location_info", {})
        validated["preferences"] = profile_data.get("preferences", {})
        
        return validated
    
    def _analyze_interest_areas(self, query: str, current_areas: List[str]) -> List[str]:
        """질문에서 관심 분야 분석"""
        query_lower = query.lower()
        detected_areas = []
        
        for area, keywords in self.interest_area_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_areas.append(area)
        
        # 기존 관심 분야와 병합
        return list(set(current_areas + detected_areas))
    
    def _determine_response_style(self, query: str, profile: Dict[str, Any]) -> str:
        """답변 스타일 결정"""
        query_lower = query.lower()
        
        # 질문에서 스타일 선호도 감지
        for style, patterns in self.response_style_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return style.lower()
        
        # 프로필의 상세도 수준에 따라 결정
        detail_level = profile.get("preferred_detail_level", DetailLevel.MEDIUM.value)
        if detail_level == DetailLevel.SIMPLE.value:
            return "간결함"
        elif detail_level == DetailLevel.DETAILED.value:
            return "상세함"
        else:
            return "보통"
    
    def _get_expertise_context(self, expertise_level: str) -> Dict[str, Any]:
        """전문성 수준에 따른 컨텍스트 제공"""
        contexts = {
            ExpertiseLevel.BEGINNER.value: {
                "explanation_depth": "basic",
                "use_examples": True,
                "avoid_jargon": True,
                "step_by_step": True
            },
            ExpertiseLevel.INTERMEDIATE.value: {
                "explanation_depth": "moderate",
                "use_examples": True,
                "avoid_jargon": False,
                "step_by_step": False
            },
            ExpertiseLevel.ADVANCED.value: {
                "explanation_depth": "detailed",
                "use_examples": False,
                "avoid_jargon": False,
                "step_by_step": False
            },
            ExpertiseLevel.EXPERT.value: {
                "explanation_depth": "comprehensive",
                "use_examples": False,
                "avoid_jargon": False,
                "step_by_step": False
            }
        }
        
        return contexts.get(expertise_level, contexts[ExpertiseLevel.BEGINNER.value])
    
    def _calculate_personalization_score(self, profile: Dict[str, Any], query: str) -> float:
        """개인화 점수 계산"""
        score = 0.0
        
        # 관심 분야 매칭
        query_lower = query.lower()
        interest_areas = profile.get("interest_areas", [])
        for area in interest_areas:
            keywords = self.interest_area_keywords.get(area, [])
            if any(keyword in query_lower for keyword in keywords):
                score += 0.3
        
        # 전문성 수준 매칭
        expertise_level = profile.get("expertise_level", ExpertiseLevel.BEGINNER.value)
        expertise_keywords = self.expertise_keywords.get(ExpertiseLevel(expertise_level), [])
        if any(keyword in query_lower for keyword in expertise_keywords):
            score += 0.2
        
        # 선호도 매칭
        preferences = profile.get("preferences", {})
        if preferences:
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_interest_areas_from_query(self, query: str) -> List[str]:
        """질문에서 관심 분야 감지"""
        return self._analyze_interest_areas(query, [])


# 테스트 함수
def test_user_profile_manager():
    """사용자 프로필 관리자 테스트"""
    manager = UserProfileManager()
    
    user_id = "test_user_001"
    
    print("=== 사용자 프로필 관리자 테스트 ===")
    
    # 1. 프로필 생성
    print("\n1. 프로필 생성 테스트")
    profile_data = {
        "expertise_level": "intermediate",
        "preferred_detail_level": "detailed",
        "interest_areas": ["민법", "형법"],
        "preferences": {"response_style": "professional"}
    }
    
    success = manager.create_profile(user_id, profile_data)
    print(f"프로필 생성 결과: {success}")
    
    # 2. 프로필 조회
    print("\n2. 프로필 조회 테스트")
    profile = manager.get_profile(user_id)
    if profile:
        print(f"사용자 ID: {profile['user_id']}")
        print(f"전문성 수준: {profile['expertise_level']}")
        print(f"상세도 수준: {profile['preferred_detail_level']}")
        print(f"관심 분야: {profile['interest_areas']}")
    
    # 3. 선호도 업데이트
    print("\n3. 선호도 업데이트 테스트")
    new_preferences = {"response_style": "detailed", "language": "ko"}
    success = manager.update_preferences(user_id, new_preferences)
    print(f"선호도 업데이트 결과: {success}")
    
    # 4. 전문성 수준 추적
    print("\n4. 전문성 수준 추적 테스트")
    question_history = [
        {"user_query": "민법 제750조의 손해배상 요건에 대해 자세히 설명해주세요"},
        {"user_query": "대법원 판례에서 과실비율을 어떻게 정하는지 알고 싶습니다"},
        {"user_query": "법리적 해석과 실무적 적용의 차이점은 무엇인가요?"}
    ]
    
    estimated_level = manager.track_expertise_level(user_id, question_history)
    print(f"추정된 전문성 수준: {estimated_level}")
    
    # 5. 개인화된 컨텍스트
    print("\n5. 개인화된 컨텍스트 테스트")
    query = "손해배상 청구 방법을 알려주세요"
    personalized_context = manager.get_personalized_context(user_id, query)
    print(f"개인화된 컨텍스트: {personalized_context}")
    
    # 6. 관심 분야 업데이트
    print("\n6. 관심 분야 업데이트 테스트")
    success = manager.update_interest_areas(user_id, "근로기준법에 따른 퇴직금 계산 방법")
    print(f"관심 분야 업데이트 결과: {success}")
    
    # 7. 사용자 통계
    print("\n7. 사용자 통계 테스트")
    stats = manager.get_user_statistics(user_id)
    print(f"사용자 통계: {stats}")
    
    # 테스트 데이터 정리
    print("\n테스트 완료")


if __name__ == "__main__":
    test_user_profile_manager()

