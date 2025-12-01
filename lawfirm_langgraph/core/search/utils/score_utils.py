# -*- coding: utf-8 -*-
"""검색 점수 정규화 유틸리티"""

from typing import List


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    점수를 0.0~1.0 범위로 강제 정규화
    
    Args:
        score: 정규화할 점수
        min_val: 최소값 (기본값: 0.0)
        max_val: 최대값 (기본값: 1.0)
    
    Returns:
        0.0~1.0 범위의 정규화된 점수
    """
    if score < min_val:
        return float(min_val)
    elif score > max_val:
        # 초과 점수는 로그 스케일로 완화하여 1.0으로 수렴
        excess = score - max_val
        normalized = max_val + (excess / (1.0 + excess * 10))
        return float(max(0.0, min(1.0, normalized)))
    return float(score)


def clamp_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    점수를 지정된 범위로 강제 제한 (단순 클램핑)
    
    Args:
        score: 제한할 점수
        min_val: 최소값 (기본값: 0.0)
        max_val: 최대값 (기본값: 1.0)
    
    Returns:
        제한된 점수
    """
    return float(max(min_val, min(max_val, score)))


def normalize_scores_batch(scores: List[float], method: str = "min_max") -> List[float]:
    """
    여러 점수를 일괄 정규화
    
    Args:
        scores: 정규화할 점수 리스트
        method: 정규화 방법 ("min_max", "clamp")
    
    Returns:
        정규화된 점수 리스트
    """
    if not scores:
        return scores
    
    if method == "clamp":
        return [clamp_score(score) for score in scores]
    else:  # min_max
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # 모든 점수가 같으면 0.5로 설정
        
        # min-max 정규화
        range_val = max_score - min_score
        return [
            clamp_score((score - min_score) / range_val)
            for score in scores
        ]

