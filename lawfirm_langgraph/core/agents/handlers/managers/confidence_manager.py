# -*- coding: utf-8 -*-
"""신뢰도 관리 클래스"""

import re
import logging
from typing import Optional, Tuple
from ..config.formatter_config import ConfidenceConfig


class ConfidenceManager:
    """신뢰도 계산 및 텍스트 교체 담당"""
    
    CONFIDENCE_PATTERNS = [
        (r'\*\*신뢰도:\s*[\d.]+%\*\*', '**신뢰도: {value}**'),
        (r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*', '{emoji} **신뢰도: {value}**'),
        (r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*', '{emoji} **신뢰도: {value}**'),
        (r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*', '{emoji} **신뢰도: {value}**'),
        (r'신뢰도:\s*[\d.]+%', '신뢰도: {value}'),
        (r'답변품질:\s*[\d.]+%', '답변 품질: {value}'),
        (r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(low\)', '**신뢰도: {value}** ({level})'),
        (r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(medium\)', '**신뢰도: {value}** ({level})'),
        (r'\*\*신뢰도:\s*[\d.]+%\*\*\s*\(high\)', '**신뢰도: {value}** ({level})'),
    ]
    
    def __init__(self, config: Optional[ConfidenceConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or ConfidenceConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def get_emoji(self, confidence: float) -> str:
        """신뢰도에 따른 이모지 반환"""
        if confidence >= 0.8:
            return "🟢"
        elif confidence >= 0.6:
            return "🟡"
        return "🟠"
    
    def get_level(self, confidence: float) -> str:
        """신뢰도에 따른 레벨 반환"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        return "low"
    
    def replace_all(self, text: str, confidence: float) -> str:
        """모든 신뢰도 패턴을 한 번에 교체"""
        if not text or not isinstance(text, str):
            return text
        
        confidence_str = f"{confidence:.1%}"
        emoji = self.get_emoji(confidence)
        level = self.get_level(confidence)
        
        for pattern, replacement in self.CONFIDENCE_PATTERNS:
            text = re.sub(
                pattern,
                replacement.format(value=confidence_str, emoji=emoji, level=level),
                text,
                flags=re.IGNORECASE
            )
        
        return text
    
    def replace_confidence_section(self, text: str, confidence: float) -> str:
        """신뢰도 정보 섹션 전체 교체"""
        confidence_str = f"{confidence:.1%}"
        emoji = self.get_emoji(confidence)
        level = self.get_level(confidence)
        
        new_section = (
            f'### 💡 신뢰도정보\n'
            f'{emoji} **신뢰도: {confidence_str}** ({level})\n\n'
            f'**상세점수:**\n'
            f'- 답변 품질: {confidence_str}\n\n'
            f'**설명:** 신뢰도: {confidence_str}'
        )
        
        lines = text.split('\n')
        new_lines = []
        in_confidence_section = False
        
        for line in lines:
            if re.match(r'^###\s*💡\s*신뢰도정보', line, re.IGNORECASE):
                in_confidence_section = True
                new_lines.append(new_section)
                continue
            
            if in_confidence_section:
                if line.strip() == '---' or line.strip().startswith('💼') or re.match(r'^###\s+', line):
                    in_confidence_section = False
                    new_lines.append(line)
                continue
            
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def replace_in_text(self, text: str, confidence: float, max_iterations: int = 10) -> str:
        """텍스트 내 모든 신뢰도 값 교체 (반복 적용)"""
        if not text or not isinstance(text, str) or confidence <= 0:
            return text
        
        confidence_str = f"{confidence:.1%}"
        emoji = self.get_emoji(confidence)
        level = self.get_level(confidence)
        
        for _ in range(max_iterations):
            text = self.replace_all(text, confidence)
            text = re.sub(
                r'신뢰도:\s*[\d.]+%',
                f'신뢰도: {confidence_str}',
                text,
                flags=re.IGNORECASE
            )
            text = re.sub(
                r'\*\*신뢰도:\s*[\d.]+%\*\*',
                f'**신뢰도: {confidence_str}**',
                text,
                flags=re.IGNORECASE
            )
            text = re.sub(
                r'🟡\s*\*\*신뢰도:\s*[\d.]+%\*\*',
                f'{emoji} **신뢰도: {confidence_str}**',
                text,
                flags=re.IGNORECASE
            )
            text = re.sub(
                r'🟠\s*\*\*신뢰도:\s*[\d.]+%\*\*',
                f'{emoji} **신뢰도: {confidence_str}**',
                text,
                flags=re.IGNORECASE
            )
            text = re.sub(
                r'🟢\s*\*\*신뢰도:\s*[\d.]+%\*\*',
                f'{emoji} **신뢰도: {confidence_str}**',
                text,
                flags=re.IGNORECASE
            )
        
        text = self.replace_confidence_section(text, confidence)
        
        return text

