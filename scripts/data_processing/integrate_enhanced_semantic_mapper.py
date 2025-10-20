#!/usr/bin/env python3
"""
기존 keyword_mapper.py의 SemanticKeywordMapper를 확장된 버전으로 교체하는 스크립트
"""

import os
import shutil
from datetime import datetime

def backup_original_file():
    """원본 파일 백업"""
    original_file = "source/services/langgraph/keyword_mapper.py"
    backup_file = f"source/services/langgraph/keyword_mapper_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"원본 파일 백업 완료: {backup_file}")
        return True
    return False

def integrate_enhanced_semantic_mapper():
    """향상된 SemanticKeywordMapper 통합"""
    try:
        # 백업 생성
        if not backup_original_file():
            print("원본 파일을 찾을 수 없습니다.")
            return False
        
        # 기존 파일 읽기
        with open("source/services/langgraph/keyword_mapper.py", 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 향상된 SemanticKeywordMapper 클래스 읽기
        with open("source/services/langgraph/enhanced_semantic_relations.py", 'r', encoding='utf-8') as f:
            enhanced_content = f.read()
        
        # EnhancedSemanticKeywordMapper 클래스 추출
        start_marker = "class EnhancedSemanticKeywordMapper:"
        end_marker = "# 사용 예시"
        
        start_idx = enhanced_content.find(start_marker)
        end_idx = enhanced_content.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            print("향상된 클래스를 찾을 수 없습니다.")
            return False
        
        enhanced_class = enhanced_content[start_idx:end_idx].strip()
        
        # 기존 SemanticKeywordMapper 클래스 교체
        old_start = original_content.find("class SemanticKeywordMapper:")
        if old_start == -1:
            print("기존 SemanticKeywordMapper 클래스를 찾을 수 없습니다.")
            return False
        
        # 기존 클래스의 끝 찾기
        old_end = original_content.find("class EnhancedKeywordMapper:", old_start)
        if old_end == -1:
            print("기존 클래스의 끝을 찾을 수 없습니다.")
            return False
        
        # 새로운 내용 생성
        new_content = (
            original_content[:old_start] +
            enhanced_class + "\n\n" +
            original_content[old_end:]
        )
        
        # 파일 저장
        with open("source/services/langgraph/keyword_mapper.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("향상된 SemanticKeywordMapper 통합 완료")
        return True
        
    except Exception as e:
        print(f"통합 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("SemanticKeywordMapper 확장 통합 시작")
    
    if integrate_enhanced_semantic_mapper():
        print("통합 완료!")
    else:
        print("통합 실패!")
