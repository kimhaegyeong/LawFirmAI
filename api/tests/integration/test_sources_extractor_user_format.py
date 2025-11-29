# -*- coding: utf-8 -*-
"""
실제 사용자 데이터 형식으로 통합 테스트
사용자가 제공한 JSON 형식과 동일한 데이터로 테스트
"""

import sys
import os
import json
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'api'))

from api.services.sources_extractor import SourcesExtractor

# SourcesExtractor 초기화 (간단한 Mock 사용)
def get_sources_extractor():
    """SourcesExtractor 인스턴스 생성"""
    try:
        class MockWorkflowService:
            pass
        
        class MockSessionService:
            pass
        
        workflow_service = MockWorkflowService()
        session_service = MockSessionService()
        
        extractor = SourcesExtractor(workflow_service, session_service)
        return extractor
    except Exception as e:
        print(f"⚠️ SourcesExtractor 초기화 실패: {e}")
        return None


def test_user_data_format():
    """사용자가 제공한 JSON 형식과 동일한 데이터로 테스트"""
    print("\n" + "=" * 60)
    print("실제 사용자 데이터 형식 통합 테스트")
    print("=" * 60)
    
    sources_extractor = get_sources_extractor()
    if not sources_extractor:
        print("❌ SourcesExtractor를 초기화할 수 없습니다.")
        return 1
    
    # 사용자가 제공한 JSON과 동일한 형식의 sources_detail
    sources_detail = [
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "나) 피고들은 원고에게 민법 제750조 불법행위에 기한 손해배상책임 또는 민법 제758조 공작물 소유자의 책임에 근거하여 원고가 입은 손해를 배상할 의무가 있다.",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "따라서 특별한 사정이 없는 한, 피고는 원고들에게 민법 제750조 또는 부정경쟁방지법 제5조에 따라 그로 인한 손해를 배상할 책임이 있다.",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "[1] [1] 민법 제750조, 제806조 제843조 / [2] 민법 제750조, 제806조 , 제843조 / [3] 제396조 , 제763조 , 제806조 제843조",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "[1] 민법 제840조 , /[2] 민법 제750조 , 제806조 , 제840조 , 제843조",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "민법 제750조 , 제806조 , 제843조",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "statute_name": "법령",
            "content": "민법 제750조",
            "metadata": {}
        },
        {
            "type": "case_paragraph",
            "content": "1) 원고가 피고 주택도시보증공사에 대하여 갖는 하자보수보증금채권은 피고 B, 피고 D에 대한 하자보수비 상당의 손해배상채권과 인정 근거, 권리관계의 당사자, 책임내용 등이 서로 다른 별개의 권리이므로, 원고에 대한 피고 B, 피고 D의 손해배상채무와 피고 주택도시보증공사의 하자보수보증금채무가 부진정연대채무 관계에 있지는 아니하나, 위 두 채무가 겹치는 범위 내에서는 결과적으로 동일한 하자의 보수를 위하여 존재하는 것이므로, 향후 원고가 그중 어느 한 권리를 행사하여 하자보수에 갈음한 보수비용 상당이 지급되면 다른 권리도 소멸하는 관계에 있다(대법원 2015. 3. 20. 선고 2012다107662 판결 등 참조).",
            "metadata": {}
        }
    ]
    
    print(f"\n입력 sources_detail: {len(sources_detail)}개")
    print(f"  - statute_article: {len([s for s in sources_detail if s.get('type') == 'statute_article'])}개")
    print(f"  - case_paragraph: {len([s for s in sources_detail if s.get('type') == 'case_paragraph'])}개")
    
    # sources_by_type 생성
    sources_by_type = sources_extractor._get_sources_by_type(sources_detail)
    
    statute_articles = sources_by_type.get("statute_article", [])
    case_paragraphs = sources_by_type.get("case_paragraph", [])
    
    print(f"\n출력 sources_by_type:")
    print(f"  - statute_article: {len(statute_articles)}개")
    print(f"  - case_paragraph: {len(case_paragraphs)}개")
    
    # 결과 확인
    print("\n=== statute_article 검증 ===")
    all_statutes_valid = True
    for i, statute in enumerate(statute_articles, 1):
        name = statute.get("name", "")
        statute_name = statute.get("statute_name", "")
        content = statute.get("content", "")[:50]
        
        if name and name != "법령" and statute_name and statute_name != "법령":
            print(f"✅ statute_article {i}: name='{name}', statute_name='{statute_name}'")
            print(f"   content: {content}...")
        else:
            print(f"❌ statute_article {i}: name='{name}', statute_name='{statute_name}' (예상: '민법' 또는 다른 법령명)")
            print(f"   content: {content}...")
            all_statutes_valid = False
    
    print("\n=== case_paragraph 검증 ===")
    all_cases_valid = True
    for i, case_para in enumerate(case_paragraphs, 1):
        name = case_para.get("name", "")
        case_number = case_para.get("case_number", "")
        content = case_para.get("content", "")[:50]
        
        if name:
            print(f"✅ case_paragraph {i}: name='{name}'")
            if case_number:
                print(f"   case_number='{case_number}'")
            print(f"   content: {content}...")
        else:
            print(f"❌ case_paragraph {i}: name 필드 없음")
            print(f"   content: {content}...")
            all_cases_valid = False
    
    # JSON 출력 (사용자가 제공한 형식과 비교)
    print("\n=== 생성된 sources_by_type JSON (일부) ===")
    output_json = {
        "statute_article": [
            {
                "type": s.get("type"),
                "name": s.get("name"),
                "statute_name": s.get("statute_name"),
                "content": s.get("content", "")[:100] + "..." if len(s.get("content", "")) > 100 else s.get("content", "")
            }
            for s in statute_articles[:3]
        ],
        "case_paragraph": [
            {
                "type": c.get("type"),
                "name": c.get("name"),
                "case_number": c.get("case_number"),
                "content": c.get("content", "")[:100] + "..." if len(c.get("content", "")) > 100 else c.get("content", "")
            }
            for c in case_paragraphs[:1]
        ]
    }
    print(json.dumps(output_json, ensure_ascii=False, indent=2))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    if all_statutes_valid and all_cases_valid:
        print("✅ 모든 sources에 name 필드가 올바르게 설정되었습니다!")
        print(f"\n   - statute_article: {len(statute_articles)}개 모두 법령명 표시")
        print(f"   - case_paragraph: {len(case_paragraphs)}개 모두 name 필드 있음")
        return 0
    else:
        print("❌ 일부 sources에 name 필드가 올바르게 설정되지 않았습니다!")
        if not all_statutes_valid:
            print(f"   - statute_article: 일부 항목의 name이 '법령'으로 표시됨")
        if not all_cases_valid:
            print(f"   - case_paragraph: 일부 항목에 name 필드 없음")
        return 1


if __name__ == "__main__":
    sys.exit(test_user_data_format())

