# -*- coding: utf-8 -*-
"""
가중치 조합 테스트 스크립트
여러 가중치 조합을 테스트하여 최적의 설정을 찾습니다.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import subprocess

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_dir))
sys.path.insert(0, str(project_root))

# 테스트할 가중치 조합들
WEIGHT_CONFIGS = [
    {
        "name": "기본 설정",
        "hybrid_law": {"semantic": 0.3, "keyword": 0.7},
        "hybrid_case": {"semantic": 0.7, "keyword": 0.3},
        "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
        "doc_type_boost": {"statute": 1.2, "case": 1.15},
        "quality_weight": 0.2,
        "keyword_adjustment": 1.8
    },
    {
        "name": "키워드 강조",
        "hybrid_law": {"semantic": 0.2, "keyword": 0.8},
        "hybrid_case": {"semantic": 0.6, "keyword": 0.4},
        "hybrid_general": {"semantic": 0.4, "keyword": 0.6},
        "doc_type_boost": {"statute": 1.3, "case": 1.1},
        "quality_weight": 0.15,
        "keyword_adjustment": 2.0
    },
    {
        "name": "의미 검색 강조",
        "hybrid_law": {"semantic": 0.4, "keyword": 0.6},
        "hybrid_case": {"semantic": 0.8, "keyword": 0.2},
        "hybrid_general": {"semantic": 0.6, "keyword": 0.4},
        "doc_type_boost": {"statute": 1.1, "case": 1.2},
        "quality_weight": 0.25,
        "keyword_adjustment": 1.6
    },
    {
        "name": "균형 설정",
        "hybrid_law": {"semantic": 0.35, "keyword": 0.65},
        "hybrid_case": {"semantic": 0.65, "keyword": 0.35},
        "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
        "doc_type_boost": {"statute": 1.15, "case": 1.1},
        "quality_weight": 0.2,
        "keyword_adjustment": 1.7
    },
    {
        "name": "품질 강조",
        "hybrid_law": {"semantic": 0.3, "keyword": 0.7},
        "hybrid_case": {"semantic": 0.7, "keyword": 0.3},
        "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
        "doc_type_boost": {"statute": 1.2, "case": 1.15},
        "quality_weight": 0.3,
        "keyword_adjustment": 1.8
    }
]

# 테스트 쿼리들
TEST_QUERIES = [
    "민법 제750조 손해배상에 대해 설명해주세요"
]

def update_weight_config(config):
    """가중치 설정 파일 업데이트"""
    config_file = lawfirm_langgraph_dir / "core" / "search" / "processors" / "search_result_processor.py"
    
    # 파일 읽기
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 기본 가중치 설정 찾아서 교체
    old_config = """        self.weight_config = weight_config or {
            "hybrid_law": {"semantic": 0.3, "keyword": 0.7},
            "hybrid_case": {"semantic": 0.7, "keyword": 0.3},
            "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
            "doc_type_boost": {"statute": 1.2, "case": 1.15},
            "quality_weight": 0.2,
            "keyword_adjustment": 1.8
        }"""
    
    new_config = f"""        self.weight_config = weight_config or {{
            "hybrid_law": {{"semantic": {config["hybrid_law"]["semantic"]}, "keyword": {config["hybrid_law"]["keyword"]}}},
            "hybrid_case": {{"semantic": {config["hybrid_case"]["semantic"]}, "keyword": {config["hybrid_case"]["keyword"]}}},
            "hybrid_general": {{"semantic": {config["hybrid_general"]["semantic"]}, "keyword": {config["hybrid_general"]["keyword"]}}},
            "doc_type_boost": {{"statute": {config["doc_type_boost"]["statute"]}, "case": {config["doc_type_boost"]["case"]}}},
            "quality_weight": {config["quality_weight"]},
            "keyword_adjustment": {config["keyword_adjustment"]}
        }}"""
    
    content = content.replace(old_config, new_config)
    
    # 파일 쓰기
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)

def run_test(query):
    """테스트 실행"""
    test_script = project_root / "lawfirm_langgraph" / "tests" / "scripts" / "run_query_test.py"
    result = subprocess.run(
        [sys.executable, str(test_script), query],
        capture_output=True,
        text=True,
        encoding='utf-8',
        cwd=str(project_root)
    )
    return result.stdout, result.stderr

def extract_metrics(output):
    """출력에서 메트릭 추출"""
    metrics = {}
    
    # Avg Relevance 추출
    import re
    avg_match = re.search(r'Avg Relevance: ([\d.]+)', output)
    if avg_match:
        metrics['avg_relevance'] = float(avg_match.group(1))
    
    min_match = re.search(r'Min: ([\d.]+)', output)
    if min_match:
        metrics['min_relevance'] = float(min_match.group(1))
    
    max_match = re.search(r'Max: ([\d.]+)', output)
    if max_match:
        metrics['max_relevance'] = float(max_match.group(1))
    
    keyword_match = re.search(r'Keyword Coverage: ([\d.]+)', output)
    if keyword_match:
        metrics['keyword_coverage'] = float(keyword_match.group(1))
    
    return metrics

def main():
    """메인 테스트 실행"""
    print("🚀 가중치 최적화 테스트 시작")
    print(f"테스트 설정 수: {len(WEIGHT_CONFIGS)}")
    print(f"테스트 쿼리 수: {len(TEST_QUERIES)}")
    print(f"총 테스트 수: {len(WEIGHT_CONFIGS) * len(TEST_QUERIES)}\n")
    
    all_results = []
    original_config = None
    
    try:
        # 원본 설정 백업
        config_file = lawfirm_langgraph_dir / "core" / "search" / "processors" / "search_result_processor.py"
        with open(config_file, 'r', encoding='utf-8') as f:
            original_config = f.read()
        
        for i, config in enumerate(WEIGHT_CONFIGS, 1):
            print(f"\n{'='*80}")
            print(f"테스트 {i}/{len(WEIGHT_CONFIGS)}: {config['name']}")
            print(f"{'='*80}\n")
            
            # 가중치 설정 업데이트
            update_weight_config(config)
            
            for query in TEST_QUERIES:
                print(f"📝 쿼리: {query}")
                stdout, stderr = run_test(query)
                
                metrics = extract_metrics(stdout)
                
                result = {
                    "config_name": config['name'],
                    "config": config,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "stdout": stdout[-2000:] if len(stdout) > 2000 else stdout,  # 마지막 2000자만
                    "stderr": stderr[-1000:] if len(stderr) > 1000 else stderr
                }
                
                all_results.append(result)
                
                if metrics:
                    print(f"  ✅ Avg Relevance: {metrics.get('avg_relevance', 'N/A')}")
                    print(f"  ✅ Keyword Coverage: {metrics.get('keyword_coverage', 'N/A')}")
                else:
                    print(f"  ⚠️  메트릭 추출 실패")
        
        # 원본 설정 복원
        if original_config:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(original_config)
            print("\n✅ 원본 설정 복원 완료")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        # 원본 설정 복원
        if original_config:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(original_config)
    
    # 결과 저장
    output_file = project_root / "logs" / "test" / f"weight_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "configs": WEIGHT_CONFIGS,
            "queries": TEST_QUERIES,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 테스트 완료. 결과 저장: {output_file}")
    
    # 결과 요약
    print(f"\n📊 결과 요약:")
    print(f"{'설정':<20} {'Avg Relevance':<15} {'Keyword Coverage':<15}")
    print("-" * 50)
    for result in all_results:
        metrics = result.get('metrics', {})
        print(f"{result['config_name']:<20} {metrics.get('avg_relevance', 'N/A'):<15} {metrics.get('keyword_coverage', 'N/A'):<15}")

if __name__ == "__main__":
    main()

