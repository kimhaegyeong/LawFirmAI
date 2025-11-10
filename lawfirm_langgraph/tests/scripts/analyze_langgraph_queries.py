#!/usr/bin/env python3
"""
LangGraph 질의 분석 스크립트

LangSmith SDK를 사용하여 LangGraph 실행을 분석하고 개선 방법을 제안합니다.

사용법:
    python lawfirm_langgraph/tests/scripts/analyze_langgraph_queries.py [옵션]

옵션:
    --hours: 분석할 시간 범위 (기본값: 24)
    --limit: 최대 조회 개수 (기본값: 100)
    --run-id: 특정 run ID 분석
    --output: 결과 출력 형식 (json, table, summary)
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

try:
    from core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
except ImportError:
    print("Error: langsmith_analyzer module not found. Make sure langsmith is installed.")
    print("Install with: pip install langsmith")
    sys.exit(1)


def print_summary(analyzer: LangGraphQueryAnalyzer, hours: int, limit: int):
    """요약 정보 출력"""
    print(f"\n{'='*60}")
    print(f"LangGraph 질의 분석 요약")
    print(f"{'='*60}")
    print(f"프로젝트: {analyzer.project_name}")
    print(f"분석 기간: 최근 {hours}시간")
    print(f"최대 조회: {limit}개")
    print(f"{'='*60}\n")
    
    runs = analyzer.get_recent_runs(hours=hours, limit=limit)
    
    if not runs:
        print("❌ 분석할 실행 기록이 없습니다.")
        print("   LangSmith 트레이싱이 활성화되어 있는지 확인하세요.")
        return
    
    print(f"✅ 총 {len(runs)}개의 실행 기록을 찾았습니다.\n")
    
    patterns = analyzer.analyze_query_patterns(runs)
    
    print("📊 성능 통계:")
    print(f"  - 평균 토큰 사용량: {patterns['token_usage']['average_per_run']:.0f} tokens/run")
    print(f"  - 최대 토큰 사용량: {patterns['token_usage']['max']} tokens")
    print(f"  - 총 토큰 사용량: {patterns['token_usage']['total']} tokens")
    
    if patterns.get("slow_queries"):
        print(f"\n⏱️ 느린 질의 ({len(patterns['slow_queries'])}개):")
        for query_info in patterns["slow_queries"][:5]:
            print(f"  - {query_info['query'][:60]}... ({query_info['duration']:.2f}초)")
    
    if patterns.get("error_queries"):
        print(f"\n❌ 오류 발생 질의 ({len(patterns['error_queries'])}개):")
        for query_info in patterns["error_queries"][:5]:
            print(f"  - {query_info['query'][:60]}...")
            print(f"    오류: {query_info['error'][:100]}")
    
    if patterns.get("common_nodes"):
        print(f"\n🔄 자주 실행되는 노드:")
        sorted_nodes = sorted(
            patterns["common_nodes"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for node_name, count in sorted_nodes:
            avg_duration = patterns["average_durations"].get(node_name, 0)
            print(f"  - {node_name}: {count}회 (평균 {avg_duration:.2f}초)")


def analyze_single_run(analyzer: LangGraphQueryAnalyzer, run_id: str, show_tree: bool = False):
    """단일 run 분석"""
    print(f"\n{'='*60}")
    print(f"Run 분석: {run_id}")
    print(f"{'='*60}\n")
    
    try:
        if not analyzer._validate_run_id(run_id):
            print(f"❌ 잘못된 Run ID 형식: {run_id}")
            return
        
        run = analyzer.client.read_run(run_id)
        
        tree = None
        if show_tree:
            tree = analyzer.get_run_tree(run_id, show_progress=True)
        
        analysis = analyzer.analyze_run_performance(run, tree=tree)
        
        print(f"질의: {analysis['query']}")
        print(f"상태: {analysis['status']}")
        if analysis['start_time']:
            print(f"시작 시간: {analysis['start_time']}")
        if analysis['end_time']:
            print(f"종료 시간: {analysis['end_time']}")
        print(f"실행 시간: {analysis['duration']:.2f}초" if analysis['duration'] else "N/A")
        print(f"토큰 사용량: {analysis['total_tokens']}")
        print(f"예상 비용: ${analysis['total_cost']:.4f}")
        
        state_info = analysis.get('state_info', {})
        if state_info:
            print(f"\nState 정보:")
            print(f"  - Inputs 존재: {state_info.get('has_inputs', False)}")
            print(f"  - Outputs 존재: {state_info.get('has_outputs', False)}")
            if state_info.get('state_snapshot'):
                print(f"  - State 스냅샷: {state_info['state_snapshot']}")
            if state_info.get('input_keys'):
                print(f"  - Input 키: {', '.join(state_info['input_keys'][:5])}")
            if state_info.get('output_keys'):
                print(f"  - Output 키: {', '.join(state_info['output_keys'][:5])}")
        
        if show_tree:
            print(f"\n{'='*60}")
            print("RunTree 구조:")
            print(f"{'='*60}")
            tree_visualization = analyzer.visualize_run_tree(run_id)
            print(tree_visualization)
            
            print(f"\n{'='*60}")
            print("State 흐름 분석:")
            print(f"{'='*60}")
            state_flow = analyzer.analyze_state_flow(run_id)
            if state_flow:
                summary = state_flow.get('state_changes_summary', {})
                print(f"총 노드: {summary.get('total_nodes', 0)}")
                print(f"State가 있는 노드: {summary.get('nodes_with_state', 0)}")
                print(f"State 변경이 있는 노드: {summary.get('nodes_with_changes', 0)}")
                print(f"총 State 변경 횟수: {summary.get('total_changes', 0)}")
                
                groups_usage = state_flow.get('state_groups_usage', {})
                if groups_usage:
                    print(f"\nState 그룹 사용 현황:")
                    for group, count in sorted(groups_usage.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {group}: {count}회")
                
                transitions = state_flow.get('state_transitions', [])
                if transitions:
                    print(f"\nState 전환 상세 ({len(transitions)}개):")
                    for i, transition in enumerate(transitions[:10], 1):
                        changes = transition.get('changes', {})
                        print(f"  [{i}] {transition.get('node', 'unknown')}:")
                        print(f"      추가된 키: {len(changes.get('keys_added', []))}")
                        print(f"      제거된 키: {len(changes.get('keys_removed', []))}")
                        print(f"      수정된 키: {len(changes.get('keys_modified', []))}")
                        groups_modified = changes.get('groups_modified', [])
                        if groups_modified:
                            print(f"      수정된 그룹: {[g['group'] for g in groups_modified]}")
                
                # State 전달 정보 표시
                nodes_with_state = state_flow.get('nodes_with_state', [])
                inherited_nodes = [n for n in nodes_with_state if n.get('state_inherited', False)]
                if inherited_nodes:
                    print(f"\nState 전달 확인:")
                    print(f"  - State를 상속받은 노드: {len(inherited_nodes)}개")
                    for node in inherited_nodes[:5]:
                        inherited_keys = node.get('inherited_keys', [])
                        print(f"    - {node.get('node_name', 'unknown')}: {len(inherited_keys)}개 키 상속")
                        if inherited_keys:
                            print(f"      상속된 키: {', '.join(inherited_keys[:5])}")
                else:
                    print(f"\nState 전달 확인:")
                    print(f"  - State를 상속받은 노드: 0개 (부모-자식 간 state 전달이 확인되지 않음)")
            
            cache_stats = analyzer.get_cache_stats()
            print(f"\n캐시 통계:")
            print(f"  - 히트율: {cache_stats['hit_rate']:.1f}%")
            print(f"  - 히트: {cache_stats['hits']}, 미스: {cache_stats['misses']}")
            print(f"  - Run 캐시 크기: {cache_stats['run_cache_size']}")
            print(f"  - Tree 캐시 크기: {cache_stats['tree_cache_size']}")
            
            stats = analyzer.get_run_statistics(run_id, tree=tree)
            if stats:
                print(f"\n{'='*60}")
                print("통계 정보:")
                print(f"{'='*60}")
                print(f"총 Runs: {stats['total_runs']}")
                print(f"최대 깊이: {stats['max_depth']}")
                print(f"총 실행 시간: {stats['total_duration']:.2f}초")
                print(f"평균 실행 시간: {stats['average_duration']:.2f}초")
                print(f"\nRun Type별 분포:")
                for run_type, count in stats['by_type'].items():
                    print(f"  - {run_type}: {count}")
                print(f"\nStatus별 분포:")
                for status, count in stats['by_status'].items():
                    print(f"  - {status}: {count}")
                if stats.get('state_updates'):
                    state_updates = stats['state_updates']
                    print(f"\nState 업데이트 통계:")
                    print(f"  - State가 있는 노드: {state_updates.get('nodes_with_state', 0)}")
                    print(f"  - State 전환 횟수: {state_updates.get('state_transitions', 0)}")
                if stats.get('node_durations'):
                    print(f"\n노드별 실행 시간 통계:")
                    for node_name, node_stats in sorted(
                        stats['node_durations'].items(),
                        key=lambda x: x[1]['total'],
                        reverse=True
                    )[:10]:
                        print(f"  - {node_name}:")
                        print(f"    실행 횟수: {node_stats['count']}")
                        print(f"    총 시간: {node_stats['total']:.2f}초")
                        print(f"    평균 시간: {node_stats['average']:.2f}초")
                        print(f"    최소/최대: {node_stats['min']:.2f}초 / {node_stats['max']:.2f}초")
        
        if analysis.get("nodes"):
            print(f"\n노드별 실행 정보 ({len(analysis['nodes'])}개):")
            for i, node in enumerate(analysis["nodes"], 1):
                duration_str = f"{node['duration']:.2f}초" if node.get('duration') else "N/A"
                tokens_str = f"{node['tokens']} tokens" if node.get('tokens') else "N/A"
                status_str = f" [{node['status']}]" if node.get('status') else ""
                print(f"  [{i}] {node['name']} ({node['run_type']}){status_str}: {duration_str} ({tokens_str})")
        else:
            print(f"\n노드 정보: 자식 runs를 찾을 수 없습니다.")
        
        if analysis.get("bottlenecks"):
            print(f"\n🐌 병목 지점:")
            for bottleneck in analysis["bottlenecks"]:
                print(f"  - {bottleneck['node']}: {bottleneck['duration']:.2f}초 ({bottleneck['tokens']} tokens)")
        
        suggestions = analyzer.get_improvement_suggestions(analysis)
        if suggestions:
            print(f"\n💡 개선 제안:")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        
        if analysis.get("error"):
            print(f"\n❌ 오류:")
            print(f"  {analysis['error']}")
    
    except Exception as e:
        print(f"❌ Run 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def print_table(analyzer: LangGraphQueryAnalyzer, hours: int, limit: int):
    """테이블 형식으로 출력"""
    runs = analyzer.get_recent_runs(hours=hours, limit=limit)
    
    if not runs:
        print("❌ 분석할 실행 기록이 없습니다.")
        return
    
    print(f"\n{'='*100}")
    print(f"{'질의':<50} {'상태':<10} {'시간(초)':<12} {'토큰':<12}")
    print(f"{'='*100}")
    
    for run in runs[:20]:
        query = analyzer._extract_query(run)
        query_short = query[:48] + "..." if len(query) > 50 else query
        status = run.status or "unknown"
        duration = "N/A"
        if run.start_time and run.end_time:
            duration = f"{(run.end_time - run.start_time).total_seconds():.2f}"
        
        print(f"{query_short:<50} {status:<10} {duration:<12}")


def export_json(analyzer: LangGraphQueryAnalyzer, hours: int, limit: int, output_file: str):
    """JSON 형식으로 내보내기"""
    runs = analyzer.get_recent_runs(hours=hours, limit=limit)
    
    results = {
        "project": analyzer.project_name,
        "analysis_period_hours": hours,
        "total_runs": len(runs),
        "runs": []
    }
    
    for run in runs:
        analysis = analyzer.analyze_run_performance(run)
        patterns = analyzer.analyze_query_patterns([run])
        suggestions = analyzer.get_improvement_suggestions(analysis, patterns)
        
        results["runs"].append({
            "run_id": str(run.id),
            "analysis": analysis,
            "suggestions": suggestions
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 결과를 {output_file}에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph 질의 분석 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="분석할 시간 범위 (기본값: 24)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="최대 조회 개수 (기본값: 100)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="특정 run ID 분석"
    )
    parser.add_argument(
        "--output",
        choices=["json", "table", "summary"],
        default="summary",
        help="결과 출력 형식 (기본값: summary)"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="JSON 파일로 내보내기 (파일 경로)"
    )
    parser.add_argument(
        "--show-tree",
        action="store_true",
        help="RunTree 구조와 통계 정보 표시"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = LangGraphQueryAnalyzer()
        
        if args.run_id:
            analyze_single_run(analyzer, args.run_id, show_tree=args.show_tree)
        elif args.export:
            export_json(analyzer, args.hours, args.limit, args.export)
        elif args.output == "table":
            print_table(analyzer, args.hours, args.limit)
        else:
            print_summary(analyzer, args.hours, args.limit)
    
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        print("\n환경 변수 설정:")
        print("  export LANGSMITH_API_KEY=your-api-key")
        print("  export LANGSMITH_PROJECT=LawFirmAI")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

