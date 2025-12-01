#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
리포트 생성 유틸리티

분석 결과를 다양한 형식으로 출력하는 공통 함수들
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


def print_section_header(title: str, width: int = 80, char: str = "=") -> None:
    """섹션 헤더 출력"""
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_subsection_header(title: str, width: int = 80, char: str = "-") -> None:
    """서브섹션 헤더 출력"""
    print(f"\n{title}")
    print(char * width)


def print_metrics(metrics: Dict[str, Any], indent: int = 2) -> None:
    """메트릭 출력"""
    indent_str = " " * indent
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{indent_str}{key}: {value:.4f}")
        elif isinstance(value, (int, str)):
            print(f"{indent_str}{key}: {value}")
        elif isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_metrics(value, indent + 2)
        elif isinstance(value, list):
            print(f"{indent_str}{key}: {len(value)}개")
        else:
            print(f"{indent_str}{key}: {value}")


def print_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> None:
    """테이블 형식으로 데이터 출력"""
    if not data:
        print("  데이터가 없습니다.")
        return
    
    if headers is None:
        headers = list(data[0].keys())
    
    # 컬럼 너비 계산
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], len(value))
    
    # 헤더 출력
    header_row = "  " + " | ".join(str(h).ljust(col_widths[h]) for h in headers)
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))
    
    # 데이터 출력
    for row in data:
        data_row = "  " + " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
        print(data_row)


def print_improvements(improvements: List[Dict[str, Any]]) -> None:
    """개선 사항 출력"""
    if not improvements:
        print("\n✅ 추가 개선 사항이 없습니다!")
        return
    
    for i, improvement in enumerate(improvements, 1):
        priority_emoji = {
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }.get(improvement.get("priority", ""), "ℹ️")
        
        print(f"\n{i}. [{improvement.get('priority', 'UNKNOWN')}] {priority_emoji} {improvement.get('category', 'Unknown')}")
        print(f"   현재: {improvement.get('current', 'N/A')}")
        print(f"   목표: {improvement.get('target', 'N/A')}")
        print(f"   설명: {improvement.get('description', 'N/A')}")
        print(f"   권장사항: {improvement.get('recommendation', 'N/A')}")


def print_summary(summary: Dict[str, Any]) -> None:
    """요약 정보 출력"""
    print_section_header("요약")
    
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            print_metrics(value, indent=2)
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")


def save_text_report(content: str, file_path: Path) -> None:
    """텍스트 리포트 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✅ 리포트가 저장되었습니다: {file_path}")


def generate_markdown_report(
    title: str,
    sections: List[Dict[str, Any]],
    output_path: Optional[Path] = None
) -> str:
    """마크다운 리포트 생성"""
    lines = [f"# {title}\n"]
    lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for section in sections:
        section_title = section.get("title", "")
        section_type = section.get("type", "text")
        section_data = section.get("data", {})
        
        lines.append(f"\n## {section_title}\n")
        
        if section_type == "metrics":
            for key, value in section_data.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.4f}")
                else:
                    lines.append(f"- **{key}**: {value}")
        
        elif section_type == "table":
            if section_data.get("headers"):
                lines.append("| " + " | ".join(section_data["headers"]) + " |")
                lines.append("| " + " | ".join(["---"] * len(section_data["headers"])) + " |")
            
            for row in section_data.get("rows", []):
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
        
        elif section_type == "list":
            for item in section_data.get("items", []):
                lines.append(f"- {item}")
        
        elif section_type == "text":
            lines.append(section_data.get("content", ""))
    
    markdown_content = "\n".join(lines)
    
    if output_path:
        save_text_report(markdown_content, output_path)
    
    return markdown_content

