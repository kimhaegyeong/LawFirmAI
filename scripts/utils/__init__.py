"""
공통 유틸리티 모듈

analysis 스크립트에서 사용하는 공통 유틸리티 함수들
"""

from scripts.utils.path_utils import setup_project_path, get_project_root
from scripts.utils.file_utils import load_json_file, save_json_file, load_json_files
from scripts.utils.text_utils import (
    extract_keywords,
    normalize_text,
    remove_special_chars,
    extract_legal_terms,
    calculate_text_similarity
)
from scripts.utils.log_analyzer import (
    analyze_sources_conversion_logs,
    analyze_legal_references_logs,
    analyze_answer_length_logs,
    analyze_context_usage_logs,
    identify_improvements
)
from scripts.utils.report_utils import (
    print_section_header,
    print_subsection_header,
    print_metrics,
    print_table,
    print_improvements,
    print_summary,
    save_text_report,
    generate_markdown_report
)

__all__ = [
    # path_utils
    'setup_project_path',
    'get_project_root',
    # file_utils
    'load_json_file',
    'save_json_file',
    'load_json_files',
    # text_utils
    'extract_keywords',
    'normalize_text',
    'remove_special_chars',
    'extract_legal_terms',
    'calculate_text_similarity',
    # log_analyzer
    'analyze_sources_conversion_logs',
    'analyze_legal_references_logs',
    'analyze_answer_length_logs',
    'analyze_context_usage_logs',
    'identify_improvements',
    # report_utils
    'print_section_header',
    'print_subsection_header',
    'print_metrics',
    'print_table',
    'print_improvements',
    'print_summary',
    'save_text_report',
    'generate_markdown_report',
]

