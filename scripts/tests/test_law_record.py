import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.utilities.import_laws_to_db import AssemblyLawImporter

# 테스트용 데이터
test_data = {
    'law_id': 'test_id',
    'law_name': '테스트 법률',
    'law_type': '법률',
    'category': '일반',
    'row_number': '1',
    'promulgation_number': '12345',
    'promulgation_date': '2025-01-01',
    'enforcement_date': '2025-01-01',
    'amendment_type': '신설',
    'ministry': '법무부',
    'parent_law': '',
    'related_laws': [],
    'full_text': '테스트 내용',
    'summary': '테스트 요약',
    'html_clean_text': '',
    'content_html': '',
    'raw_content': '',
    'detail_url': '',
    'cont_id': '',
    'cont_sid': '',
    'collected_at': '2025-01-01',
    'processed_at': '2025-01-01',
    'processing_version': '1.0',
    'data_quality': {},
    'ml_enhanced': True,
    'articles': []
}

importer = AssemblyLawImporter('data/lawfirm.db')
law_record = importer._prepare_law_record(test_data)
print(f'반환된 값의 개수: {len(law_record)}')
print('값들:')
for i, value in enumerate(law_record):
    print(f'{i+1}: {value}')

