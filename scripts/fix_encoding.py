#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 파일 인코딩 확인 및 수정 스크립트
"""

import os
import glob

def check_and_fix_encoding(file_path):
    """파일의 인코딩 선언을 확인하고 필요시 추가"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        if lines and not lines[0].startswith('# -*- coding: utf-8 -*-'):
            if lines[0].startswith('#!'):
                # Shebang이 있는 경우 그 다음 줄에 추가
                lines.insert(1, '# -*- coding: utf-8 -*-')
            else:
                # 첫 번째 줄에 추가
                lines.insert(0, '# -*- coding: utf-8 -*-')

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f'Fixed encoding declaration in: {file_path}')
            return True
        else:
            print(f'Encoding declaration already exists in: {file_path}')
            return False
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return False

def main():
    """메인 함수"""
    print("Python 파일 인코딩 확인 중...")

    # 모든 Python 파일 확인
    py_files = glob.glob('source/**/*.py', recursive=True)
    fixed_count = 0

    for py_file in py_files:
        if check_and_fix_encoding(py_file):
            fixed_count += 1

    print(f"\n총 {len(py_files)}개 파일 확인 완료")
    print(f"{fixed_count}개 파일에 인코딩 선언 추가됨")

if __name__ == "__main__":
    main()
