#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정규식 테스트
"""

import re

def test_regex():
    test_content = '제5조(직무교육 소집) 농림축산식품부장관·검역본부장 또는 시·도지사는 법 제6조제2항 또는 제3항에 따라 공중방역수의사에 대한 직무교육을 실시하기 위하여 해당공중방역수의사를 소집할 때에는 소집일 5일 전까지 소집대상자의 인적사항·소집일시 및 장소 등 필요한 사항을 명시하여 소집통지를 하여야 한다.'
    
    pattern = re.compile(r'제(\d+(?:의\d+)?)조(?:\s*\(([^)]+)\))?')
    match = pattern.search(test_content)
    
    if match:
        print('Article number: 제' + match.group(1) + '조')
        print('Article title: ' + (match.group(2) if match.group(2) else 'None'))
        print('Full match: ' + match.group(0))
    else:
        print('No match found')

if __name__ == "__main__":
    test_regex()
