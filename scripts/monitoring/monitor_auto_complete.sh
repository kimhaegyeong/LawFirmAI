#!/bin/bash
# 자동 완료 스크립트 모니터링 (Linux/Mac용)

LOG_FILE="logs/auto_complete_dynamic_chunking.log"
ERROR_LOG="logs/auto_complete_dynamic_chunking_error.log"

echo "============================================================"
echo "자동 완료 스크립트 모니터링"
echo "============================================================"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "✓ 로그 파일: $LOG_FILE"
    echo "  크기: $(du -h "$LOG_FILE" | cut -f1)"
    echo "  수정 시간: $(stat -f "%Sm" "$LOG_FILE" 2>/dev/null || stat -c "%y" "$LOG_FILE" 2>/dev/null)"
    echo ""
    echo "최근 로그 (마지막 20줄):"
    echo "------------------------------------------------------------"
    tail -20 "$LOG_FILE"
    echo "------------------------------------------------------------"
else
    echo "✗ 로그 파일 없음: $LOG_FILE"
fi

echo ""

if [ -f "$ERROR_LOG" ]; then
    if [ -s "$ERROR_LOG" ]; then
        echo "⚠️  에러 로그에 내용이 있습니다:"
        tail -10 "$ERROR_LOG"
    else
        echo "✓ 에러 로그 없음 (정상)"
    fi
fi

echo ""
echo "프로세스 확인:"
ps aux | grep "auto_complete_dynamic_chunking" | grep -v grep

