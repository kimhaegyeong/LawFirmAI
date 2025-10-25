# Windows 환경에서 Python UTF-8 인코딩 설정
# 이 파일을 실행하여 환경 변수를 설정하세요

echo "Python UTF-8 인코딩 환경 변수 설정 중..."

# 현재 사용자용 환경 변수 설정
setx PYTHONIOENCODING "utf-8"
setx PYTHONUTF8 "1"

echo "환경 변수 설정 완료!"
echo "새로운 터미널을 열어서 변경사항을 적용하세요."
echo ""
echo "설정된 환경 변수:"
echo "PYTHONIOENCODING=utf-8"
echo "PYTHONUTF8=1"
