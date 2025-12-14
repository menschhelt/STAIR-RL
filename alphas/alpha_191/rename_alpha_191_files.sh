#!/bin/bash

# Alpha 158 파일명을 3자리 숫자 형식으로 변경하는 스크립트
# alpha_158_1.py -> alpha_158_001.py

echo "Alpha 191 파일명을 3자리 숫자 형식으로 변경을 시작합니다..."

# 현재 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# 변경된 파일 수를 카운트
changed_count=0

# alpha_158_[숫자].py 형식의 파일들을 찾아서 처리
for file in alpha_191_*.py; do
    # 파일이 존재하는지 확인
    if [[ -f "$file" ]]; then
        # 파일명에서 숫자 부분을 추출
        if [[ $file =~ alpha_191_([0-9]+)\.py$ ]]; then
            number="${BASH_REMATCH[1]}"
            
            # 숫자가 이미 3자리인지 확인 (100 이상)
            if [[ ${#number} -lt 3 ]]; then
                # 3자리로 패딩된 새 파일명 생성
                new_number=$(printf "%03d" $number)
                new_filename="alpha_191_${new_number}.py"
                
                # 파일명 변경
                if mv "$file" "$new_filename"; then
                    echo "✓ $file -> $new_filename"
                    ((changed_count++))
                else
                    echo "✗ 실패: $file -> $new_filename"
                fi
            else
                echo "- $file (이미 3자리, 변경 불필요)"
            fi
        fi
    fi
done

echo ""
echo "파일명 변경 완료!"
echo "총 $changed_count 개의 파일이 변경되었습니다."

# 변경 후 파일 목록 확인 (처음 10개만)
echo ""
echo "변경된 파일 목록 (처음 10개):"
ls alpha_191_*.py | head -10

echo ""
echo "전체 alpha_191 파일 개수:"
ls alpha_191_*.py | wc -l