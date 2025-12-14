#!/usr/bin/env python3
"""
Alpha 파일들을 Wide-format DataFrame 지원하도록 일괄 수정

변경사항:
1. Dict import 추가
2. calculate() 시그니처 변경
3. dataframe -> data 변수명 변경
"""

import os
import re
import glob

def transform_alpha_file(filepath: str) -> bool:
    """단일 알파 파일 변환"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. Dict import 추가 (이미 있으면 스킵)
        if 'from typing import Dict' not in content:
            # from pandas import DataFrame 다음에 추가
            if 'from pandas import DataFrame' in content:
                content = content.replace(
                    'from pandas import DataFrame',
                    'from pandas import DataFrame\nfrom typing import Dict'
                )
            else:
                # import pandas as pd 다음에 추가
                content = re.sub(
                    r'(import pandas as pd\n)',
                    r'\1from typing import Dict\n',
                    content
                )

        # 2. calculate() 시그니처 변경
        # 패턴: def calculate(self, dataframe: DataFrame, pair: str) -> pd.Series:
        old_signature_patterns = [
            r'def calculate\(self, dataframe: DataFrame, pair: str\) -> pd\.Series:',
            r'def calculate\(self, dataframe: DataFrame, pair: str\) -> pd\.DataFrame:',
            r'def calculate\(self, dataframe: DataFrame, pair: str\):',
        ]
        new_signature = 'def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:'

        for pattern in old_signature_patterns:
            content = re.sub(pattern, new_signature, content)

        # 3. dataframe['xxx'] -> data['xxx'] 변경
        content = re.sub(r"dataframe\['", "data['", content)
        content = re.sub(r'dataframe\["', 'data["', content)

        # 4. 남아있는 dataframe 변수 참조 변경 (메서드 호출 등)
        # dataframe.xxx -> data.xxx
        content = re.sub(r'\bdataframe\.', 'data.', content)

        # 변경이 있는 경우만 저장
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    base_path = '/home/work/RL/stair-local/alphas'

    # alpha_101 파일들
    alpha_101_files = glob.glob(os.path.join(base_path, 'alpha_101', 'alpha_101_*.py'))

    # alpha_191 파일들
    alpha_191_files = glob.glob(os.path.join(base_path, 'alpha_191', 'alpha_191_*.py'))

    all_files = alpha_101_files + alpha_191_files

    print(f"Total files to process: {len(all_files)}")
    print(f"  - alpha_101: {len(alpha_101_files)}")
    print(f"  - alpha_191: {len(alpha_191_files)}")

    modified_count = 0
    failed_files = []

    for filepath in sorted(all_files):
        filename = os.path.basename(filepath)
        result = transform_alpha_file(filepath)
        if result:
            modified_count += 1
            print(f"✓ Modified: {filename}")
        else:
            # 변경이 없거나 실패
            with open(filepath, 'r') as f:
                content = f.read()
            if 'Dict[str, pd.DataFrame]' in content:
                print(f"  Skipped (already converted): {filename}")
            else:
                failed_files.append(filename)
                print(f"✗ Failed: {filename}")

    print(f"\n=== Summary ===")
    print(f"Modified: {modified_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print(f"Failed files: {failed_files}")


if __name__ == '__main__':
    main()
