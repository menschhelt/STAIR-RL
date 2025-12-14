#!/usr/bin/env python3
"""
data.columns -> data 로 변경 (Dict 키 체크)
"""

import os
import re
import glob

def fix_file(filepath: str) -> bool:
    """파일에서 data.columns를 data로 변경"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # 'xxx' not in data.columns -> 'xxx' not in data
        content = re.sub(
            r"'(\w+)'\s+not\s+in\s+data\.columns",
            r"'\1' not in data",
            content
        )

        # 'xxx' in data.columns -> 'xxx' in data
        content = re.sub(
            r"'(\w+)'\s+in\s+data\.columns",
            r"'\1' in data",
            content
        )

        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error: {filepath}: {e}")
        return False


def main():
    base_path = '/home/work/RL/stair-local/alphas'

    files = glob.glob(os.path.join(base_path, 'alpha_101', 'alpha_101_*.py'))
    files += glob.glob(os.path.join(base_path, 'alpha_191', 'alpha_191_*.py'))

    modified = 0
    for f in sorted(files):
        if fix_file(f):
            print(f"✓ Fixed: {os.path.basename(f)}")
            modified += 1

    print(f"\n=== Modified: {modified} files ===")


if __name__ == '__main__':
    main()
