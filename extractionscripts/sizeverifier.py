import os
import sys
from PIL import Image


def validate_jpg(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
            if img.format != 'JPEG':
                return False
        return True
    except Exception:
        return False

def main(folder):
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        return

    # os.walk is recursive, so it naturally traverses all subdirectories.
    for root, _, files in os.walk(folder):
        for entry in files:
            if entry.lower().endswith(('.jpg', '.jpeg')):
                full_path = os.path.join(root, entry)
                if validate_jpg(full_path):
                    pass
                else:
                    print(f"{full_path}: Invalid JPEG")

main('/Users/apple/Documents/Git_grapeDisease_test/microbify/scidb/')