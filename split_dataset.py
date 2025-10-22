import os
import shutil
from pathlib import Path
import math

def split_dataset(input_root, output_root, num_parts):
    """
    Chia dataset gốc thành nhiều phần (shard).
    - input_root: thư mục gốc chứa 'smoking' và 'notsmoking'
    - output_root: thư mục để lưu các part-xxxx
    - num_parts: số phần muốn chia
    """

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    classes = ["smoking", "notsmoking"]

    # Đếm số ảnh mỗi class
    files_by_class = {}
    for cls in classes:
        files = list((input_root / cls).glob("*"))
        files_by_class[cls] = files
        print(f"[INFO] Class {cls}: {len(files)} files")

    # Chia file mỗi class thành nhiều part
    for cls, files in files_by_class.items():
        part_size = math.ceil(len(files) / num_parts)

        for i in range(num_parts):
            part_dir = output_root / f"part-{i:04d}" / cls
            part_dir.mkdir(parents=True, exist_ok=True)

            start = i * part_size
            end = min((i + 1) * part_size, len(files))
            shard_files = files[start:end]

            for f in shard_files:
                shutil.copy(f, part_dir)

            print(f"[OK] Copied {len(shard_files)} {cls} files to {part_dir}")

    print(f"\n✅ Dataset split done! Parts saved in {output_root}")


if __name__ == "__main__":
    # Ví dụ: dataset gốc ở /kaggle/working/clip_repo/data/smoking_datasets
    input_root = r"G:\smoking_datasets"
    output_root = r"G:\smoking_datasets_split"

    num_parts = 4  # Chia thành 4 phần (bạn có thể đổi tùy dataset lớn nhỏ)

    split_dataset(input_root, output_root, num_parts)
