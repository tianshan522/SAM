#!/usr/bin/env python3
"""
将原始人脸图片目录整理为 SAM 训练所需的格式：
- 统一重命名为 00000.png, 00001.png, ...
- 统一转为 RGB PNG，resize 到 256x256
- 自动划分 train / test（默认 90% / 10%）
"""
import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def collect_images(src_dir: Path) -> list[Path]:
    images = []
    for p in sorted(src_dir.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return images


def main():
    parser = argparse.ArgumentParser(description='整理人脸数据集')
    parser.add_argument('--src', type=str, required=True,
                        help='原始图片目录')
    parser.add_argument('--dst', type=str, required=True,
                        help='输出目录（会创建 train/ 和 test/ 子目录）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例 (默认 0.1)')
    parser.add_argument('--size', type=int, default=256,
                        help='输出图片尺寸 (默认 256)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--dry_run', action='store_true',
                        help='只统计不实际复制')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    images = collect_images(src_dir)
    print(f"找到 {len(images)} 张图片")

    if not images:
        print("没有找到图片，退出。")
        return

    random.seed(args.seed)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - args.test_ratio))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    print(f"训练集: {len(train_images)} 张")
    print(f"测试集: {len(test_images)} 张")

    if args.dry_run:
        print("[DRY RUN] 不执行实际操作。")
        return

    train_dir = dst_dir / 'train'
    test_dir = dst_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def process_split(image_list, out_dir, label):
        errors = []
        for idx, img_path in enumerate(tqdm(image_list, desc=label)):
            out_name = f"{idx:05d}.png"
            out_path = out_dir / out_name
            try:
                img = Image.open(img_path).convert('RGB')
                if img.size != (args.size, args.size):
                    img = img.resize((args.size, args.size), Image.LANCZOS)
                img.save(out_path)
            except Exception as e:
                errors.append((img_path, str(e)))
        if errors:
            print(f"\n  {label} 有 {len(errors)} 张图片处理失败:")
            for p, err in errors[:10]:
                print(f"    {p}: {err}")

    process_split(train_images, train_dir, "训练集")
    process_split(test_images, test_dir, "测试集")

    print(f"\n完成！输出目录: {dst_dir}")
    print(f"  训练集: {train_dir} ({len(list(train_dir.glob('*.png')))} 张)")
    print(f"  测试集: {test_dir} ({len(list(test_dir.glob('*.png')))} 张)")


if __name__ == '__main__':
    main()
