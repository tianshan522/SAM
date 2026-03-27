#!/usr/bin/env python3
"""
SAM 模型评估脚本：计算 FID、身份保持度（ArcFace Cosine Similarity）、年龄准确度（DEX）。

用法:
    python scripts/local/evaluate.py \
        --generated_dir  workspace/runs/quick_infer_manual/inference_results/1 \
        --source_dir     notebooks/images \
        --target_age     1

    --generated_dir  推理生成的图片目录（与 source_dir 中的文件名一一对应）
    --source_dir     原始输入图片目录（用于计算身份保持度）
    --real_dir       真实图片目录（用于计算 FID，可选，默认使用 source_dir）
    --target_age     期望的目标年龄（用于计算年龄准确度）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from models.dex_vgg import VGG


def load_arcface(device):
    model = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.load_state_dict(torch.load(model_paths['ir_se50'], map_location='cpu'))
    model.to(device).eval()
    return model


def load_dex(device):
    model = VGG()
    ckpt = torch.load(model_paths['age_predictor'], map_location='cpu')['state_dict']
    ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


def predict_age(dex_model, img_tensor):
    """img_tensor: (1, 3, H, W) normalised to [-1, 1]"""
    x = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        pb = F.softmax(dex_model(x)['fc8'], dim=1)
    ages = torch.arange(0, pb.shape[1], dtype=pb.dtype, device=pb.device)
    return (pb * ages).sum(dim=1).item()


def extract_arcface_feat(model, img_tensor, face_pool):
    """img_tensor: (1, 3, 256, 256) normalised to [-1, 1]"""
    x = img_tensor[:, :, 35:223, 32:220]
    x = face_pool(x)
    with torch.no_grad():
        feat = model(x)
    return feat


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

transform_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def list_images(directory):
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def compute_identity_and_age(args, device):
    arcface = load_arcface(device)
    face_pool = torch.nn.AdaptiveAvgPool2d((112, 112)).to(device)
    dex = load_dex(device)

    gen_images = list_images(args.generated_dir)
    if not gen_images:
        print(f"[ERROR] generated_dir 中没有找到图片: {args.generated_dir}")
        return

    src_dir = Path(args.source_dir)
    similarities = []
    age_errors = []
    predicted_ages = []

    for gen_path in gen_images:
        src_path = src_dir / gen_path.name
        if not src_path.exists():
            print(f"  [SKIP] 源图片不存在: {src_path}")
            continue

        gen_img = transform_256(Image.open(gen_path).convert('RGB')).unsqueeze(0).to(device)
        src_img = transform_256(Image.open(src_path).convert('RGB')).unsqueeze(0).to(device)

        gen_feat = extract_arcface_feat(arcface, gen_img, face_pool)
        src_feat = extract_arcface_feat(arcface, src_img, face_pool)
        cos_sim = F.cosine_similarity(gen_feat, src_feat).item()
        similarities.append(cos_sim)

        age = predict_age(dex, gen_img)
        predicted_ages.append(age)
        if args.target_age is not None:
            age_errors.append(abs(age - args.target_age))

        print(f"  {gen_path.name}: cos_sim={cos_sim:.4f}, predicted_age={age:.1f}")

    print("\n========== 评估结果 ==========")
    if similarities:
        print(f"身份保持度 (ArcFace Cosine Similarity):")
        print(f"  均值: {np.mean(similarities):.4f}")
        print(f"  标准差: {np.std(similarities):.4f}")
        print(f"  最小值: {np.min(similarities):.4f}")
        print(f"  最大值: {np.max(similarities):.4f}")

    if predicted_ages:
        print(f"\n预测年龄分布:")
        print(f"  均值: {np.mean(predicted_ages):.1f}")
        print(f"  标准差: {np.std(predicted_ages):.1f}")

    if age_errors:
        print(f"\n年龄准确度 (目标={args.target_age}):")
        print(f"  平均绝对误差 (MAE): {np.mean(age_errors):.2f}")

    return {
        'id_sim_mean': float(np.mean(similarities)) if similarities else None,
        'age_mae': float(np.mean(age_errors)) if age_errors else None,
        'age_mean': float(np.mean(predicted_ages)) if predicted_ages else None,
    }


def compute_fid(args):
    real_dir = args.real_dir or args.source_dir
    gen_dir = args.generated_dir
    print(f"\n计算 FID ...")
    print(f"  真实图片目录: {real_dir}")
    print(f"  生成图片目录: {gen_dir}")

    real_count = len(list_images(real_dir))
    gen_count = len(list_images(gen_dir))
    if real_count < 2 or gen_count < 2:
        print(f"  [SKIP] FID 需要至少 2 张图片 (real={real_count}, gen={gen_count})")
        print(f"  提示: FID 在大数据集 (>1000) 上才有统计意义")
        return None

    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(gen_dir)],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048,
        )
        print(f"\nFID: {fid:.2f}")
        print(f"  (越低越好，< 50 通常表示不错的生成质量)")
        return fid
    except Exception as e:
        print(f"  [ERROR] FID 计算失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='SAM 模型评估')
    parser.add_argument('--generated_dir', type=str, required=True,
                        help='推理生成的图片目录')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='原始输入图片目录 (用于身份保持度)')
    parser.add_argument('--real_dir', type=str, default=None,
                        help='真实图片目录 (用于 FID, 默认同 source_dir)')
    parser.add_argument('--target_age', type=float, default=None,
                        help='期望的目标年龄')
    parser.add_argument('--skip_fid', action='store_true',
                        help='跳过 FID 计算')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"生成图片目录: {args.generated_dir}")
    print(f"源图片目录: {args.source_dir}")
    if args.target_age is not None:
        print(f"目标年龄: {args.target_age}")
    print()

    metrics = compute_identity_and_age(args, device)

    if not args.skip_fid:
        fid = compute_fid(args)
        if metrics and fid is not None:
            metrics['fid'] = fid

    print("\n========== 完成 ==========")


if __name__ == '__main__':
    main()
