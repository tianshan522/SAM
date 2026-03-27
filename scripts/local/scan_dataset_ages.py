#!/usr/bin/env python3
"""
Scan a dataset directory with the DEX age predictor and output age distribution.

Usage:
    python scripts/local/scan_dataset_ages.py --src workspace/datasets/humans_aligned/train
    python scripts/local/scan_dataset_ages.py --src workspace/datasets/humans_aligned/train --save ages.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs.paths_config import model_paths
from models.dex_vgg import VGG

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

transform_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_dex(device):
    model = VGG()
    ckpt = torch.load(model_paths['age_predictor'], map_location='cpu')['state_dict']
    ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


def predict_age(dex_model, img_tensor):
    x = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        pb = F.softmax(dex_model(x)['fc8'], dim=1)
    ages = torch.arange(0, pb.shape[1], dtype=pb.dtype, device=pb.device)
    return (pb * ages).sum(dim=1).item()


def collect_images(src_dir: Path):
    return sorted(
        p for p in src_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main():
    parser = argparse.ArgumentParser(description='DEX age distribution scanner')
    parser.add_argument('--src', type=str, required=True, help='Image directory to scan')
    parser.add_argument('--save', type=str, default=None, help='Save per-image ages to JSON')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for inference')
    args = parser.parse_args()

    src_dir = Path(args.src)
    images = collect_images(src_dir)
    print(f'Found {len(images)} images in {src_dir}')
    if not images:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dex = load_dex(device)

    age_map = {}
    ages_list = []

    for i in range(0, len(images), args.batch):
        batch_paths = images[i:i + args.batch]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                tensors.append(transform_256(img))
            except Exception as e:
                print(f'  [SKIP] {p.name}: {e}')
                continue

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        x = F.interpolate(batch_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pb = F.softmax(dex(x)['fc8'], dim=1)
        age_range = torch.arange(0, pb.shape[1], dtype=pb.dtype, device=pb.device)
        predicted_ages = (pb * age_range).sum(dim=1).cpu().tolist()

        for p, age in zip(batch_paths, predicted_ages):
            age_map[p.name] = round(age, 2)
            ages_list.append(age)

        if (i // args.batch) % 20 == 0:
            print(f'  processed {min(i + args.batch, len(images))}/{len(images)}')

    print(f'\nScanned {len(ages_list)} images.\n')

    bins = [(0, 5), (6, 10), (11, 20), (21, 30), (31, 40),
            (41, 50), (51, 60), (61, 70), (71, 80), (81, 100)]
    print(f'{"Age range":>12s}  {"Count":>6s}  {"Pct":>6s}  Histogram')
    print('-' * 60)
    for lo, hi in bins:
        count = sum(1 for a in ages_list if lo <= a <= hi)
        pct = count / len(ages_list) * 100 if ages_list else 0
        bar = '#' * int(pct / 2)
        print(f'  {lo:3d} - {hi:3d}    {count:6d}  {pct:5.1f}%  {bar}')

    import statistics
    print(f'\nMean age: {statistics.mean(ages_list):.1f}')
    print(f'Median age: {statistics.median(ages_list):.1f}')
    print(f'Std dev: {statistics.stdev(ages_list):.1f}')
    print(f'Min: {min(ages_list):.1f}  Max: {max(ages_list):.1f}')

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(age_map, f, indent=2)
        print(f'\nSaved per-image ages to {save_path}')


if __name__ == '__main__':
    main()
