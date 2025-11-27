#!/usr/bin/env python3
"""
Profile load_mosaic internals to find the real bottleneck.

This script measures time spent in each step INSIDE load_mosaic:
1. load_image (4x)
2. img4 creation (np.full)
3. image placement (slice assignment)
4. copy_paste
5. random_perspective

Usage:
    python tests/profile_mosaic_detail.py --data data/coco320.yaml --num-samples 100
"""

import argparse
import os
import sys
import time
import random
from collections import defaultdict

import numpy as np
import cv2
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import LoadImagesAndLabels, load_image, random_perspective, copy_paste
from utils.general import xywhn2xyxy, xyn2xy


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Timer:
    def __init__(self):
        self.times = defaultdict(list)

    def __call__(self, name):
        return TimerContext(self, name)

    def report(self, title="PROFILING RESULTS"):
        print(f"\n{'='*70}")
        print(title)
        print("="*70)

        total = 0
        results = []
        for name, times in self.times.items():
            avg = np.mean(times) * 1000  # ms
            std = np.std(times) * 1000
            total_time = np.sum(times) * 1000
            results.append((name, avg, std, total_time, len(times)))
            total += total_time

        # Sort by total time
        results.sort(key=lambda x: -x[3])

        print(f"\n{'Step':<35} {'Avg (ms)':<12} {'Std (ms)':<12} {'Total (ms)':<12} {'%':<8} {'Count'}")
        print("-"*80)
        for name, avg, std, total_time, count in results:
            pct = (total_time / total * 100) if total > 0 else 0
            print(f"{name:<35} {avg:<12.3f} {std:<12.3f} {total_time:<12.1f} {pct:<8.1f} {count}")

        print("-"*80)
        print(f"{'TOTAL':<35} {'':<12} {'':<12} {total:<12.1f} {'100.0':<8}")
        print("="*70)
        return total


class TimerContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.timer.times[self.name].append(time.perf_counter() - self.start)


def profile_load_mosaic_detail(dataset, index, timer):
    """Profile load_mosaic with detailed timing for each internal step."""
    s = dataset.img_size

    # Step 1: Random center calculation
    with timer("1.1 mosaic_center_calc"):
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in dataset.mosaic_border]
        indices = [index] + random.choices(dataset.indices, k=3)

    labels4, segments4 = [], []

    # Step 2: Load 4 images and place them
    for i, idx in enumerate(indices):
        # Load image
        with timer("1.2 load_image"):
            img, _, (h, w) = load_image(dataset, idx)

        # Create base image (only first iteration)
        if i == 0:
            with timer("1.3 create_img4"):
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)

        # Calculate placement coordinates
        with timer("1.4 calc_coords"):
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            padw = x1a - x1b
            padh = y1a - y1b

        # Place image
        with timer("1.5 place_image"):
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # Process labels
        with timer("1.6 process_labels"):
            labels, segments = dataset.labels[idx].copy(), dataset.segments[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

    # Concat/clip labels
    with timer("1.7 concat_labels"):
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)

    # copy_paste augmentation
    with timer("1.8 copy_paste"):
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=dataset.hyp['copy_paste'])

    # random_perspective (THE BIG ONE)
    with timer("1.9 random_perspective"):
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=dataset.hyp['degrees'],
                                           translate=dataset.hyp['translate'],
                                           scale=dataset.hyp['scale'],
                                           shear=dataset.hyp['shear'],
                                           perspective=dataset.hyp['perspective'],
                                           border=dataset.mosaic_border)

    return img4, labels4


def main():
    parser = argparse.ArgumentParser(description='Profile load_mosaic internals')
    parser.add_argument('--data', type=str, default='data/coco320.yaml')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to profile')
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"# load_mosaic Internal Profiler")
    print(f"{'#'*70}")
    print(f"\nSettings:")
    print(f"  Data: {args.data}")
    print(f"  Image size: {args.img_size}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Cache images: {args.cache_images}")

    # Load config
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)
    train_path = data_dict['train']

    # Hyp
    hyp = {
        'mosaic': 1.0,
        'mixup': 0.0,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'paste_in': 0.0,
        'copy_paste': 0.0,
    }

    print(f"\nLoading dataset...")
    set_seed(args.seed)

    dataset = LoadImagesAndLabels(
        train_path,
        args.img_size,
        args.batch_size,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=args.cache_images,
        single_cls=False,
        stride=32,
        pad=0.0,
        image_weights=False,
        prefix='profile: '
    )
    print(f"Dataset size: {len(dataset)} images")

    # Generate random indices
    set_seed(args.seed)
    indices = random.sample(range(len(dataset)), args.num_samples)

    # Profile
    print(f"\n{'='*70}")
    print("PROFILING: load_mosaic INTERNALS")
    print(f"{'='*70}")

    timer = Timer()

    for i, idx in enumerate(indices):
        profile_load_mosaic_detail(dataset, idx, timer)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{args.num_samples} samples...")

    total = timer.report("load_mosaic INTERNAL BREAKDOWN")

    # Summary
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}")

    # Group by category
    load_times = sum(sum(v) for k, v in timer.times.items() if 'load_image' in k) * 1000
    place_times = sum(sum(v) for k, v in timer.times.items() if 'place_image' in k) * 1000
    persp_times = sum(sum(v) for k, v in timer.times.items() if 'random_perspective' in k) * 1000
    other_times = total - load_times - place_times - persp_times

    print(f"\nTime breakdown:")
    print(f"  load_image (4x per mosaic):     {load_times:8.1f} ms ({load_times/total*100:5.1f}%)")
    print(f"  place_image (slice assignment): {place_times:8.1f} ms ({place_times/total*100:5.1f}%)")
    print(f"  random_perspective:             {persp_times:8.1f} ms ({persp_times/total*100:5.1f}%)")
    print(f"  Other operations:               {other_times:8.1f} ms ({other_times/total*100:5.1f}%)")
    print(f"\n  TOTAL: {total:.1f} ms for {args.num_samples} samples")
    print(f"  Average: {total/args.num_samples:.3f} ms per sample")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
