#!/usr/bin/env python3
"""
Profile __getitem__ to find the real bottleneck.

This script measures time spent in each step of data loading:
1. load_mosaic / load_image
2. augmentation (random_perspective, hsv, flip)
3. tensor conversion
4. collate (stack)

Usage:
    python tests/profile_getitem.py --data data/coco320.yaml --num-samples 100
"""

import argparse
import os
import sys
import time
import random
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import LoadImagesAndLabels, letterbox, load_mosaic, load_mosaic9
from utils.datasets import load_image, augment_hsv, random_perspective
from utils.general import xywhn2xyxy, xyxy2xywh


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Timer:
    def __init__(self):
        self.times = defaultdict(list)

    def __call__(self, name):
        return TimerContext(self, name)

    def report(self):
        print("\n" + "="*70)
        print("PROFILING RESULTS")
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

        print(f"\n{'Step':<30} {'Avg (ms)':<12} {'Std (ms)':<12} {'Total (ms)':<12} {'%':<8} {'Count'}")
        print("-"*70)
        for name, avg, std, total_time, count in results:
            pct = (total_time / total * 100) if total > 0 else 0
            print(f"{name:<30} {avg:<12.3f} {std:<12.3f} {total_time:<12.1f} {pct:<8.1f} {count}")

        print("-"*70)
        print(f"{'TOTAL':<30} {'':<12} {'':<12} {total:<12.1f} {'100.0':<8}")
        print("="*70)


class TimerContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.timer.times[self.name].append(time.perf_counter() - self.start)


def profile_getitem_mosaic_on(dataset, index, timer):
    """Profile __getitem__ with mosaic ON."""
    hyp = dataset.hyp

    # Step 1: Load mosaic
    with timer("1. load_mosaic"):
        if random.random() < 0.8:
            img, labels = load_mosaic(dataset, index)
        else:
            img, labels = load_mosaic9(dataset, index)
    shapes = None

    # Step 2: MixUp (usually skipped, but measure anyway)
    with timer("2. mixup_check"):
        do_mixup = random.random() < hyp.get('mixup', 0)

    if do_mixup:
        with timer("2. mixup"):
            if random.random() < 0.8:
                img2, labels2 = load_mosaic(dataset, random.randint(0, len(dataset.labels) - 1))
            else:
                img2, labels2 = load_mosaic9(dataset, random.randint(0, len(dataset.labels) - 1))
            r = np.random.beta(8.0, 8.0)
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
            labels = np.concatenate((labels, labels2), 0)

    # Step 3: HSV augmentation
    with timer("3. augment_hsv"):
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

    # Step 4: Flip
    with timer("4. flip"):
        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]

        if random.random() < hyp.get('flipud', 0):
            img = np.flipud(img)
            if nL:
                labels[:, 2] = 1 - labels[:, 2]

        if random.random() < hyp.get('fliplr', 0.5):
            img = np.fliplr(img)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]

    # Step 5: Label tensor
    with timer("5. label_tensor"):
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

    # Step 6: BGR->RGB, HWC->CHW
    with timer("6. transpose"):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

    # Step 7: to tensor
    with timer("7. from_numpy"):
        img = torch.from_numpy(img)

    return img, labels_out


def profile_getitem_mosaic_off(dataset, index, timer):
    """Profile __getitem__ with mosaic OFF."""
    hyp = dataset.hyp

    # Step 1: Load image
    with timer("1. load_image"):
        img, (h0, w0), (h, w) = load_image(dataset, index)

    # Step 2: Letterbox
    with timer("2. letterbox"):
        shape = dataset.img_size
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=True)

    # Step 3: Copy labels
    with timer("3. copy_labels"):
        labels = dataset.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

    # Step 4: random_perspective
    with timer("4. random_perspective"):
        img, labels = random_perspective(img, labels,
                                         degrees=hyp['degrees'],
                                         translate=hyp['translate'],
                                         scale=hyp['scale'],
                                         shear=hyp['shear'],
                                         perspective=hyp['perspective'])

    # Step 5: HSV augmentation
    with timer("5. augment_hsv"):
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

    # Step 6: Flip
    with timer("6. flip"):
        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]

        if random.random() < hyp.get('flipud', 0):
            img = np.flipud(img)
            if nL:
                labels[:, 2] = 1 - labels[:, 2]

        if random.random() < hyp.get('fliplr', 0.5):
            img = np.fliplr(img)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]

    # Step 7: Label tensor
    with timer("7. label_tensor"):
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

    # Step 8: BGR->RGB, HWC->CHW
    with timer("8. transpose"):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

    # Step 9: to tensor
    with timer("9. from_numpy"):
        img = torch.from_numpy(img)

    return img, labels_out


def profile_collate(imgs, labels, timer):
    """Profile collate function."""
    with timer("collate.stack"):
        img_batch = torch.stack(imgs, 0)

    with timer("collate.label_idx"):
        for i, l in enumerate(labels):
            l[:, 0] = i

    with timer("collate.cat"):
        label_batch = torch.cat(labels, 0)

    return img_batch, label_batch


def main():
    parser = argparse.ArgumentParser(description='Profile __getitem__')
    parser.add_argument('--data', type=str, default='data/coco320.yaml')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to profile')
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"# __getitem__ Profiler")
    print(f"{'#'*70}")
    print(f"\nSettings:")
    print(f"  Data: {args.data}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
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

    # =========================================================================
    # Profile MOSAIC ON
    # =========================================================================
    print(f"\n{'='*70}")
    print("PROFILING: MOSAIC ON")
    print(f"{'='*70}")

    timer_mosaic_on = Timer()
    imgs_on = []
    labels_on = []

    for i, idx in enumerate(indices):
        img, label = profile_getitem_mosaic_on(dataset, idx, timer_mosaic_on)
        imgs_on.append(img)
        labels_on.append(label)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{args.num_samples} samples...")

    # Profile collate
    print(f"  Profiling collate for {len(imgs_on)} images...")
    for i in range(0, len(imgs_on), args.batch_size):
        batch_imgs = imgs_on[i:i+args.batch_size]
        batch_labels = labels_on[i:i+args.batch_size]
        if len(batch_imgs) == args.batch_size:
            profile_collate(batch_imgs, batch_labels, timer_mosaic_on)

    timer_mosaic_on.report()

    # =========================================================================
    # Profile MOSAIC OFF
    # =========================================================================
    print(f"\n{'='*70}")
    print("PROFILING: MOSAIC OFF")
    print(f"{'='*70}")

    timer_mosaic_off = Timer()
    imgs_off = []
    labels_off = []

    for i, idx in enumerate(indices):
        img, label = profile_getitem_mosaic_off(dataset, idx, timer_mosaic_off)
        imgs_off.append(img)
        labels_off.append(label)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{args.num_samples} samples...")

    # Profile collate
    print(f"  Profiling collate for {len(imgs_off)} images...")
    for i in range(0, len(imgs_off), args.batch_size):
        batch_imgs = imgs_off[i:i+args.batch_size]
        batch_labels = labels_off[i:i+args.batch_size]
        if len(batch_imgs) == args.batch_size:
            profile_collate(batch_imgs, batch_labels, timer_mosaic_off)

    timer_mosaic_off.report()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}")

    total_on = sum(sum(t) for t in timer_mosaic_on.times.values()) * 1000
    total_off = sum(sum(t) for t in timer_mosaic_off.times.values()) * 1000

    print(f"\nMosaic ON  total: {total_on:.1f} ms for {args.num_samples} samples")
    print(f"Mosaic OFF total: {total_off:.1f} ms for {args.num_samples} samples")
    print(f"Ratio: Mosaic ON is {total_on/total_off:.2f}x slower than Mosaic OFF")

    # Find biggest difference
    print(f"\nBiggest time consumers (Mosaic ON):")
    on_times = [(k, sum(v)*1000) for k, v in timer_mosaic_on.times.items()]
    on_times.sort(key=lambda x: -x[1])
    for name, t in on_times[:3]:
        print(f"  {name}: {t:.1f} ms ({t/total_on*100:.1f}%)")


if __name__ == '__main__':
    main()
