#!/usr/bin/env python3
"""
Phase 2 Micro-Collate Verification Script

This script verifies that the Micro-Collate pipeline produces
mathematically identical output to the original pipeline.

Usage:
    python tests/verify_micro_collate.py --data data/coco320.yaml --batch-size 16 --micro-batch-size 4
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets import (
    LoadImagesAndLabels,
    MicroBatchSampler,
    MicroBatchDataset,
    collate_fn_micro,
)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_original_batch(dataset, indices):
    """Create a batch using the original pipeline (simulating collate_fn)."""
    imgs = []
    labels = []
    paths = []
    shapes = []

    for idx in indices:
        img, label, path, shape = dataset[idx]
        imgs.append(img)
        labels.append(label)
        paths.append(path)
        shapes.append(shape)

    # Original collate_fn behavior
    img_batch = torch.stack(imgs, 0)

    for i, l in enumerate(labels):
        l[:, 0] = i  # add target image index

    label_batch = torch.cat(labels, 0)

    return img_batch, label_batch, paths, shapes


def create_micro_batch(dataset, indices, micro_batch_size):
    """Create a batch using the Micro-Collate pipeline."""
    # Wrap dataset
    micro_dataset = MicroBatchDataset(dataset)

    # Group indices into micro-batches
    micro_batches_indices = []
    for i in range(0, len(indices), micro_batch_size):
        micro_batches_indices.append(indices[i:i + micro_batch_size])

    # Simulate worker processing
    batch = []
    for micro_indices in micro_batches_indices:
        result = micro_dataset[micro_indices]
        batch.append(result)

    # Apply micro collate function
    img_batch, label_batch, paths, shapes = collate_fn_micro(batch)

    return img_batch, label_batch, list(paths), list(shapes)


def verify_batches(batch1, batch2, batch_name="batch"):
    """Verify two batches are identical."""
    img1, label1, paths1, shapes1 = batch1
    img2, label2, paths2, shapes2 = batch2

    print(f"\n{'='*60}")
    print(f"Verifying {batch_name}")
    print(f"{'='*60}")

    # Check image tensors
    print(f"\nImage tensor shapes: {img1.shape} vs {img2.shape}")
    if img1.shape != img2.shape:
        print(f"  [FAIL] Shape mismatch!")
        return False

    img_match = torch.allclose(img1.float(), img2.float(), rtol=1e-5, atol=1e-5)
    if img_match:
        print(f"  [PASS] Image tensors are identical (torch.allclose)")
    else:
        diff = (img1.float() - img2.float()).abs()
        print(f"  [FAIL] Image tensors differ! Max diff: {diff.max().item()}")
        return False

    # Check label tensors
    print(f"\nLabel tensor shapes: {label1.shape} vs {label2.shape}")
    if label1.shape != label2.shape:
        print(f"  [FAIL] Shape mismatch!")
        return False

    label_match = torch.allclose(label1.float(), label2.float(), rtol=1e-5, atol=1e-5)
    if label_match:
        print(f"  [PASS] Label tensors are identical (torch.allclose)")
    else:
        diff = (label1.float() - label2.float()).abs()
        print(f"  [FAIL] Label tensors differ! Max diff: {diff.max().item()}")
        return False

    # Check paths
    paths_match = list(paths1) == list(paths2)
    if paths_match:
        print(f"\n  [PASS] Paths are identical ({len(paths1)} images)")
    else:
        print(f"\n  [FAIL] Paths differ!")
        return False

    # Check shapes
    shapes_match = list(shapes1) == list(shapes2)
    if shapes_match:
        print(f"  [PASS] Shapes are identical")
    else:
        print(f"  [FAIL] Shapes differ!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Verify Micro-Collate Pipeline')
    parser.add_argument('--data', type=str, default='data/coco320.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--micro-batch-size', type=int, default=4, help='micro batch size')
    parser.add_argument('--img-size', type=int, default=320, help='image size')
    parser.add_argument('--num-tests', type=int, default=3, help='number of test batches')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--augment', action='store_true', help='enable augmentation (mosaic, etc.)')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Phase 2 Micro-Collate Verification")
    print(f"{'#'*60}")
    print(f"\nSettings:")
    print(f"  Data: {args.data}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Micro-batch size: {args.micro_batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Random seed: {args.seed}")

    # Load data config
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)

    train_path = data_dict['train']
    print(f"\nTrain path: {train_path}")

    # Create hyp dict (minimal for testing)
    hyp = {
        'mosaic': 1.0 if args.augment else 0.0,
        'mixup': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0 if not args.augment else 0.5,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'paste_in': 0.0,
    }

    # Create dataset
    print(f"\nLoading dataset...")

    class FakeOpt:
        single_cls = False

    set_seed(args.seed)
    dataset = LoadImagesAndLabels(
        train_path,
        args.img_size,
        args.batch_size,
        augment=args.augment,
        hyp=hyp,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        image_weights=False,
        prefix='verify: '
    )

    print(f"Dataset size: {len(dataset)} images")

    # Run verification tests
    all_passed = True
    for test_idx in range(args.num_tests):
        print(f"\n{'='*60}")
        print(f"Test {test_idx + 1}/{args.num_tests}")
        print(f"{'='*60}")

        # Generate random indices
        set_seed(args.seed + test_idx)
        indices = random.sample(range(len(dataset)), args.batch_size)
        print(f"\nIndices: {indices[:8]}..." if len(indices) > 8 else f"\nIndices: {indices}")

        # Create batch with original pipeline
        print("\n[1] Creating batch with ORIGINAL pipeline...")
        set_seed(args.seed + test_idx + 1000)  # Same seed for both
        batch1 = create_original_batch(dataset, indices)
        print(f"    Image shape: {batch1[0].shape}, Label shape: {batch1[1].shape}")

        # Create batch with micro-collate pipeline
        print("\n[2] Creating batch with MICRO-COLLATE pipeline...")
        set_seed(args.seed + test_idx + 1000)  # Same seed for both
        batch2 = create_micro_batch(dataset, indices, args.micro_batch_size)
        print(f"    Image shape: {batch2[0].shape}, Label shape: {batch2[1].shape}")

        # Verify
        if verify_batches(batch1, batch2, f"Test {test_idx + 1}"):
            print(f"\n[PASSED] Test {test_idx + 1}")
        else:
            print(f"\n[FAILED] Test {test_idx + 1}")
            all_passed = False

    # Final result
    print(f"\n{'#'*60}")
    if all_passed:
        print("# ALL TESTS PASSED!")
        print("# Micro-Collate pipeline produces IDENTICAL output")
        print("# to the original pipeline.")
    else:
        print("# SOME TESTS FAILED!")
        print("# Please investigate the differences.")
    print(f"{'#'*60}\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
