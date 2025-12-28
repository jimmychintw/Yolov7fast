#!/usr/bin/env python3
"""
Confidence-FP 診斷工具

讀取 test.py 產生的預測結果（含 conf），與 val GT 做 IoU=0.5 配對，
計算 TP/FP，輸出 confidence bins 的 TP/FP histogram。
支援用 head_config.yaml 把 class ids 聚合成 per-head 統計。

Usage:
    # 先用 test.py 產生帶 conf 的預測結果
    python test.py --weights xxx.pt --data data/coco320.yaml \
        --save-txt --save-conf --project runs/diag --name stage2_best

    # 然後執行此診斷
    python tools/diag_conf_fp.py \
        --pred-dir runs/diag/stage2_best/labels \
        --gt-dir /path/to/val/labels \
        --head-config data/coco_320_1b4h_anticonfusion.yaml \
        --out runs/diag/conf_fp

Output:
    - conf_fp_stats.csv: 每個 conf bin 的 TP/FP 統計
    - conf_fp_histogram.png: TP/FP histogram
    - per_head_stats.csv: 每個 head 的統計 (若有 head_config)
    - per_head_histogram.png: per-head histogram
    - summary.txt: 摘要報告
"""

import argparse
import os
import csv
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def load_predictions(pred_dir):
    """
    載入預測結果

    格式: class_id x_center y_center width height conf
    """
    predictions = {}

    txt_files = glob.glob(os.path.join(pred_dir, '*.txt'))

    for txt_file in txt_files:
        img_name = os.path.splitext(os.path.basename(txt_file))[0]
        preds = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:  # class, x, y, w, h, conf
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        conf = float(parts[5])
                        preds.append({
                            'class': cls_id,
                            'bbox': [x, y, w, h],
                            'conf': conf
                        })
                    except ValueError:
                        continue

        predictions[img_name] = preds

    return predictions


def load_ground_truth(gt_dir):
    """
    載入 Ground Truth

    格式: class_id x_center y_center width height
    """
    ground_truths = {}

    txt_files = glob.glob(os.path.join(gt_dir, '*.txt'))

    for txt_file in txt_files:
        img_name = os.path.splitext(os.path.basename(txt_file))[0]
        gts = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        gts.append({
                            'class': cls_id,
                            'bbox': [x, y, w, h],
                            'matched': False
                        })
                    except ValueError:
                        continue

        ground_truths[img_name] = gts

    return ground_truths


def load_head_config(config_path):
    """載入 head config，取得 class 到 head 的對應"""
    if not config_path or not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 建立 class_id -> head_id 對應
    class_to_head = {}
    head_names = []

    # 支援兩種格式: 'heads' (list) 或 'head_assignments' (dict)
    if 'heads' in config and isinstance(config['heads'], list):
        for head_id, head_info in enumerate(config['heads']):
            head_name = head_info.get('name', f'head_{head_id}')
            head_names.append(head_name)

            if 'classes' in head_info:
                for cls_id in head_info['classes']:
                    class_to_head[cls_id] = head_id

    elif 'head_assignments' in config:
        # 格式: head_assignments: {head_0: {name, classes}, head_1: {...}, ...}
        head_assignments = config['head_assignments']
        # 排序確保順序正確
        sorted_keys = sorted(head_assignments.keys(), key=lambda x: int(x.split('_')[1]))

        for head_id, key in enumerate(sorted_keys):
            head_info = head_assignments[key]
            head_name = head_info.get('name', key)
            head_names.append(head_name)

            if 'classes' in head_info:
                for cls_id in head_info['classes']:
                    class_to_head[cls_id] = head_id

    if not head_names:
        return None

    return {
        'class_to_head': class_to_head,
        'head_names': head_names,
        'num_heads': len(head_names)
    }


def compute_iou(box1, box2):
    """
    計算兩個 bbox 的 IoU (normalized xywh format)
    """
    # 轉換為 xyxy
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # 計算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 計算聯集
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def match_predictions(predictions, ground_truths, iou_threshold=0.5):
    """
    配對預測與 GT，計算 TP/FP

    Returns:
        list of dict: 每個預測的配對結果
    """
    results = []

    for img_name, preds in predictions.items():
        # 取得對應的 GT
        gts = ground_truths.get(img_name, [])

        # 重置 GT matched 狀態
        for gt in gts:
            gt['matched'] = False

        # 按 conf 降序排序預測
        preds_sorted = sorted(preds, key=lambda x: x['conf'], reverse=True)

        for pred in preds_sorted:
            pred_class = pred['class']
            pred_bbox = pred['bbox']
            pred_conf = pred['conf']

            # 找最佳匹配的 GT
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt['matched']:
                    continue
                if gt['class'] != pred_class:
                    continue

                iou = compute_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 判斷 TP/FP
            if best_iou >= iou_threshold:
                gts[best_gt_idx]['matched'] = True
                is_tp = True
            else:
                is_tp = False

            results.append({
                'image': img_name,
                'class': pred_class,
                'conf': pred_conf,
                'iou': best_iou,
                'is_tp': is_tp
            })

    return results


def compute_conf_bins(results, num_bins=10):
    """計算 confidence bins 的 TP/FP 統計"""
    bins = np.linspace(0, 1, num_bins + 1)
    bin_stats = []

    for i in range(num_bins):
        bin_min = bins[i]
        bin_max = bins[i + 1]

        bin_results = [r for r in results if bin_min <= r['conf'] < bin_max]

        tp = sum(1 for r in bin_results if r['is_tp'])
        fp = sum(1 for r in bin_results if not r['is_tp'])
        total = len(bin_results)

        precision = tp / total if total > 0 else 0

        bin_stats.append({
            'bin_min': bin_min,
            'bin_max': bin_max,
            'bin_label': f'{bin_min:.1f}-{bin_max:.1f}',
            'tp': tp,
            'fp': fp,
            'total': total,
            'precision': precision
        })

    return bin_stats


def compute_per_head_stats(results, head_config):
    """計算 per-head 統計"""
    if not head_config:
        return None

    class_to_head = head_config['class_to_head']
    head_names = head_config['head_names']
    num_heads = head_config['num_heads']

    head_stats = []

    for head_id in range(num_heads):
        head_results = [r for r in results
                       if class_to_head.get(r['class'], -1) == head_id]

        tp = sum(1 for r in head_results if r['is_tp'])
        fp = sum(1 for r in head_results if not r['is_tp'])
        total = len(head_results)

        precision = tp / total if total > 0 else 0
        avg_conf = np.mean([r['conf'] for r in head_results]) if head_results else 0

        head_stats.append({
            'head_id': head_id,
            'head_name': head_names[head_id],
            'tp': tp,
            'fp': fp,
            'total': total,
            'precision': precision,
            'avg_conf': avg_conf
        })

    return head_stats


def export_conf_csv(bin_stats, output_path):
    """輸出 conf bins CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['conf_bin', 'tp', 'fp', 'total', 'precision'])

        for stat in bin_stats:
            writer.writerow([
                stat['bin_label'],
                stat['tp'],
                stat['fp'],
                stat['total'],
                f"{stat['precision']:.4f}"
            ])

    print(f"CSV exported: {output_path}")


def export_head_csv(head_stats, output_path):
    """輸出 per-head CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['head_id', 'head_name', 'tp', 'fp', 'total', 'precision', 'avg_conf'])

        for stat in head_stats:
            writer.writerow([
                stat['head_id'],
                stat['head_name'],
                stat['tp'],
                stat['fp'],
                stat['total'],
                f"{stat['precision']:.4f}",
                f"{stat['avg_conf']:.4f}"
            ])

    print(f"CSV exported: {output_path}")


def plot_conf_histogram(bin_stats, output_path):
    """繪製 conf bins histogram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [s['bin_label'] for s in bin_stats]
    tp_counts = [s['tp'] for s in bin_stats]
    fp_counts = [s['fp'] for s in bin_stats]
    precisions = [s['precision'] for s in bin_stats]

    x = np.arange(len(labels))
    width = 0.35

    # 左圖: TP/FP 堆疊直方圖
    ax1 = axes[0]
    bars1 = ax1.bar(x, tp_counts, width, label='TP', color='green', alpha=0.8)
    bars2 = ax1.bar(x, fp_counts, width, bottom=tp_counts, label='FP', color='red', alpha=0.8)

    ax1.set_xlabel('Confidence Bin')
    ax1.set_ylabel('Count')
    ax1.set_title('TP/FP Distribution by Confidence')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 右圖: Precision by conf bin
    ax2 = axes[1]
    bars3 = ax2.bar(x, precisions, width * 1.5, color='blue', alpha=0.8)

    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision by Confidence Bin')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # 標註數值
    for bar, prec in zip(bars3, precisions):
        if prec > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{prec:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {output_path}")


def plot_head_histogram(head_stats, output_path):
    """繪製 per-head histogram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [s['head_name'] for s in head_stats]
    tp_counts = [s['tp'] for s in head_stats]
    fp_counts = [s['fp'] for s in head_stats]
    precisions = [s['precision'] for s in head_stats]

    x = np.arange(len(labels))
    width = 0.35

    # 左圖: TP/FP per head
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, tp_counts, width, label='TP', color='green', alpha=0.8)
    bars2 = ax1.bar(x + width/2, fp_counts, width, label='FP', color='red', alpha=0.8)

    ax1.set_xlabel('Head')
    ax1.set_ylabel('Count')
    ax1.set_title('TP/FP by Head')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 右圖: Precision per head
    ax2 = axes[1]
    bars3 = ax2.bar(x, precisions, width * 1.5, color='blue', alpha=0.8)

    ax2.set_xlabel('Head')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision by Head')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # 標註數值
    for bar, prec in zip(bars3, precisions):
        if prec > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{prec:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {output_path}")


def write_summary(results, bin_stats, head_stats, output_path):
    """輸出摘要報告"""
    total_tp = sum(1 for r in results if r['is_tp'])
    total_fp = sum(1 for r in results if not r['is_tp'])
    total_preds = len(results)
    overall_precision = total_tp / total_preds if total_preds > 0 else 0

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Confidence-FP Diagnostic Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("-" * 60 + "\n")
        f.write("Overall Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total Predictions: {total_preds}\n")
        f.write(f"  True Positives:    {total_tp}\n")
        f.write(f"  False Positives:   {total_fp}\n")
        f.write(f"  Overall Precision: {overall_precision:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Confidence Bin Analysis:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Bin':<12} {'TP':<8} {'FP':<8} {'Total':<8} {'Precision':<10}\n")
        for stat in bin_stats:
            f.write(f"{stat['bin_label']:<12} {stat['tp']:<8} {stat['fp']:<8} "
                   f"{stat['total']:<8} {stat['precision']:.4f}\n")

        if head_stats:
            f.write("\n" + "-" * 60 + "\n")
            f.write("Per-Head Analysis:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Head':<15} {'TP':<8} {'FP':<8} {'Total':<8} {'Precision':<10} {'Avg Conf':<10}\n")
            for stat in head_stats:
                f.write(f"{stat['head_name']:<15} {stat['tp']:<8} {stat['fp']:<8} "
                       f"{stat['total']:<8} {stat['precision']:.4f}     {stat['avg_conf']:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Summary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Confidence-FP Diagnostic Tool')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Directory containing prediction .txt files (with conf)')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='Directory containing ground truth .txt files')
    parser.add_argument('--head-config', type=str, default=None,
                        help='Path to head_config.yaml for per-head aggregation')
    parser.add_argument('--out', type=str, default='runs/diag/conf_fp',
                        help='Output directory')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for TP/FP matching')
    parser.add_argument('--num-bins', type=int, default=10,
                        help='Number of confidence bins')

    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.out, exist_ok=True)

    # 載入預測與 GT
    print(f"Loading predictions from: {args.pred_dir}")
    predictions = load_predictions(args.pred_dir)
    print(f"  Found {len(predictions)} images with predictions")

    print(f"Loading ground truth from: {args.gt_dir}")
    ground_truths = load_ground_truth(args.gt_dir)
    print(f"  Found {len(ground_truths)} images with GT")

    # 載入 head config
    head_config = None
    if args.head_config:
        print(f"Loading head config: {args.head_config}")
        head_config = load_head_config(args.head_config)
        if head_config:
            print(f"  Found {head_config['num_heads']} heads: {head_config['head_names']}")

    # 配對計算 TP/FP
    print(f"\nMatching predictions with GT (IoU threshold: {args.iou_threshold})...")
    results = match_predictions(predictions, ground_truths, args.iou_threshold)
    print(f"  Total predictions matched: {len(results)}")

    if len(results) == 0:
        print("Warning: No predictions found. Check your pred-dir and file format.")
        return

    # 計算統計
    bin_stats = compute_conf_bins(results, args.num_bins)
    head_stats = compute_per_head_stats(results, head_config) if head_config else None

    # 輸出
    export_conf_csv(bin_stats, os.path.join(args.out, 'conf_fp_stats.csv'))
    plot_conf_histogram(bin_stats, os.path.join(args.out, 'conf_fp_histogram.png'))

    if head_stats:
        export_head_csv(head_stats, os.path.join(args.out, 'per_head_stats.csv'))
        plot_head_histogram(head_stats, os.path.join(args.out, 'per_head_histogram.png'))

    write_summary(results, bin_stats, head_stats, os.path.join(args.out, 'summary.txt'))

    print(f"\nDiagnostics complete. Output: {args.out}")


if __name__ == '__main__':
    main()
