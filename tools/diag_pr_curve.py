#!/usr/bin/env python3
"""
PR Curve 診斷工具

解析 results.txt，輸出每 epoch 的 P/R/mAP50 CSV 與曲線圖，標註 best epoch。

Usage:
    python tools/diag_pr_curve.py \
        --results runs/train/1b4h_stage2_late_backbone/results.txt \
        --out runs/diag/pr_curve

Output:
    - pr_metrics.csv: 每 epoch 的 P/R/mAP50/mAP50-95
    - pr_curve.png: P/R/mAP 曲線圖
    - summary.txt: 摘要報告
"""

import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def parse_results(filepath):
    """
    解析 YOLOv7 results.txt

    格式: epoch/total  gpu_mem  box  obj  cls  total  labels  img_size  P  R  mAP50  mAP50-95  ...
    """
    data = {
        'epoch': [],
        'box_loss': [],
        'obj_loss': [],
        'cls_loss': [],
        'total_loss': [],
        'precision': [],
        'recall': [],
        'mAP50': [],
        'mAP50_95': []
    }

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 12:
                try:
                    epoch_str = parts[0].split('/')[0]
                    data['epoch'].append(int(epoch_str))
                    data['box_loss'].append(float(parts[2]))
                    data['obj_loss'].append(float(parts[3]))
                    data['cls_loss'].append(float(parts[4]))
                    data['total_loss'].append(float(parts[5]))
                    data['precision'].append(float(parts[8]))
                    data['recall'].append(float(parts[9]))
                    data['mAP50'].append(float(parts[10]))
                    data['mAP50_95'].append(float(parts[11]))
                except (ValueError, IndexError):
                    continue

    return {k: np.array(v) for k, v in data.items()}


def find_best_epoch(data, metric='mAP50'):
    """找出指定 metric 最佳的 epoch"""
    idx = np.argmax(data[metric])
    return {
        'epoch': int(data['epoch'][idx]),
        'index': idx,
        'value': data[metric][idx]
    }


def export_csv(data, output_path):
    """輸出 CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'precision', 'recall', 'mAP50', 'mAP50_95',
                        'box_loss', 'obj_loss', 'cls_loss', 'total_loss'])

        for i in range(len(data['epoch'])):
            writer.writerow([
                int(data['epoch'][i]),
                f"{data['precision'][i]:.6f}",
                f"{data['recall'][i]:.6f}",
                f"{data['mAP50'][i]:.6f}",
                f"{data['mAP50_95'][i]:.6f}",
                f"{data['box_loss'][i]:.6f}",
                f"{data['obj_loss'][i]:.6f}",
                f"{data['cls_loss'][i]:.6f}",
                f"{data['total_loss'][i]:.6f}"
            ])

    print(f"CSV exported: {output_path}")


def plot_pr_curves(data, best_info, output_path):
    """繪製 P/R/mAP 曲線"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = data['epoch']
    best_epoch = best_info['epoch']
    best_idx = best_info['index']

    # 1. mAP@0.5
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['mAP50'], 'b-', linewidth=1.5, label='mAP@0.5')
    ax1.scatter([best_epoch], [data['mAP50'][best_idx]], color='red', s=100,
                zorder=5, label=f'Best: {data["mAP50"][best_idx]:.4f} @ ep{best_epoch}')
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('mAP@0.5')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. Precision & Recall
    ax2 = axes[0, 1]
    ax2.plot(epochs, data['precision'], 'g-', linewidth=1.5, label='Precision')
    ax2.plot(epochs, data['recall'], 'orange', linewidth=1.5, label='Recall')
    ax2.scatter([best_epoch], [data['precision'][best_idx]], color='green', s=80, zorder=5)
    ax2.scatter([best_epoch], [data['recall'][best_idx]], color='orange', s=80, zorder=5)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best epoch: {best_epoch}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('Precision & Recall')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 3. P-R Curve (scatter)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(data['recall'], data['precision'], c=epochs, cmap='viridis',
                          s=20, alpha=0.7)
    ax3.scatter([data['recall'][best_idx]], [data['precision'][best_idx]],
                color='red', s=150, marker='*', zorder=5,
                label=f'Best: P={data["precision"][best_idx]:.3f}, R={data["recall"][best_idx]:.3f}')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('P-R Trajectory')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Epoch')

    # 4. Loss curves
    ax4 = axes[1, 1]
    ax4.plot(epochs, data['obj_loss'], 'r-', linewidth=1.5, label='Obj Loss')
    ax4.plot(epochs, data['box_loss'], 'b-', linewidth=1.5, label='Box Loss')
    ax4.plot(epochs, data['cls_loss'], 'g-', linewidth=1.5, label='Cls Loss')
    ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Curves')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Training Diagnostics (Best mAP@0.5 at Epoch {best_epoch})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {output_path}")


def write_summary(data, best_info, output_path):
    """輸出摘要報告"""
    best_idx = best_info['index']

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PR Curve Diagnostic Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Epochs: {len(data['epoch'])}\n")
        f.write(f"Best Epoch (by mAP@0.5): {best_info['epoch']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Best Epoch Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  mAP@0.5:     {data['mAP50'][best_idx]:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {data['mAP50_95'][best_idx]:.4f}\n")
        f.write(f"  Precision:   {data['precision'][best_idx]:.4f}\n")
        f.write(f"  Recall:      {data['recall'][best_idx]:.4f}\n")
        f.write(f"  Obj Loss:    {data['obj_loss'][best_idx]:.4f}\n")
        f.write(f"  Box Loss:    {data['box_loss'][best_idx]:.4f}\n")
        f.write(f"  Cls Loss:    {data['cls_loss'][best_idx]:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Final Epoch Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  mAP@0.5:     {data['mAP50'][-1]:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {data['mAP50_95'][-1]:.4f}\n")
        f.write(f"  Precision:   {data['precision'][-1]:.4f}\n")
        f.write(f"  Recall:      {data['recall'][-1]:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  mAP@0.5:  min={data['mAP50'].min():.4f}, max={data['mAP50'].max():.4f}, "
                f"mean={data['mAP50'].mean():.4f}\n")
        f.write(f"  Precision: min={data['precision'].min():.4f}, max={data['precision'].max():.4f}, "
                f"mean={data['precision'].mean():.4f}\n")
        f.write(f"  Recall:    min={data['recall'].min():.4f}, max={data['recall'].max():.4f}, "
                f"mean={data['recall'].mean():.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Summary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PR Curve Diagnostic Tool')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results.txt')
    parser.add_argument('--out', type=str, default='runs/diag/pr_curve',
                        help='Output directory')
    parser.add_argument('--metric', type=str, default='mAP50',
                        choices=['mAP50', 'mAP50_95', 'precision', 'recall'],
                        help='Metric to find best epoch')

    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.out, exist_ok=True)

    # 解析結果
    print(f"Parsing: {args.results}")
    data = parse_results(args.results)

    if len(data['epoch']) == 0:
        print("Error: No valid data found in results file")
        return

    print(f"Found {len(data['epoch'])} epochs")

    # 找出最佳 epoch
    best_info = find_best_epoch(data, args.metric)
    print(f"Best {args.metric}: {best_info['value']:.4f} at epoch {best_info['epoch']}")

    # 輸出
    export_csv(data, os.path.join(args.out, 'pr_metrics.csv'))
    plot_pr_curves(data, best_info, os.path.join(args.out, 'pr_curve.png'))
    write_summary(data, best_info, os.path.join(args.out, 'summary.txt'))

    print(f"\nDiagnostics complete. Output: {args.out}")


if __name__ == '__main__':
    main()
