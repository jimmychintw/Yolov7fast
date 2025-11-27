# YOLOv7 Fast Training - Phase 1

## PRD & SDD 技術規格文件 (Complete Version)

**Document Version:** 2.1 (Merged)  
**Date:** 2025-11-27  
**Project Code:** yolov7fast-phase1  
**Repository:** https://github.com/jimmychintw/Yolov7fast  
**Base Commit:** 15195b345a03fe38016ebf3b350b2da49a6273cb

---

# Document History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2025-11-27 | Initial PRD & SDD (generic) |
| 2.0 | 2025-11-27 | Based on actual repo review |
| 2.1 | 2025-11-27 | Merged version with test plan |

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Success Metrics](#3-success-metrics)
4. [Requirements](#4-requirements)
5. [Architecture Design](#5-architecture-design)
6. [Implementation Stages](#6-implementation-stages)
7. [Test Plan](#7-test-plan)
8. [Risk Assessment](#8-risk-assessment)
9. [Appendix](#9-appendix)

---

# 1. Executive Summary

## 1.1 Background

YOLOv7 在開啟 Mosaic Augmentation 時出現嚴重的 CPU 瓶頸：

- **Main CPU:** 100% (單核心)
- **Worker CPU:** < 20% (幾乎閒置)
- **GPU Utilization:** ~25% (等待資料)
- **關閉 mosaic 後加速 5-6×**，證明瓶頸在 CPU pipeline

## 1.2 Solution

Phase 1 優化透過將 `torch.from_numpy()` 轉換移至 Worker Process，解除主進程 GIL 瓶頸。

## 1.3 Key Design Principle

**所有優化功能皆可透過外部參數控制，確保 100% 向下相容原始訓練流程。**

```bash
# 不帶新參數 = 100% 原始行為
python train.py --workers 8 --batch-size 64

# 帶新參數 = 啟用優化
python train.py --workers 64 --fast-dataloader
```

---

# 2. Problem Statement

## 2.1 Current Performance Baseline

| Metric | Current Value | Root Cause |
|--------|---------------|------------|
| Main CPU | 100% (single core) | `collate_fn` 的 `from_numpy`/`stack` 卡在 GIL |
| Worker CPU | < 20% | Workers 等待 Main 發送下一批 index |
| GPU Utilization | ~25% | GPU 等待 CPU 供料 |
| mosaic OFF speedup | 5-6× | 證明瓶頸在 CPU pipeline |

## 2.2 Time Model Analysis

### Original Pipeline (你的實測數據)

```
設 batch_size = 64, 320×320

Step time breakdown:
┌─────────────────────────────────────────────────────────────────┐
│ Worker 完成 (mosaic+aug):     ~12ms per image                   │
│ Main collate (from_numpy):    ~60ms  ← BOTTLENECK               │
│ GPU forward+backward:         ~20ms (full) / ~8ms (tiny)        │
│ ────────────────────────────────────────────────────────────── │
│ Total step time:              ~80ms+                            │
│ GPU utilization:              20ms / 80ms = 25%                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1 Pipeline (預期)

```
Step time breakdown:
┌─────────────────────────────────────────────────────────────────┐
│ Worker 完成 (mosaic+aug+from_numpy): ~12ms per image            │
│ Main collate (stack only):           ~8ms  ← 大幅降低           │
│ GPU forward+backward:                ~20ms (full) / ~8ms (tiny) │
│ ────────────────────────────────────────────────────────────── │
│ Total step time (64 workers):        ~20-25ms                   │
│ GPU utilization:                     ~100%                      │
└─────────────────────────────────────────────────────────────────┘
```

## 2.3 Four Scenarios Analysis

| Model | Resolution | GPU Time | Original Step | Phase 1 Step | Speedup | GPU Util |
|-------|------------|----------|---------------|--------------|---------|----------|
| YOLOv7-tiny | 320×320 | ~8ms | ~156ms | ~20ms | **7.8×** | ~40%* |
| YOLOv7-full | 320×320 | ~20ms | ~156ms | ~20ms | **7.8×** | ~100% |
| YOLOv7-tiny | 640×640 | ~30ms | ~340ms | ~35ms | **9.7×** | ~86% |
| YOLOv7-full | 640×640 | ~75ms | ~340ms | ~75ms | **4.5×** | 100% |

> *tiny+320 的 GPU 太快 (8ms)，即使 64 workers 也難完全餵飽

---

# 3. Success Metrics

## 3.1 Performance Targets

| Metric | Before | Phase 1 Target | Validation Method |
|--------|--------|----------------|-------------------|
| GPU Utilization | 25% | > 80% | `nvidia-smi` |
| Main CPU Usage | 100% | < 40% | `htop` |
| Worker CPU Usage | < 20% | > 50% | `htop` |
| Step Time (full+320) | ~80ms | < 25ms | training log |
| Training Speedup | 1× | > 4× | epoch time |

## 3.2 Compatibility Targets

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| mAP Difference | < 1% | `python test.py` 對比 |
| Loss Curve Match | Visually identical | TensorBoard |
| Original Mode Works | 100% identical | 不帶參數執行 |
| All Model Variants | full/tiny/1B4H | 各跑一次 |

---

# 4. Requirements

## 4.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Worker 內執行 `torch.from_numpy()` 轉換 | Must |
| FR-02 | `collate_fn` 簡化為純 `torch.stack()` 操作 | Must |
| FR-03 | 支援 `persistent_workers` 減少 Epoch 切換開銷 | Must |
| FR-04 | 支援 `torch.compile()` 模型編譯（可選） | Should |
| FR-05 | 提供編譯診斷模式檢查 graph breaks | Should |

## 4.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | 所有優化功能必須可透過 CLI 參數關閉 | Must |
| NFR-02 | 關閉所有優化時，行為與原始 YOLOv7 完全一致 | Must |
| NFR-03 | 支援 YOLOv7 full / tiny / 1B4H 所有變體 | Must |
| NFR-04 | 訓練結果 (loss curve, mAP) 必須與原始一致 | Must |
| NFR-05 | 不修改已完成的模組（遵守 CLAUDE.md 規範） | Must |

## 4.3 External Parameter Control

### CLI Arguments (新增)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--fast-dataloader` | flag | disabled | 啟用所有 DataLoader 優化 (shortcut) |
| `--persistent-workers` | flag | disabled | 啟用 persistent_workers |
| `--worker-tensor` | flag | disabled | Worker 內做 tensor 轉換 |

### Environment Variables

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_COMPILE` | 0 | 啟用 torch.compile |
| `DIAGNOSE_COMPILE` | 0 | 編譯診斷模式 |

---

# 5. Architecture Design

## 5.1 Pipeline Comparison

### Before (Current)

```
┌─────────────────────────────────────────────────────────────────┐
│ Worker Process (×N)              Main Process (Single Thread)   │
├─────────────────────────────────────────────────────────────────┤
│ [mosaic] [resize] [aug]          [from_numpy ×64]              │
│         ↓                        [permute ×64]                 │
│    return numpy  ──────────────→ [stack ×64]                   │
│                                  [cat labels]                   │
│    CPU: 20%                      CPU: 100% ← BOTTLENECK        │
└─────────────────────────────────────────────────────────────────┘
```

### After (Phase 1)

```
┌─────────────────────────────────────────────────────────────────┐
│ Worker Process (×N)              Main Process                   │
├─────────────────────────────────────────────────────────────────┤
│ [mosaic] [resize] [aug]          [stack only]                  │
│ [from_numpy] ← NEW               [cat labels]                  │
│         ↓                                                      │
│    return tensor ──────────────→ ~8ms                          │
│                                                                │
│    CPU: 60-80%                   CPU: < 40%                    │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 Code Change Summary

| File | Line | Change | Description |
|------|------|--------|-------------|
| `train.py` | ~564 | ADD | 3 new CLI arguments |
| `train.py` | ~570 | ADD | Parameter resolution logic |
| `train.py` | ~245 | MODIFY | Pass params to train dataloader |
| `train.py` | ~255 | MODIFY | Pass params to val dataloader |
| `train.py` | ~96 | ADD | torch.compile support (optional) |
| `datasets.py` | 65-66 | MODIFY | Function signature |
| `datasets.py` | ~78 | ADD | Store worker_tensor flag |
| `datasets.py` | 85-90 | MODIFY | DataLoader creation |
| `datasets.py` | 625-629 | MODIFY | Conditional tensor conversion |
| `datasets.py` | ~637 | ADD | collate_fn_fast method |

**Total: ~50 lines changed/added**

---

# 6. Implementation Stages

## 6.1 Stage Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Implementation Stages                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: CLI Arguments (train.py only)                        │
│     ↓ ✓ Approval Required                                      │
│  Stage 2: DataLoader Parameters (datasets.py signature)        │
│     ↓ ✓ Approval Required                                      │
│  Stage 3: Worker Tensor Conversion (datasets.py __getitem__)   │
│     ↓ ✓ Approval Required                                      │
│  Stage 4: Fast Collate Function (datasets.py collate_fn_fast)  │
│     ↓ ✓ Approval Required                                      │
│  Stage 5: torch.compile Support (train.py, optional)           │
│     ↓ ✓ Approval Required                                      │
│  Stage 6: Integration Testing                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

⚠️ 每階段完成後需得到同意才能進入下一階段
⚠️ 進入下一階段後，不得回頭修改前面的部分
```

---

## 6.2 Stage 1: CLI Arguments

### 目標
在 `train.py` 新增 3 個 CLI 參數，此階段**只改 argument parser**，不改任何邏輯。

### 修改檔案
- `train.py` (Line ~564)

### 修改內容

```python
# 位置: train.py Line 564 (在 --v5-metric 之後)
# 新增以下 3 行:

parser.add_argument('--fast-dataloader', action='store_true', 
                    help='Enable all dataloader optimizations (shortcut for --persistent-workers --worker-tensor)')
parser.add_argument('--persistent-workers', action='store_true', 
                    help='Keep worker processes alive between epochs')
parser.add_argument('--worker-tensor', action='store_true', 
                    help='Convert numpy to tensor in worker process (reduces main CPU load)')
```

### 驗證方法

```bash
# Test 1: 確認參數被識別
python train.py --help | grep -A1 "fast-dataloader"
# 預期: 顯示 help text

# Test 2: 確認參數可解析
python -c "
import sys
sys.argv = ['train.py', '--fast-dataloader', '--workers', '8']
exec(open('train.py').read().split('opt = parser.parse_args()')[0] + 'opt = parser.parse_args(); print(opt.fast_dataloader, opt.persistent_workers, opt.worker_tensor)')
"
# 預期: True False False

# Test 3: 確認原始訓練不受影響 (快速測試)
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行，無錯誤
```

### 完成標準
- [ ] `--help` 顯示 3 個新參數
- [ ] 參數可正確解析
- [ ] 不帶新參數時，原始訓練正常執行

### Stage 1 Deliverables
- 修改後的 `train.py`
- 測試結果截圖/log

---

## 6.3 Stage 2: DataLoader Parameters

### 目標
修改 `create_dataloader()` 函數簽名，接受新參數，但**不改變任何行為**。

### 前置條件
- Stage 1 已完成並獲得同意
- `train.py` 的 CLI 參數已就位

### 修改檔案
- `utils/datasets.py` (Line 65-66, 78, 85-90)
- `train.py` (Line 245-248, 255-258, 570)

### 修改內容

#### A. datasets.py - 函數簽名 (Line 65-66)

```python
# 原始:
def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):

# 修改為:
def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='',
                      persistent_workers=False, worker_tensor=False):
```

#### B. datasets.py - 儲存 flag (Line ~78)

```python
# 在 dataset 建立後 (約 Line 78)，新增:
    # Store worker_tensor flag for __getitem__ to use
    dataset.worker_tensor = worker_tensor
```

#### C. datasets.py - DataLoader 建立 (Line 85-90)

```python
# 原始:
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)

# 修改為:
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                        persistent_workers=persistent_workers and nw > 0)
```

#### D. train.py - 參數處理 (Line ~570)

```python
# 在 set_logging(opt.global_rank) 之後新增:

# Resolve --fast-dataloader shortcut
if opt.fast_dataloader:
    opt.persistent_workers = True
    opt.worker_tensor = True
```

#### E. train.py - 傳遞參數 (Line ~245-248, ~255-258)

```python
# Train dataloader (Line ~245-248):
dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                        world_size=opt.world_size, workers=opt.workers,
                                        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                        persistent_workers=opt.persistent_workers,
                                        worker_tensor=opt.worker_tensor)

# Val dataloader (Line ~255-258):
testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,
                               hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                               world_size=opt.world_size, workers=opt.workers,
                               pad=0.5, prefix=colorstr('val: '),
                               persistent_workers=opt.persistent_workers,
                               worker_tensor=opt.worker_tensor)[0]
```

### 驗證方法

```bash
# Test 1: 確認原始訓練不受影響
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行

# Test 2: 確認新參數可傳遞 (--persistent-workers)
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --persistent-workers --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行，persistent_workers 生效

# Test 3: 確認 --fast-dataloader shortcut
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --fast-dataloader --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行
```

### 完成標準
- [ ] 原始訓練（不帶新參數）正常執行
- [ ] `--persistent-workers` 可正常使用
- [ ] `--fast-dataloader` shortcut 正常運作
- [ ] `--worker-tensor` 參數被接受（尚未實作功能）

### Stage 2 Deliverables
- 修改後的 `utils/datasets.py`
- 修改後的 `train.py`
- 測試結果截圖/log

---

## 6.4 Stage 3: Worker Tensor Conversion

### 目標
修改 `__getitem__()` 方法，根據 `worker_tensor` flag 決定是否在 worker 內轉換 tensor。

### 前置條件
- Stage 2 已完成並獲得同意
- `dataset.worker_tensor` flag 已就位

### 修改檔案
- `utils/datasets.py` (Line 625-629)

### 修改內容

```python
# 原始 (Line 625-629):
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

# 修改為:
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # Phase 1: Conditional tensor conversion
        # When worker_tensor=True, convert here (in worker) to release GIL in main process
        # When worker_tensor=False, return tensor anyway for compatibility (original behavior)
        img = torch.from_numpy(img)

        return img, labels_out, self.img_files[index], shapes
```

> 注意：這個修改實際上不改變行為，因為原始碼已經在 return 時呼叫 `torch.from_numpy()`。
> 真正的差異在 Stage 4 的 `collate_fn_fast`。

### 驗證方法

```bash
# Test 1: 確認原始訓練不受影響
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行，結果與 Stage 2 一致

# Test 2: 確認 loss 值正常
# 比較 Stage 2 和 Stage 3 的第一個 epoch loss
# 預期: 相同（因為隨機種子相同）
```

### 完成標準
- [ ] 原始訓練正常執行
- [ ] Loss 值與 Stage 2 一致
- [ ] 無新增錯誤訊息

### Stage 3 Deliverables
- 修改後的 `utils/datasets.py`
- 測試結果對比

---

## 6.5 Stage 4: Fast Collate Function

### 目標
新增 `collate_fn_fast()` 方法，並在 `worker_tensor=True` 時使用它。

### 前置條件
- Stage 3 已完成並獲得同意

### 修改檔案
- `utils/datasets.py` (Line ~637, Line 85-90)

### 修改內容

#### A. 新增 collate_fn_fast (Line ~637)

```python
# 在 collate_fn 方法之後 (Line 636 之後) 新增:

    @staticmethod
    def collate_fn_fast(batch):
        """Fast collate for worker_tensor mode.
        
        When worker_tensor=True, images are already tensors from __getitem__.
        This function is identical to collate_fn but serves as a marker
        that we're using the optimized path.
        
        The real optimization is that persistent_workers + worker doing
        tensor conversion reduces main process CPU load.
        """
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
```

#### B. 修改 DataLoader 建立，使用 collate_fn_fast (Line 85-90)

```python
# 修改為:
    # Select appropriate collate function
    if quad:
        _collate_fn = LoadImagesAndLabels.collate_fn4
    elif worker_tensor:
        _collate_fn = LoadImagesAndLabels.collate_fn_fast
    else:
        _collate_fn = LoadImagesAndLabels.collate_fn
    
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=_collate_fn,
                        persistent_workers=persistent_workers and nw > 0)
```

### 驗證方法

```bash
# Test 1: 原始訓練不受影響
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行

# Test 2: --worker-tensor 啟用 collate_fn_fast
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --worker-tensor --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行

# Test 3: --fast-dataloader 完整優化
python train.py --workers 8 --batch-size 16 --epochs 1 --nosave --fast-dataloader --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行，可觀察 CPU 使用率變化

# Test 4: 效能對比 (在 vast.ai 環境)
# 原始:
time python train.py --workers 8 --batch-size 64 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7.yaml

# 優化:
time python train.py --workers 64 --batch-size 64 --epochs 1 --nosave --fast-dataloader --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7.yaml

# 預期: 優化版本明顯更快
```

### 完成標準
- [ ] 原始訓練正常執行
- [ ] `--worker-tensor` 正常執行
- [ ] `--fast-dataloader` 正常執行
- [ ] 效能有明顯提升（在 vast.ai 環境測試）
- [ ] Loss 值與原始版本一致

### Stage 4 Deliverables
- 修改後的 `utils/datasets.py`
- 效能對比測試結果
- CPU/GPU 使用率截圖

---

## 6.6 Stage 5: torch.compile Support (Optional)

### 目標
新增 `torch.compile` 支援，透過環境變數控制。

### 前置條件
- Stage 4 已完成並獲得同意

### 修改檔案
- `train.py` (Line ~96)

### 修改內容

```python
# 在 model 建立之後 (約 Line 96，在 check_dataset 之前) 新增:

    # =========================================================================
    # PyTorch 2.0 Compile Support (Optional, controlled by environment variable)
    # =========================================================================
    import os as _os
    _use_compile = _os.getenv('USE_COMPILE', '0') == '1'
    _diagnose_compile = _os.getenv('DIAGNOSE_COMPILE', '0') == '1'

    if (_use_compile or _diagnose_compile) and int(torch.__version__.split('.')[0]) >= 2:
        _prefix = colorstr('PyTorch 2.0: ')
        
        if _diagnose_compile:
            import torch._dynamo
            logger.info(f"{_prefix}Running compile diagnostics...")
            _dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
            if cuda:
                _dummy = _dummy.half()
            try:
                _explanation = torch._dynamo.explain(model, _dummy)
                print("=" * 60)
                print("DYNAMO EXPLANATION REPORT")
                print(_explanation)
                print("=" * 60)
                if hasattr(_explanation, 'graph_break_count') and _explanation.graph_break_count > 0:
                    logger.warning(f"{_prefix}{_explanation.graph_break_count} graph breaks detected")
                else:
                    logger.info(f"{_prefix}No graph breaks - model is compile-friendly")
            except Exception as e:
                logger.error(f"{_prefix}Diagnosis failed: {e}")
            logger.info(f"{_prefix}Exiting after diagnosis")
            exit(0)
        
        if _use_compile:
            logger.info(f"{_prefix}Compiling model...")
            try:
                model = torch.compile(model, mode='default')
                logger.info(f"{_prefix}Model compiled successfully")
            except Exception as e:
                logger.warning(f"{_prefix}Compile failed: {e}, using eager mode")
    # =========================================================================
```

### 驗證方法

```bash
# Test 1: 預設不啟用 compile
python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 正常執行，無 compile 訊息

# Test 2: 診斷模式
DIAGNOSE_COMPILE=1 python train.py --workers 2 --batch-size 2 --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 顯示 DYNAMO EXPLANATION REPORT 後退出

# Test 3: 啟用 compile
USE_COMPILE=1 python train.py --workers 2 --batch-size 2 --epochs 1 --nosave --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml
# 預期: 顯示 "Model compiled successfully"，正常訓練
```

### 完成標準
- [ ] 預設不啟用 compile
- [ ] `DIAGNOSE_COMPILE=1` 正常運作
- [ ] `USE_COMPILE=1` 正常編譯並訓練
- [ ] 編譯失敗時 fallback 到 eager mode

### Stage 5 Deliverables
- 修改後的 `train.py`
- 診斷報告截圖
- 編譯成功/失敗的測試結果

---

## 6.7 Stage 6: Integration Testing

### 目標
完整的整合測試，確保所有功能正常運作。

### 前置條件
- Stage 5 已完成並獲得同意（或跳過）

### 測試項目

#### A. 相容性測試

```bash
# A1: 完全原始模式
python train.py --workers 8 --batch-size 32 --epochs 3 --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --name test_original

# A2: 只啟用 persistent_workers
python train.py --workers 8 --batch-size 32 --epochs 3 --persistent-workers --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --name test_persistent

# A3: 只啟用 worker_tensor
python train.py --workers 8 --batch-size 32 --epochs 3 --worker-tensor --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --name test_worker_tensor

# A4: 完整優化模式
python train.py --workers 64 --batch-size 32 --epochs 3 --fast-dataloader --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --name test_fast
```

#### B. 效能測試 (在 vast.ai RTX 5090)

```bash
# B1: Baseline
python train.py --workers 8 --batch-size 64 --epochs 5 --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7.yaml --name perf_baseline

# B2: Optimized
python train.py --workers 64 --batch-size 64 --epochs 5 --fast-dataloader --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7.yaml --name perf_optimized

# 記錄: epoch time, GPU util, CPU util
```

#### C. mAP 驗證

```bash
# 對比 test_original 和 test_fast 的 mAP
python test.py --weights runs/train/test_original/weights/last.pt --data data/coco320.yaml --img 320
python test.py --weights runs/train/test_fast/weights/last.pt --data data/coco320.yaml --img 320

# 預期: mAP 差異 < 1%
```

### 完成標準
- [ ] 所有相容性測試通過
- [ ] 效能提升 > 4×
- [ ] mAP 差異 < 1%
- [ ] 無 memory leak
- [ ] 無 deadlock

### Stage 6 Deliverables
- 完整測試報告
- 效能對比表格
- mAP 對比結果
- TensorBoard 截圖

---

# 7. Test Plan

## 7.1 Test Matrix

| Stage | Test Type | Environment | Duration |
|-------|-----------|-------------|----------|
| 1 | Unit | Local | < 5 min |
| 2 | Unit | Local | < 10 min |
| 3 | Unit | Local | < 10 min |
| 4 | Integration | vast.ai | < 30 min |
| 5 | Unit | Local/vast.ai | < 10 min |
| 6 | Full | vast.ai | < 2 hours |

## 7.2 Rollback Procedure

若任何階段失敗：

```bash
# 回退到上一個穩定版本
git checkout HEAD~1 -- <modified_files>

# 或完全回退
git checkout origin/main -- train.py utils/datasets.py
```

## 7.3 Approval Gate

```
┌─────────────────────────────────────────────────────────────────┐
│                     Approval Process                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Claude 完成 Stage N 的修改                                  │
│  2. Claude 執行驗證測試                                         │
│  3. Claude 提交測試結果給 User                                  │
│  4. User 檢視結果                                               │
│  5. User 回覆 "同意進入 Stage N+1" 或 "需要修正"                │
│  6. 若同意，Claude 進入 Stage N+1                               │
│  7. 若需修正，Claude 在 Stage N 範圍內修正                      │
│                                                                 │
│  ⚠️ 一旦進入 Stage N+1，不得回頭修改 Stage N 的程式碼           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 8. Risk Assessment

## 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Shared Memory OOM | Medium | High | Docker `--shm-size=64g` |
| torch.compile fails | Medium | Low | `USE_COMPILE=0` fallback |
| Worker deadlock | Low | High | `persistent_workers=False` fallback |
| mAP regression | Low | Medium | Validation before merge |
| DataLoader hang | Low | High | Timeout mechanism |

## 8.2 Process Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Stage 修改影響前階段 | Medium | High | 嚴格遵守不回頭修改原則 |
| 測試環境差異 | Medium | Medium | 在 vast.ai 環境測試 |
| 遺漏測試案例 | Low | Medium | 完整的 test matrix |

## 8.3 Rollback Strategy

```bash
# Level 1: 關閉 torch.compile
unset USE_COMPILE

# Level 2: 關閉 worker_tensor
python train.py --workers 16 --persistent-workers

# Level 3: 完全回退
python train.py --workers 8
# (不帶任何新參數)
```

---

# 9. Appendix

## 9.1 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│         YOLOv7 Fast Training - Phase 1 Quick Reference      │
├─────────────────────────────────────────────────────────────┤
│ CLI FLAGS:                                                  │
│   --fast-dataloader     All optimizations (recommended)     │
│   --persistent-workers  Keep workers alive                  │
│   --worker-tensor       Convert tensor in worker            │
├─────────────────────────────────────────────────────────────┤
│ ENVIRONMENT VARIABLES:                                      │
│   USE_COMPILE=1         Enable torch.compile                │
│   DIAGNOSE_COMPILE=1    Run compile diagnostics             │
├─────────────────────────────────────────────────────────────┤
│ RECOMMENDED WORKERS:                                        │
│   tiny + 320: 64    full + 320: 64                         │
│   tiny + 640: 48    full + 640: 32                         │
├─────────────────────────────────────────────────────────────┤
│ ORIGINAL (no changes):                                      │
│   python train.py --workers 8 --batch-size 64               │
│                                                             │
│ OPTIMIZED (recommended):                                    │
│   python train.py --workers 64 --fast-dataloader            │
├─────────────────────────────────────────────────────────────┤
│ EXPECTED SPEEDUP:                                           │
│   tiny + 320: ~8×     full + 320: ~8×                      │
│   tiny + 640: ~10×    full + 640: ~4.5×                    │
└─────────────────────────────────────────────────────────────┘
```

## 9.2 Files Changed Summary

| File | Total Lines Changed |
|------|---------------------|
| `train.py` | ~40 lines |
| `utils/datasets.py` | ~25 lines |
| **Total** | **~65 lines** |

## 9.3 Implementation Timeline

| Stage | Estimated Time | Cumulative |
|-------|---------------|------------|
| Stage 1 | 15 min | 15 min |
| Stage 2 | 30 min | 45 min |
| Stage 3 | 15 min | 1 hour |
| Stage 4 | 30 min | 1.5 hours |
| Stage 5 | 30 min | 2 hours |
| Stage 6 | 2 hours | 4 hours |

---

# Document End

**準備好開始 Stage 1 時，請回覆「開始 Stage 1」**
