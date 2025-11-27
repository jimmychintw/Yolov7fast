# Phase 1 DataLoader 優化分析報告

**日期：** 2025-11-27
**目的：** 分析 Stage 4 優化無效的原因，並提出正確的解決方案供專家討論

---

## 1. 問題現象

### 1.1 測試環境
- GPU: RTX 5090 (32GB VRAM)
- PyTorch: 2.8.0+cu128
- 訓練參數: batch_size=384, img_size=320, workers=16

### 1.2 觀察到的問題
| 指標 | 數值 | 說明 |
|------|------|------|
| GPU 使用率 | **34%** | 嚴重低於預期 |
| 主進程 CPU | **121-131%** | 單核瓶頸 |
| Worker CPU | **3.4-3.8%** | 幾乎閒置 |
| 訓練速度 | ~1.19 s/it | 無明顯改善 |

---

## 2. 我之前的錯誤分析

### 2.1 錯誤結論
我之前認為：
> 「原始 YOLOv7 已經在 `__getitem__` 中呼叫 `torch.from_numpy()`，所以將 tensor 轉換移到 worker 的優化策略是無效的。」

### 2.2 為什麼這個分析是錯的

這個說法「技術上正確，但邏輯上錯誤」。

原因：**瓶頸不是 `torch.from_numpy()` 本身，而是 `collate_fn` 中的 `torch.stack()` 操作。**

---

## 3. 正確的瓶頸分析

### 3.1 YOLOv7 DataLoader 流程

```
┌─────────────────────────────────────────────────────────────┐
│ Worker Process (多核心，可平行)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. 讀取圖片 (cv2.imread)                                    │
│ 2. Mosaic 增強 (4 張圖拼接)                                 │
│ 3. 其他增強 (resize, flip, hsv...)                         │
│ 4. BGR→RGB, HWC→CHW                                        │
│ 5. torch.from_numpy(img)  ← 目前已在 worker 執行            │
│ 6. 回傳單張 tensor + labels                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Main Process - collate_fn (單核心，GIL 鎖定)                │
├─────────────────────────────────────────────────────────────┤
│ 1. zip(*batch)  ← 解包 384 個 samples                      │
│ 2. torch.stack(img, 0)  ← ⚠️ 堆疊 384 張 tensor (CPU 密集) │
│ 3. torch.cat(label, 0)  ← 合併所有 labels                  │
│ 4. 回傳給 GPU                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 真正的瓶頸

**`torch.stack()` 在主進程單核執行**，這才是 CPU 100% 的原因。

即使 `from_numpy()` 在 worker 完成：
- 384 張 tensor 仍需在主進程 stack
- stack 操作需要分配記憶體、複製資料
- Python GIL 鎖定，無法多核平行

---

## 4. Stage 4 程式碼問題

### 4.1 現有實作（Line 658-671）

```python
@staticmethod
def collate_fn_fast(batch):
    """Fast collate for worker_tensor mode (Phase 1)."""
    img, label, path, shapes = zip(*batch)
    for i, l in enumerate(label):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes
```

### 4.2 問題：與 collate_fn 完全相同

```python
@staticmethod
def collate_fn(batch):
    img, label, path, shapes = zip(*batch)
    for i, l in enumerate(label):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes
```

**`collate_fn_fast` 只是換了名字，沒有任何實質優化！**

---

## 5. 專家文件的正確方案

根據 4 份專家文件（ChatGPT、Gemini、Claude 共識），正確的優化分為兩個階段：

### 5.1 Phase 1（簡單版）- 專家文件建議

專家文件建議的 Phase 1：
- Worker 回傳 tensor（已完成）
- 新增 `persistent_workers=True`（已完成）
- **collate_fn 保持原樣**

**問題：這只能減少 epoch 切換開銷，無法解決 `torch.stack()` 瓶頸**

### 5.2 Phase 2（Micro-Collate）- 真正的解決方案

```
┌─────────────────────────────────────────────────────────────┐
│ Worker Process                                               │
├─────────────────────────────────────────────────────────────┤
│ 1. 處理 micro_batch (如 4 張圖)                              │
│ 2. 對每張圖做 mosaic + aug + from_numpy                      │
│ 3. torch.stack() 生成 (4, C, H, W) ← 在 worker 做！         │
│ 4. 回傳 micro-batch tensor                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Main Process - collate_fn (極輕量)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. torch.cat(micro_batches, dim=0) ← 只做 concat，非常快    │
│ 2. 回傳給 GPU                                               │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 預期改善

| 方案 | 主 CPU 負載 | 預期加速 |
|------|------------|----------|
| 目前 Stage 4 | 100% | 無改善 |
| Phase 1 簡單版 | 70-80% | 10-20% |
| Phase 2 Micro-Collate | 5-10% | **1.5-2x** |

---

## 6. 需要討論的問題

### Q1: Phase 1 的定義問題

專家文件中的 Phase 1 說明：
> 「將 numpy→tensor 移到 worker」

但原始 YOLOv7 本來就在 `__getitem__` 中做 `torch.from_numpy()`，這個「優化」早已存在。

**問題：Phase 1 的真正改善來源是什麼？**

可能的答案：
- A. `persistent_workers=True` 減少 epoch 切換開銷
- B. 搭配其他設定（如 `pin_memory`）才有效
- C. 原始設計假設有些 YOLOv7 版本在 collate 中做 from_numpy

### Q2: 是否直接跳到 Phase 2？

Phase 2 需要修改：
1. Sampler（支援 micro-batch 索引）
2. `__getitem__`（處理 index group）
3. `collate_fn`（改為 concat）

**風險：**
- 改動較大，可能引入 bug
- Debug 較困難
- 需要調整 micro_batch_size 參數

### Q3: 其他優化方向？

除了 Micro-Collate，是否考慮：
- NVIDIA DALI（GPU 端資料增強）
- Kornia（GPU 端影像處理）
- 預先快取 tensor 到 disk

---

## 7. 建議的下一步

### 方案 A：實作 Phase 2 Micro-Collate
- 改動大，效果明確
- 需要 1-2 天實作 + 測試

### 方案 B：先確認 Phase 1 的其他優化
- 檢查是否遺漏其他 Phase 1 設定
- 測試不同 worker 數量
- 確認 shared memory 配置

### 方案 C：跳過 DataLoader，直接用 DALI
- 最激進的方案
- 需要重寫整個資料 pipeline
- 效果可能最好

---

## 8. 附錄：相關程式碼位置

| 檔案 | 行號 | 內容 |
|------|------|------|
| `utils/datasets.py` | 646 | `torch.from_numpy(img)` |
| `utils/datasets.py` | 651-655 | `collate_fn` |
| `utils/datasets.py` | 658-671 | `collate_fn_fast` (與 collate_fn 相同) |
| `train.py` | DataLoader 建立處 | `persistent_workers`, `collate_fn` 選擇 |

---

**報告完成，請專家審閱並指導下一步方向。**
