# 專案進度報告

## 目前狀態：Person-only 實驗進行中 (2025-12-28)

---

## Person-only 實驗 (2025-12-28)

### 實驗目的

基於 1B1H (yolov7-tiny.pt) 權重，凍結 Backbone，只訓練 Neck+Head，專注於 person 單一類別 (nc=1)。

### 實驗配置

```bash
python train.py --img-size 320 320 --batch-size 64 --test-batch-size 64 \
    --weights yolov7-tiny.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --cfg cfg/training/yolov7-tiny-320.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 --workers 16 \
    --project runs/train \
    --name 1b1h_person_only_stage1_500ep2 \
    --noautoanchor --cache-images \
    --focus-class person \
    --stage stage1_neck_tune \
    --lr-mult-head 1.0 \
    --lr-mult-neck 0.3 \
    --lr-mult-backbone 0.05 \
    --warmup-restart-epochs 5
```

### 關鍵發現：V-Shape Recovery

訓練過程中觀察到有趣的 V-Shape 恢復模式：

| 階段 | Epoch 範圍 | mAP@0.5 變化 | 說明 |
|------|-----------|-------------|------|
| **Initial Rise** | 0-19 | 0.598 → 0.630 | Pretrained 優勢快速上升 |
| **Adaptation Dip** | 20-45 | 0.630 → 0.601 | Neck+Head 適應 person-only |
| **Recovery** | 45-129 | 0.601 → **0.638** | 收斂到新的最佳解，持續上升 |

### 訓練結果 (Epoch 129/500)

| 指標 | 1B1H Baseline | Best (Epoch) | Current (Ep129) | vs Baseline |
|------|---------------|--------------|-----------------|-------------|
| **mAP@0.5** | 0.617 | 0.639 (Ep116) | **0.638** | **+3.3%** |
| **mAP@0.5:0.95** | 0.365 | **0.382 (Ep129)** | **0.382** | **+4.6%** |
| **Precision** | 0.718 | 0.801 (Ep5) | 0.792 | +10.3% |
| **Recall** | 0.562 | 0.560 (Ep99) | 0.548 | -2.5% |

### Loss 變化

| Loss | Start (Ep0) | Current (Ep129) | 變化 |
|------|-------------|-----------------|------|
| Box Loss | 0.0732 | 0.0540 | **-26.2%** |
| Obj Loss | 0.0138 | 0.0132 | -4.4% |
| Total Loss | 0.0870 | 0.0672 | **-22.7%** |

### 分析結論

1. **V-Shape Recovery 成功**：
   - 初期快速上升沾了 pretrained 的光
   - 中期下降是 Neck+Head 從 80-class 適應到 person-only
   - 後期回升證明模型成功收斂到 person-only 最佳解

2. **Neck+Head 參數可視為「需重訓練」**：
   - 真正保護的只有 Backbone 特徵
   - Neck+Head 經過 V-shape 適應期後表現更好

3. **大幅超越 Baseline**：
   - mAP@0.5: 0.638 > 0.617 (+3.5%)
   - mAP@0.5:0.95: 0.381 > 0.365 (+4.4%)
   - 證明 person-only 專注訓練非常有效

### 數據檔案

| 檔案 | 說明 |
|------|------|
| `temp/person_only_results.txt` | 原始訓練結果 |
| `temp/person_only_training_data.csv` | CSV 格式數據 (含 Loss, P, R, mAP) |
| `temp/person_only_full_comparison.png` | 完整比較圖表 |

### 伺服器狀態

| 伺服器 | 任務 | 進度 | 狀態 |
|--------|------|------|------|
| 285 | Person-only Stage1 500ep | 130/500 | 🔄 進行中 |

---

## Stage 1 訓練結果總結 (2025-12-27)

### 訓練配置與結果

| 版本 | Batch Size | LR | Epochs | Best mAP@0.5 | 狀態 |
|------|------------|-----|--------|--------------|------|
| Stage1 100ep | 64 | 0.01 | 100 | 0.4270 | ✅ 完成 |
| Stage1_500 bs64 | 64 | 0.01 | 500 | 0.4300+ | 🔄 285 進行中 (428/500) |
| **Stage1_500 bs128** | 128 | 0.01 | 500 | **0.4300** | ✅ 完成 |
| Stage1 bs512 lr=0.08 | 512 | 0.08 | ~15 | 發散 | ❌ 失敗 |
| Stage1 bs512 lr=0.04 | 512 | 0.04 | 160 | 0.3682 | ❌ 收斂慢 |
| Stage1 bs512 lr=0.03 | 512 | 0.03 | 209 | 0.3692 | ❌ 停止 |

### 關鍵發現

1. **bs128 是最佳 batch size**：
   - Best mAP@0.5 = 0.4300，超越基線 Stage1 100ep (0.4270)
   - 比 1B1H baseline (0.4353) 還差 0.5%

2. **bs512 不適合此任務**：
   - 即使調整 LR (0.08 → 0.04 → 0.03)，效果仍不如 bs128
   - 大 batch size 導致收斂困難，Loss 偏高

3. **Precision-Recall Trade-off**：
   - Stage1 500ep 傾向「多檢測」(高 Recall) 而非「精準檢測」(高 Precision)
   - Precision: 0.5847 (bs128 500ep) vs 0.6206 (100ep)
   - Recall: 0.4108 (bs128 500ep) vs 0.3943 (100ep)

4. **Stage1 的 Ceiling**：
   - mAP@0.5 約在 0.43，受限於 Backbone 凍結
   - 需進入 Stage2 解凍 late backbone 才能突破

### 伺服器狀態

| 伺服器 | 任務 | 狀態 |
|--------|------|------|
| 285 | Stage1_500 bs64 | 🔄 428/500 進行中 |
| 9950 | (已停止 bs512) | ⏸️ 待命 |

---

## 下一步計畫：Stage 2 訓練

### Stage 2 配置 (Late Backbone Tune)

```bash
python train.py --img-size 320 320 --batch-size 128 --test-batch-size 128 \
    --weights runs/train/1b4h_stage1_500ep_bs1283/weights/best.pt \
    --epochs 300 \
    --data data/coco320.yaml \
    --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 --workers 16 \
    --project runs/train \
    --name 1b4h_stage2_300ep \
    --noautoanchor --cache-images \
    --heads 4 \
    --head-config data/coco_320_1b4h_anticonfusion.yaml \
    --ignore-other-heads \
    --stage stage2_late_backbone_tune \
    --lr-mult-head 1.0 \
    --lr-mult-neck 0.3 \
    --lr-mult-backbone 0.05 \
    --warmup-restart-epochs 10 \
    --print-stage-summary
```

### Stage 2 預期

- 解凍 BACKBONE_LATE (model.23 ~ model.37)
- 使用 Stage1 bs128 best weights 作為基底
- 目標：突破 0.43，接近或超越 1B1H (0.4353)

---

## 新增 Hyp 配置檔案

### hyp.scratch.tiny.bs512.yaml
- lr0: 0.03 (for bs512)
- warmup_epochs: 10
- 結論：不建議使用，bs512 效果不佳

### hyp.scratch.tiny.high_precision.yaml
- obj_pw: 1.5, cls_pw: 1.5
- fl_gamma: 1.5 (Focal Loss)
- iou_t: 0.25
- mosaic: 0.8, mixup: 0.0
- 用途：提升 Precision

---

## 1B1H vs 1B4H Loss 分析 (2025-12-27)

### Loss 計算差異

| 指標 | 1B1H (avg) | 1B4H Stage1 | 比值 |
|------|------------|-------------|------|
| Box Loss | 0.0570 | 0.2182 | 3.83x |
| Obj Loss | 0.04404 | 0.05100 | 1.16x |
| Cls Loss | 0.0347 | 0.1439 | **4.14x** |
| Total Loss | 0.1358 | 0.4131 | 3.04x |

### 原因
- Cls Loss 差異 ~4x 是因為 BCEWithLogitsLoss 使用 reduction='mean'
- 1B4H 每個 Head 只有 20 個類別，平均時分母較小

### 有效比較指標
- ✅ mAP@0.5, Precision, Recall（不受 loss 計算影響）
- ❌ Cls Loss, Total Loss（無法直接比較）

---

## Batch Size Scaling 經驗總結

| Batch Size | LR | 效果 |
|------------|-----|------|
| 64 | 0.01 | ✅ 穩定 |
| 128 | 0.01 | ✅ 最佳（應調到 0.014-0.02 更好）|
| 512 | 0.08 (8x) | ❌ 發散 |
| 512 | 0.04 (4x) | ❌ 收斂慢 |
| 512 | 0.03 (3x) | ❌ 仍不如 bs128 |

**結論**：對於 YOLOv7-Tiny 1B4H Stage1，bs128 + lr=0.01 是最佳配置。

---

## 伺服器資訊

### 285 (當前使用)
```
SSH: ssh -p 45897 root@173.239.88.241 -L 8080:localhost:8080
GPU: RTX 5090
tmux: 4 windows (train, cpu, gpu, terminal)
任務: Person-only Stage1 500ep (進行中)
```

### 9950
```
SSH: ssh -p 52652 root@79.117.62.136 -L 8080:localhost:8080
GPU: RTX 5090 (32GB)
狀態: 閒置，待命中
```

---

## 變更歷史

### 2025-12-28 (Person-only 實驗)
- 實作 `--focus-class` 功能，支援單一類別訓練
- 基於 yolov7-tiny.pt (1B1H) 啟動 Person-only 訓練
- 發現 V-Shape Recovery 現象：
  - Epoch 0-19: 快速上升 (0.598 → 0.630)
  - Epoch 20-45: 適應期下降 (0.630 → 0.601)
  - Epoch 45-109: 恢復上升 (0.601 → 0.638)
- 最新結果 (Epoch 109/500):
  - mAP@0.5: 0.638 (Baseline: 0.617, **+3.5%**)
  - mAP@0.5:0.95: 0.381 (Baseline: 0.365, **+4.4%**)
  - Total Loss: 0.087 → 0.067 (-22.5%)
- 結論：Neck+Head 可視為「需重訓練」，真正保護的是 Backbone 特徵
- 數據檔案：temp/person_only_training_data.csv

### 2025-12-27 (Stage 1 實驗完成)
- Stage1_500 bs128 完成：Best mAP@0.5 = 0.4300
- bs512 實驗失敗：lr=0.08/0.04/0.03 都不如 bs128
- 創建 hyp.scratch.tiny.bs512.yaml (lr=0.03)
- 創建 hyp.scratch.tiny.high_precision.yaml
- 分析 1B1H vs 1B4H Loss 計算差異
- 準備進入 Stage 2 訓練

### 2025-12-26 (Stage 1 訓練)
- 啟動 Stage1_500 bs64 (285) 和 bs128 (9950)
- 分析 LR scaling 理論

### 2025-12-25 (Two-Stage Unfreezing 實作)
- 實作 Stage1/Stage2 訓練模式
- utils/freeze_by_index.py 實作

### 2025-12-24 (策略 A 訓練完成)
- 策略 A + AntiConfusion 500ep 完成
- Best mAP@0.5 = 0.4353 (與 1B1H 持平)
