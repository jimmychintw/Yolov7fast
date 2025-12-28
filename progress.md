# 專案進度報告

## 目前狀態：Full-Width 1B4N4H 訓練進行中 (2025-12-29)

---

## 1B1H Per-Head Baseline (正確版本)

### 重要發現：test.py iou_thres 不一致問題

訓練過程中的 validation 使用 `iou_thres=0.6`，但 test.py CLI 預設使用 `iou_thres=0.65`。
這會導致約 1% 的 mAP 差異。**比較時必須使用相同的 iou_thres！**

```python
# test.py 函數 default (train.py 調用時使用)
iou_thres=0.6

# test.py CLI default (命令列執行時使用)
parser.add_argument('--iou-thres', type=float, default=0.65, ...)
```

### 正確的 Per-Head Baseline

使用 `python test.py --iou-thres 0.6 --verbose` 測試 `1b1h_best.pt`：

| Head | 類別數 | mAP@0.5 Baseline | 說明 |
|------|--------|------------------|------|
| H0 | 1 | **0.6160** | person only |
| H1 | 26 | **0.4503** | AntiConfusion Group 1 |
| H2 | 26 | **0.4429** | AntiConfusion Group 2 |
| H3 | 27 | **0.4051** | AntiConfusion Group 3 |
| **整體** | 80 | **0.435** | 與 training log 吻合 |

### 資料來源

- 權重檔：`1b1h_best.pt` (來自 `runs/train/20251201_1b1h_500ep_bs128/weights/best.pt`)
- 測試指令：`python test.py --weights 1b1h_best.pt --data data/coco320.yaml --img-size 320 --iou-thres 0.6 --verbose`
- 計算腳本：`temp/calc_perhead_baseline.py`

---

## Full-Width 1B4N4H 訓練 (2025-12-29)

### 實驗目的

基於 1B1H 500ep 權重 (`1b1h_best.pt`)，使用 `--focus-head` 分別訓練各 Head 的類別。
目標：驗證 Full-Width Neck+Head 專注訓練能否超越 1B1H baseline。

### 訓練配置

```bash
# H0 (person, 1 class)
python train.py --img-size 320 320 --batch-size 128 \
    --weights 1b1h_best.pt --epochs 500 \
    --data data/coco320.yaml \
    --head-config data/coco_320_1b4h_anticonfusion.yaml \
    --focus-head 0 --stage stage1_neck_tune \
    --name head0_full_from1b1h_500ep

# H1 (26 classes)
python train.py ... --focus-head 1 --name head1_full_from1b1h_500ep

# H2 (26 classes)
python train.py ... --focus-head 2 --name head2_full_from1b1h_500ep
```

### 訓練進度 (2025-12-29 更新)

| Head | Epochs | 當前 mAP | Baseline | 絕對進步 | 相對進步 | 伺服器 |
|------|--------|----------|----------|----------|----------|--------|
| H0 | 121/500 | 0.6407 | 0.6160 | **+2.47%** | +4.0% ✓ | 285 |
| H1 | 108/500 | 0.4578 | 0.4503 | **+0.75%** | +1.7% ✓ | 9950 |
| H2 | 111/500 | 0.4657 | 0.4429 | **+2.28%** | +5.1% ✓ | 2852 |
| H3 | - | - | 0.4051 | - | - | 待訓練 |

**結論：三個 Head 全部超越 1B1H baseline！**

### 伺服器狀態

| 伺服器 | 任務 | 進度 | 狀態 |
|--------|------|------|------|
| 285 | H0 Full-Width | 121/500 | 🔄 進行中 |
| 9950 | H1 Full-Width | 108/500 | 🔄 進行中 |
| 2852 | H2 Full-Width | 111/500 | 🔄 進行中 |

### 數據檔案

| 檔案 | 說明 |
|------|------|
| `temp/h0_from1b1h_results.txt` | H0 訓練結果 |
| `temp/h1_from1b1h_results.txt` | H1 訓練結果 |
| `temp/h2_from1b1h_results.txt` | H2 訓練結果 |
| `temp/fullwidth_progress.png` | 進度對比圖表 |
| `temp/calc_perhead_baseline.py` | Per-head baseline 計算腳本 |

---

## Person-only 實驗 (2025-12-28)

### 實驗目的

基於 1B1H (yolov7-tiny.pt) 權重，凍結 Backbone，只訓練 Neck+Head，專注於 person 單一類別 (nc=1)。

### 實驗配置

```bash
python train.py --img-size 320 320 --batch-size 128 --test-batch-size 128 \
    --weights yolov7-tiny.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --cfg cfg/training/yolov7-tiny-320.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 --workers 16 \
    --project runs/train \
    --name 1b1h_person_only_stage1_500ep \
    --noautoanchor --cache-images \
    --focus-class person \
    --stage stage1_neck_tune \
    --lr-mult-head 1.0 \
    --lr-mult-neck 0.3 \
    --warmup-restart-epochs 5 \
    --print-stage-summary
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
| 285 | Person-only Full-Width | 145/500 | 🔄 進行中 |
| 9950 | Person-only Half-Width | 待啟動 | ⏳ 就緒 |

---

## Person-only Half-Width 實驗 (2025-12-28)

### 實驗目的

與 Full-Width 實驗對照，測試 Neck/Head 通道減半對 Person-only 任務的影響。

### 實驗配置

```bash
python train.py --img-size 320 320 --batch-size 128 --test-batch-size 128 \
    --weights yolov7-tiny.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 --workers 16 \
    --project runs/train \
    --name 1b1h_person_only_half_stage1_500ep \
    --noautoanchor --cache-images \
    --focus-class person \
    --nh-width-mult 0.5 \
    --stage stage1_neck_tune \
    --lr-mult-head 1.0 \
    --lr-mult-neck 0.3 \
    --warmup-restart-epochs 5
```

### 架構差異

| 位置 | Full-Width (285) | Half-Width (9950) |
|------|------------------|-------------------|
| Backbone | 128/256/512 ch | 128/256/512 ch (不變) |
| Adapter | 無 | 1x1 Conv 降維 |
| Neck P3/P4/P5 | 64/128/256 ch | **32/64/128 ch** |
| Det P3/P4/P5 | 128/256/512 ch | **64/128/256 ch** |
| 配置檔 | yolov7-tiny-320.yaml | yolov7-tiny-320-half-adapter.yaml |

### 起點差異 (重要)

| 實驗 | Backbone | Neck+Head 初始化 | 起點 mAP@0.5 |
|------|----------|------------------|--------------|
| **285 Full-Width** | 凍結 (pretrained) | **Pretrained** (1B1H 權重) | ~0.60 |
| **9950 Half-Width** | 凍結 (pretrained) | **隨機初始化** (通道不匹配) | ~0.58 |

> Half-Width 的 Neck+Head 通道數與 yolov7-tiny.pt 不匹配，無法載入預訓練權重，必須從頭訓練。

### 預期比較

| 指標 | Full-Width | Half-Width Stage1 | Half-Width Stage2 (計畫) |
|------|------------|-------------------|-------------------------|
| 參數量 | ~6M | ~3M | ~3M |
| 起點 | Pretrained | Random | **Trained (Stage1 best)** |
| mAP@0.5 | 0.65 | 0.61 | **0.64 (目標)** |
| vs Baseline | +5.3% | -1.1% | **+3.7%** |

### Half-Width Stage2 訓練計畫

Stage1 完成後，使用 best.pt 重新訓練 500 epochs：

```bash
python train.py --img-size 320 320 --batch-size 128 --test-batch-size 128 \
    --weights runs/train/1b1h_person_only_half_stage1_500ep/weights/best.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 --workers 16 \
    --project runs/train \
    --name 1b1h_person_only_half_stage2_500ep \
    --noautoanchor --cache-images \
    --focus-class person \
    --nh-width-mult 0.5 \
    --stage stage1_neck_tune \
    --lr-mult-head 1.0 \
    --lr-mult-neck 0.3 \
    --warmup-restart-epochs 10
```

**目標：mAP@0.5 = 0.64，超越 Baseline 2%+**

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

### 285 (Full-Width)
```
SSH: ssh -p 45897 root@173.239.88.241 -L 8080:localhost:8080
GPU: RTX 5090 (32GB)
tmux: 4 windows (train, cpu, gpu, terminal) - 綠色狀態列
任務: Person-only Full-Width Stage1 500ep (進行中)
```

### 9950 (Half-Width)
```
SSH: ssh -p 52652 root@79.117.62.136 -L 8080:localhost:8080
GPU: RTX 5090 (32GB)
tmux: 4 windows (train, cpu, gpu, terminal) - 黃色狀態列
任務: Person-only Half-Width Stage1 500ep (待啟動)
```

---

## Focus-Head 多類別專注訓練 (2025-12-28)

### 功能說明

新增 `--focus-head` 參數，可指定訓練 AntiConfusion 配置中的某個 Head 的所有類別。

### 與 Focus-Class 的差異

| 項目 | --focus-class | --focus-head |
|------|---------------|--------------|
| 用途 | 單一類別訓練 (如 person) | 多類別 Head 訓練 |
| nc | **1** (remap 到 class 0) | **80** (保持原始 COCO ID) |
| 配置需求 | 無 | 需搭配 --head-config |
| 範例 | --focus-class person | --focus-head 1 --head-config data/coco_320_1b4h_anticonfusion.yaml |

### Head 分配 (AntiConfusion)

| Head | 名稱 | 類別數 | 說明 |
|------|------|--------|------|
| 0 | Person_Specialist | 1 | person 獨立 |
| 1 | AntiConfusion_Group_1 | 26 | car, motorcycle, airplane... |
| 2 | AntiConfusion_Group_2 | 26 | bus, cat, sheep... |
| 3 | AntiConfusion_Group_3 | 27 | bicycle, truck, dog... |

### 使用範例

```bash
# Head 1 Full-Width (26 classes)
python train.py --img-size 320 320 --batch-size 128 \
    --weights yolov7-tiny.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --head-config data/coco_320_1b4h_anticonfusion.yaml \
    --focus-head 1 \
    --stage stage1_neck_tune \
    --name head1_full_stage1_500ep

# Head 1 Half-Width (26 classes)
python train.py --img-size 320 320 --batch-size 128 \
    --weights yolov7-tiny.pt \
    --epochs 500 \
    --data data/coco320.yaml \
    --head-config data/coco_320_1b4h_anticonfusion.yaml \
    --focus-head 1 \
    --nh-width-mult 0.5 \
    --stage stage1_neck_tune \
    --name head1_half_stage1_500ep
```

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| train.py | 新增 --focus-head 參數，處理邏輯 |
| utils/focus_class.py | 新增 parse_focus_head, filter_labels_by_head |
| utils/datasets.py | 支援 focus_head 參數傳遞 |

---

## 變更歷史

### 2025-12-28 (Focus-Head 功能實作)
- 實作 `--focus-head` 參數，支援多類別 Head 訓練
- 新增 `parse_focus_head()` 和 `filter_labels_by_head()` 函數
- 保持 nc=80，只過濾 labels（不 remap class ID）
- 測試驗證通過：語法檢查 ✓、parse_focus_head ✓、filter_labels_by_head ✓

### 2025-12-28 (Person-only 雙實驗)
- 實作 `--focus-class` 功能，支援單一類別訓練
- 實作 `--nh-width-mult` 功能，支援 Neck/Head 通道縮放
- **Full-Width 實驗 (285)**：
  - 基於 yolov7-tiny.pt (1B1H) 啟動 Person-only 訓練
  - 發現 V-Shape Recovery 現象
  - 最新結果 (Epoch 145/500): mAP@0.5 = 0.639, +3.5% vs baseline
- **Half-Width 實驗 (9950)**：
  - Neck/Head 通道減半 (32/64/128 vs 64/128/256)
  - 使用 1x1 Adapter 連接 Backbone
  - 配置檔：yolov7-tiny-320-half-adapter.yaml
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
