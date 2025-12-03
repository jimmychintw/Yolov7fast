# 1B4H 漸進式解凍訓練計畫

## 問題背景

目前 1B4H 實驗都使用 `--freeze 50` 凍結 backbone，結果：
- **優點**：快速收斂到 mAP ≈ 0.42
- **缺點**：天花板低，無法超越 1B1H 的 0.4353

直接 unfreeze 會導致 mAP 從 0.42 暴跌到 0.14，因為 learning rate 太大破壞了已學好的權重。

## 解決方案：漸進式解凍

從靠近 Head 的深層開始，逐步解凍更淺的層，每階段使用更低的 learning rate。

```
YOLOv7-tiny 層結構：
┌─────────────────────────────────────────────────────────────┐
│  層 0-10:   淺層特徵（邊緣、顏色、紋理）    ← Stage 4 解凍  │
│  層 11-30:  中層特徵（形狀、部件）          ← Stage 3 解凍  │
│  層 31-50:  深層特徵（物體、語意）          ← Stage 2 解凍  │
│  層 51+:    Head 層（分類、回歸）           ← Stage 1 已解凍 │
└─────────────────────────────────────────────────────────────┘
```

---

## 訓練計畫

### 總覽

| Stage | Freeze | lr0 | Epochs | 累計 | 說明 |
|-------|--------|-----|--------|------|------|
| 1 | 50 | 0.01 | 100 | 100 | 只訓練 Head，快速收斂 |
| 2 | 30 | 0.001 | 100 | 200 | 解凍深層 (31-50)，適應 4H |
| 3 | 10 | 0.0005 | 100 | 300 | 解凍中層 (11-30) |
| 4 | 0 | 0.0001 | 200 | 500 | 全層微調 |

### Hyperparameter 檔案

| Stage | 檔案 | lr0 |
|-------|------|-----|
| 1 | `hyp.scratch.tiny.noota.yaml` | 0.01 |
| 2 | `hyp.scratch.tiny.noota.stage2.yaml` | 0.001 |
| 3 | `hyp.scratch.tiny.noota.stage3.yaml` | 0.0005 |
| 4 | `hyp.scratch.tiny.noota.stage4.yaml` | 0.0001 |

---

## 訓練指令

### 使用 AntiConfusion 分類（推薦）

```bash
# 設定共用變數
cd /workspace/Yolov7fast && source venv/bin/activate
BASE_ARGS="--img-size 320 320 --batch-size 64 --test-batch-size 64 \
    --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --device 0 --workers 16 --project runs/train --noautoanchor --cache-images \
    --heads 4 --head-config data/coco_320_1b4h_anticonfusion.yaml"

# Stage 1: freeze 50, lr=0.01, 100ep
python train.py $BASE_ARGS \
    --weights runs/train/20251201_1b1h_500ep_bs128/weights/best.pt \
    --transfer-weights --freeze 50 --epochs 100 \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --name gradual_anti_stage1_freeze50

# Stage 2: freeze 30, lr=0.001, 100ep
python train.py $BASE_ARGS \
    --weights runs/train/gradual_anti_stage1_freeze50/weights/best.pt \
    --freeze 30 --epochs 100 \
    --hyp data/hyp.scratch.tiny.noota.stage2.yaml \
    --name gradual_anti_stage2_freeze30

# Stage 3: freeze 10, lr=0.0005, 100ep
python train.py $BASE_ARGS \
    --weights runs/train/gradual_anti_stage2_freeze30/weights/best.pt \
    --freeze 10 --epochs 100 \
    --hyp data/hyp.scratch.tiny.noota.stage3.yaml \
    --name gradual_anti_stage3_freeze10

# Stage 4: freeze 0, lr=0.0001, 200ep
python train.py $BASE_ARGS \
    --weights runs/train/gradual_anti_stage3_freeze10/weights/best.pt \
    --freeze 0 --epochs 200 \
    --hyp data/hyp.scratch.tiny.noota.stage4.yaml \
    --name gradual_anti_stage4_freeze0
```

---

## 預期結果

| Stage | 預期 mAP | 說明 |
|-------|----------|------|
| Stage 1 | 0.42 | 與現有 freeze 50 結果相當 |
| Stage 2 | 0.43 | 深層適應 4H，應該開始超越 |
| Stage 3 | 0.435 | 中層加入，接近 1B1H |
| Stage 4 | **0.44+** | 全層微調，有機會超越 1B1H |

---

## 監控指標

每個 Stage 結束時檢查：

1. **mAP 是否上升**：若下降超過 0.01，可能 lr 太大
2. **Loss 曲線**：應該平穩下降，不應該有跳動
3. **各 Head 的 AP**：確保沒有某個 Head 崩掉

---

## 替代方案

### 方案 B：更保守的解凍

如果 Stage 2 仍然崩掉，改用更小的步進：

```
Stage 1: freeze 50 → Stage 2: freeze 40 → Stage 3: freeze 30 → ...
```

### 方案 C：更低的 lr

```
Stage 2: lr=0.0005 (而非 0.001)
Stage 3: lr=0.0002
Stage 4: lr=0.00005
```

---

## 執行順序建議

1. 先讓目前的 9950 (Hybrid) 和 7950 (AntiConfusion) 跑完
2. 等 AntiConfusion 500ep 結果出來，確認分類策略
3. 再用最佳分類策略執行漸進式解凍
4. 或者直接開新機器跑漸進式解凍

---

*計畫建立時間: 2025-12-02*
