# 專案進度報告

## 目前狀態：OTA vs non-OTA 對比實驗完成 ✅

### 訓練結果對比 (2025-11-28)

| 版本 | 訓練時間 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 狀態 |
|------|---------|---------|--------------|-----------|--------|------|
| **non-OTA** | 1.81 小時 | 0.385 | 0.226 | **0.568** | 0.355 | ✅ 完成 |
| **OTA** | 10.65 小時 | **0.414** | **0.251** | 0.558 | **0.400** | ✅ 完成 |
| **差異** | 5.9x 更慢 | +7.5% | +11.1% | -1.8% | +12.7% | |

### 關鍵結論

1. **OTA Loss 精度較高**: mAP@0.5 提升 7.5%，mAP@0.5:0.95 提升 11.1%
2. **non-OTA 效率較高**: 訓練速度快 5.9 倍，GPU 利用率 90% vs 13%
3. **瓶頸分析**: OTA 的 SimOTA 匹配演算法佔用 86.5% 的迭代時間

詳細分析請見 [OTA_ANALYSIS_REPORT.md](OTA_ANALYSIS_REPORT.md)

---

## 訓練詳細結果

### non-OTA 訓練結果 (noota_100ep2)

- **訓練目錄**：`runs/train/noota_100ep2`
- **最終 mAP@0.5**：0.385
- **最終 mAP@0.5:0.95**：0.226
- **Precision**：0.568
- **Recall**：0.355
- **訓練時間**：1.81 小時
- **訓練速度**：~5.76 it/s
- **GPU 利用率**：~90%

### OTA 訓練結果 (ota_100ep4)

- **訓練目錄**：`runs/train/ota_100ep4`
- **最終 mAP@0.5**：0.414
- **最終 mAP@0.5:0.95**：0.251
- **Precision**：0.558
- **Recall**：0.400
- **訓練時間**：10.65 小時
- **訓練速度**：~0.97 it/s
- **GPU 利用率**：~13%

---

## 效能分析結果 (2025-11-27)

**ComputeLossOTA 是訓練緩慢的主要原因**

| 指標 | ComputeLossOTA | ComputeLoss | 改善 |
|------|----------------|-------------|------|
| **loss計算** | 1016.03 ms | **11.22 ms** | **90.5x 更快** |
| forward | 50.20 ms | 50.24 ms | 相同 |
| backward | 103.39 ms | 103.60 ms | 相同 |
| **Total/iter** | 1174.06 ms | **170.89 ms** | **6.9x 更快** |
| **GPU利用率** | 13.1% | **90.0%** | 從超低變正常 |

---

## 已完成項目

| 日期 | 項目 | 說明 |
|------|------|------|
| 2025-11-25 | 專案初始化 | 建立 Python 3.12 venv |
| 2025-11-25 | GitHub 設定 | 建立遠端倉庫 jimmychintw/Yolov7fast |
| 2025-11-25 | 基礎程式碼 | 從 jimmychintw/yolov7 複製 YOLOv7 原始碼 |
| 2025-11-25 | 開發規範 | 建立 CLAUDE.md 定義開發規則 |
| 2025-11-25 | 進度追蹤 | 建立 progress.md 進度報告機制 |
| 2025-11-26 | COCO 資料集 | 確認本地有 320x320 版本 (5.9GB, 118287 張) |
| 2025-11-26 | 多解析度支援 | 建立 coco320/480/640 目錄結構與設定檔 |
| 2025-11-26 | 文檔更新 | 重寫 README.md 為簡潔版本 |
| 2025-11-27 | vast.ai 環境 | 新 instance 設定完成 (RTX 5090 + PyTorch 2.8.0) |
| 2025-11-27 | 程式碼修正 | 修正 test.py 硬編碼 annotations 路徑問題 |
| 2025-11-27 | 設定文檔 | 重寫 VAST_SETUP.md 為一鍵設定指南 |
| 2025-11-27 | 效能分析 | 找出 ComputeLossOTA 是瓶頸 |
| 2025-11-28 | PyTorch 2.8 相容 | 修正所有 torch.load 加入 weights_only=False |
| 2025-11-28 | non-OTA 訓練 | 完成 100 epochs，mAP@0.5 = 0.385 |
| 2025-11-28 | OTA 訓練 | 完成 100 epochs，mAP@0.5 = 0.414 |
| 2025-11-28 | 分析報告 | 撰寫完整 OTA vs non-OTA 分析報告 |

---

## vast.ai 遠端環境

```
SSH: ssh -p 21024 root@116.122.206.233 -L 6006:localhost:6006
GPU: RTX 5090 (32GB VRAM)
PyTorch: 2.8.0+cu128 (支援 Blackwell sm_120)
tmux session: vast (4 windows: train, cpu, gpu, terminal)
TensorBoard: http://localhost:6006
```

---

## 下次繼續事項

- [x] 上傳 coco320 資料集到 vast.ai (5.9GB)
- [x] 在遠端開始第一次訓練測試
- [x] 找出效能瓶頸 (ComputeLossOTA)
- [x] 使用 `loss_ota: 0` 跑完整訓練 100 epochs
- [x] 跑 OTA 版本 100 epochs 對比
- [x] 對比 OTA vs non-OTA 訓練的精度差異 (mAP)
- [ ] 決定後續優化方向
- [ ] 測試 non-OTA 300 epochs 是否能達到 OTA 100 epochs 的精度
- [ ] 調查 OTA 的 GPU 實現可能性
- [ ] 測試 640x640 解析度下的差異

---

## 訓練指令參考

```bash
# 在 vast.ai 遠端執行
cd /workspace/Yolov7fast

# non-OTA (快速訓練，~1.8 小時)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor

# OTA (標準訓練，~10-12 小時)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.yaml --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor
```

---

## 變更歷史

### 2025-11-28 (下午 - 分析報告)
- OTA 100 epochs 訓練完成，mAP@0.5 = 0.414
- 撰寫完整 OTA vs non-OTA 分析報告 (OTA_ANALYSIS_REPORT.md)
- 更新 progress.md 加入完整對比結果

### 2025-11-28 (凌晨)
- non-OTA 100 epochs 訓練完成，mAP@0.5 = 0.385
- 修正 PyTorch 2.8 相容性問題 (torch.load weights_only)
- 啟動 OTA 100 epochs 訓練

### 2025-11-27 (下午 - 效能分析)
- 建立效能分析計劃 PERFORMANCE_ANALYSIS_PLAN_V2.md
- 建立 CUDA Events 剖析工具 tests/profile_training_loop.py
- 發現 ComputeLossOTA 佔用 86.5% 訓練時間
- 建立 hyp.scratch.tiny.noota.yaml (關閉 OTA Loss)
- 驗證：關閉 OTA 後訓練速度提升 6.9x，GPU 利用率從 13% 提升到 90%

### 2025-11-27 (上午)
- 租用新 vast.ai instance (RTX 5090)
- 設定 SSH key 連線
- 安裝 PyTorch 2.8.0 + CUDA 12.8 (支援 Blackwell 架構)
- 安裝所有依賴套件
- Clone 專案到 /workspace/Yolov7fast
- 建立 tmux 環境 (train, cpu, gpu, terminal)
- 啟動 TensorBoard (port 6006)
- 刪除空的 coco/ 目錄
- 修正 test.py 硬編碼 annotations 路徑（改為從 data yaml 自動推導）
- 重寫 VAST_SETUP.md 為一鍵設定指南

### 2025-11-26
- 修正 COCO 資料集路徑設定（移除硬編碼）
- 確認本地資料集為 320x320 版本
- 重命名 coco → coco320
- 建立多解析度支援：coco320.yaml, coco480.yaml, coco640.yaml
- 重寫 README.md 為簡潔版本
- 建立 VAST_SETUP.md 設定文檔

### 2025-11-25
- 專案建立
- GitHub 倉庫初始化：https://github.com/jimmychintw/Yolov7fast
- 匯入 YOLOv7 基礎程式碼（107 個檔案）
- 建立 CLAUDE.md 開發規範（6 條規定）
- 建立 progress.md 進度追蹤機制
