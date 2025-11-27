# 專案進度報告

## 目前狀態：效能瓶頸已找到 - ComputeLossOTA 是主因

### 重大發現 (2025-11-27)

**效能分析結果：ComputeLossOTA 是訓練緩慢的主要原因**

| 指標 | ComputeLossOTA | ComputeLoss | 改善 |
|------|----------------|-------------|------|
| **loss計算** | 1016.03 ms | **11.22 ms** | **90.5x 更快** |
| forward | 50.20 ms | 50.24 ms | 相同 |
| backward | 103.39 ms | 103.60 ms | 相同 |
| **Total/iter** | 1174.06 ms | **170.89 ms** | **6.9x 更快** |
| **GPU利用率** | 13.1% | **90.0%** | 從超低變正常 |

**結論：**
- OTA Loss 的計算佔用了 86.5% 的訓練時間
- 關閉 OTA (`loss_ota: 0`) 後，訓練速度提升 **6.9x**
- 這是之前看到 6x 提速的真正原因（不是 mosaic）

**建議優化方向：**
1. 使用 `hyp.scratch.tiny.noota.yaml` 進行快速訓練
2. 研究 OTA Loss 的 GPU 優化可能性
3. 考慮在訓練後期才啟用 OTA 以獲得精度收益

### 已完成項目

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
| 2025-11-27 | 清理 | 刪除空的 coco/ 目錄 |

### 目前專案結構

```
Yolov7fast/
├── cfg/training/          # 模型架構配置
│   ├── yolov7-tiny.yaml
│   ├── yolov7.yaml
│   └── ...
├── data/                  # 資料集配置
│   ├── coco320.yaml      # 320x320 (有資料)
│   ├── coco480.yaml      # 480x480 (空)
│   └── coco640.yaml      # 640x640 (空)
├── coco320/              # 320x320 資料集 (5.9GB)
│   ├── images/train2017/ # 118,287 張
│   ├── images/val2017/   # 5,000 張
│   ├── labels/
│   └── annotations/
├── coco480/              # 待填入
├── coco640/              # 待填入
├── train.py
├── detect.py
├── test.py
├── CLAUDE.md
├── README.md
├── VAST_SETUP.md         # vast.ai 一鍵設定指南
└── progress.md
```

### vast.ai 遠端環境

```
SSH: ssh -p 21024 root@116.122.206.233 -L 6006:localhost:6006
GPU: RTX 5090 (32GB VRAM)
PyTorch: 2.8.0+cu128 (支援 Blackwell sm_120)
tmux session: vast (4 windows: train, cpu, gpu, terminal)
TensorBoard: http://localhost:6006
```

### 目前進行中 (2025-11-27 晚間)

**non-OTA 100 epochs 訓練正在執行中**
- 訓練目錄：`runs/train/noota_100ep2`
- hyp 檔案：`data/hyp.scratch.tiny.noota.yaml` (loss_ota: 0)
- 預估完成時間：約 1.5 小時
- 訓練速度：~5.8 it/s (170ms/iter)

**查看進度指令：**
```bash
ssh -p 21024 root@116.122.206.233 "tmux capture-pane -t vast:train -p | tail -10"
```

### 下次繼續事項

- [x] ~~上傳 coco320 資料集到 vast.ai (5.9GB)~~ (已完成)
- [x] ~~在遠端開始第一次訓練測試~~ (已完成)
- [x] ~~找出效能瓶頸~~ (已完成 - ComputeLossOTA)
- [ ] **[進行中]** 使用 `loss_ota: 0` 跑完整訓練 100 epochs
- [ ] 跑 OTA 版本 100 epochs 對比
- [ ] 對比 OTA vs non-OTA 訓練的精度差異 (mAP)

### 訓練指令參考

```bash
# 在 vast.ai 遠端執行
cd /workspace/Yolov7fast

# YOLOv7-Tiny with 320x320 (fastest)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100

# YOLOv7 with 640x640 (standard)
python train.py --data data/coco640.yaml --img 640 --cfg cfg/training/yolov7.yaml --batch-size 32 --epochs 100
```

---

## 變更歷史

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
