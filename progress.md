# 專案進度報告

## 目前狀態：訓練環境設定完成，ready to train

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
| 2025-11-26 | vast.ai 設定 | 設定 SSH 連線與 tmux 環境 (RTX 5090) |
| 2025-11-26 | 監控系統 | 建立 GPU/CPU 監控腳本 (/workspace/monitor.csv) |
| 2025-11-26 | 文檔更新 | 重寫 README.md 為簡潔版本 |

### 目前專案結構

```
Yolov7fast/
├── cfg/training/          # 模型架構配置
│   ├── yolov7-tiny.yaml
│   ├── yolov7.yaml
│   └── ...
├── data/                  # 資料集配置
│   ├── coco.yaml         # 原版 COCO (空)
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
├── .claude/commands/     # Slash commands
├── train.py
├── detect.py
├── test.py
├── CLAUDE.md
├── README.md
├── VAST_SETUP.md         # vast.ai 設定文檔
└── progress.md
```

### vast.ai 遠端環境

```
SSH: ssh -p 60002 root@153.198.29.53
GPU: RTX 5090
tmux session: vast (4 windows: train, cpu, gpu, terminal)
監控: /workspace/monitor.csv (每秒記錄 CPU/GPU 使用率)
```

### 下次繼續事項

- [ ] 在 vast.ai 上下載原版 COCO 資料集
- [ ] 準備 coco480 和 coco640 資料集
- [ ] 在遠端開始第一次訓練測試
- [ ] 測試 YOLOv7-Tiny + coco320 組合

### 訓練指令參考

```bash
# YOLOv7-Tiny with 320x320 (fastest)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100

# YOLOv7 with 640x640 (standard)
python train.py --data data/coco640.yaml --img 640 --cfg cfg/training/yolov7.yaml --batch-size 32 --epochs 100
```

---

## 變更歷史

### 2025-11-26
- 修正 COCO 資料集路徑設定（移除硬編碼）
- 確認本地資料集為 320x320 版本
- 重命名 coco → coco320
- 建立多解析度支援：coco.yaml, coco320.yaml, coco480.yaml, coco640.yaml
- 設定 vast.ai SSH 連線（處理 SSH key 問題）
- 建立 tmux 環境（4 個 windows: train, cpu, gpu, terminal）
- 建立 GPU/CPU 監控腳本
- 建立 .claude/commands/ slash commands
- 重寫 README.md 為簡潔版本
- 建立 VAST_SETUP.md 設定文檔

### 2025-11-25
- 專案建立
- GitHub 倉庫初始化：https://github.com/jimmychintw/Yolov7fast
- 匯入 YOLOv7 基礎程式碼（107 個檔案）
- 建立 CLAUDE.md 開發規範（6 條規定）
- 建立 progress.md 進度追蹤機制
