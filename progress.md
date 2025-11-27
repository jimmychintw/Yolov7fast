# 專案進度報告

## 目前狀態：YOLOv7-Tiny 320x320 基準訓練完成，準備演算法優化

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
| 2025-11-27 | vast.ai 環境 | RTX 5090 + PyTorch 2.8.0 設定完成 |
| 2025-11-27 | 資料集上傳 | coco320 上傳至 vast.ai 完成 |
| 2025-11-27 | **基準訓練** | YOLOv7-Tiny 320x320 訓練 21 epochs 完成 |
| 2025-11-27 | 結果備份 | 訓練結果備份至 320_Tiny_Original |
| 2025-11-27 | VS Code 設定 | 設定 Remote SSH 連線 vast.ai |
| 2025-11-27 | 備份優化 | 建立 .syncignore 排除大型目錄 |

---

## 基準訓練結果 (Baseline)

### YOLOv7-Tiny 320x320 (21 epochs)

| 指標 | 數值 |
|------|------|
| mAP@.5 | **0.297** |
| mAP@.5:.95 | **0.164** |
| Precision | 0.427 |
| Recall | 0.338 |
| 訓練時間 | ~2.5 小時 |
| GPU | RTX 5090 (32GB) |
| Batch Size | 64 |

### 訓練結果存放位置

```
vast.ai: /workspace/Yolov7fast/runs/train/
├── 320_Tiny_Original/    ← 基準備份 (21 epochs)
│   └── weights/
│       ├── best.pt       (48 MB)
│       └── last.pt       (48 MB)
└── tiny320_scratch/      ← 原始訓練目錄
```

---

## 目前專案結構

```
Yolov7fast/
├── cfg/training/          # 模型架構配置
│   ├── yolov7-tiny-640.yaml
│   ├── yolov7-640.yaml
│   ├── yolov7x-640.yaml
│   ├── yolov7x-480.yaml
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
├── train.py
├── detect.py
├── test.py
├── CLAUDE.md
├── README.md
├── VAST_SETUP.md         # vast.ai 一鍵設定指南
└── progress.md
```

---

## vast.ai 遠端環境

```
SSH: ssh -p 21024 root@116.122.206.233
VS Code: Host "Vast_RTX5090" (已設定在 ~/.ssh/config)
GPU: RTX 5090 (32GB VRAM, Blackwell sm_120)
PyTorch: 2.8.0+cu128
tmux session: vast (4 windows: train, cpu, gpu, terminal)
TensorBoard: http://localhost:6006
```

### SSH Config (~/.ssh/config)
```
Host Vast_RTX5090
  HostName 116.122.206.233
  User root
  Port 21024
  IdentityFile ~/.ssh/id_ed25519
```

---

## 下次繼續事項

- [ ] 開始演算法優化（加速）
- [ ] 比較不同模型架構的效能
- [ ] 測試 YOLOv7 640x640 標準訓練
- [ ] 評估推論速度與準確度的權衡

---

## 變更歷史

### 2025-11-27 (下午)
- 完成 YOLOv7-Tiny 320x320 基準訓練 (21 epochs)
- 訓練結果：mAP@.5 = 0.297, mAP@.5:.95 = 0.164
- 備份訓練結果至 `320_Tiny_Original` 目錄
- 設定 VS Code Remote SSH (`~/.ssh/config` 新增 Vast_RTX5090)
- 建立 `.syncignore` 排除 coco、venv 等大型目錄
- 更新 VAST_SETUP.md 為更完整的設定指南

### 2025-11-27 (上午)
- 租用新 vast.ai instance (RTX 5090)
- 設定 SSH key 連線
- 安裝 PyTorch 2.8.0 + CUDA 12.8 (支援 Blackwell 架構)
- 安裝所有依賴套件
- Clone 專案到 /workspace/Yolov7fast
- 建立 tmux 環境 (train, cpu, gpu, terminal)
- 上傳 coco320 資料集 (~6GB)
- 開始 YOLOv7-Tiny 訓練

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
