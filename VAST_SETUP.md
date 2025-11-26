# Vast.ai 遠端訓練環境設定指南

## 快速設定（一鍵腳本）

租用新 instance 後，依序執行以下步驟：

### Step 1: 添加 SSH Key

1. 取得本機 public key：
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. 在 vast.ai 控制台：
   - Instance → Connect → Manage SSH Keys
   - 貼上 public key → ADD SSH KEY

### Step 2: 設定環境變數

```bash
# 根據 vast.ai 提供的連線資訊修改
export VAST_HOST="root@116.122.206.233"
export VAST_PORT="21024"
```

### Step 3: 一鍵安裝腳本

```bash
# 複製以下內容到終端機執行
ssh -p $VAST_PORT $VAST_HOST -o StrictHostKeyChecking=no 'bash -s' << 'EOF'
set -e
echo "=== 開始設定 vast.ai 環境 ==="

# 1. 升級 pip
echo "[1/5] 升級 pip, setuptools, wheel..."
pip install -U pip setuptools wheel --break-system-packages -q

# 2. 安裝 PyTorch 2.8.0 + CUDA 12.8 (支援 RTX 5090 Blackwell 架構)
echo "[2/5] 安裝 PyTorch 2.8.0 (CUDA 12.8)..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --break-system-packages -q

# 3. 安裝其他依賴
echo "[3/5] 安裝其他依賴套件..."
pip install --break-system-packages -q \
    matplotlib opencv-python Pillow PyYAML requests scipy tqdm \
    tensorboard torch-tb-profiler pandas seaborn ipython psutil thop pycocotools

# 4. Clone 專案
echo "[4/5] Clone YOLOv7fast 專案..."
cd /workspace
if [ ! -d "Yolov7fast" ]; then
    git clone https://github.com/jimmychintw/Yolov7fast.git
fi

# 5. 建立 tmux 環境
echo "[5/5] 建立 tmux 環境..."
tmux kill-server 2>/dev/null || true
tmux new -d -s vast -n train
tmux new-window -t vast -n cpu
tmux new-window -t vast -n gpu
tmux new-window -t vast -n terminal
tmux send-keys -t vast:cpu 'htop' Enter
tmux send-keys -t vast:gpu 'watch -n 1 nvidia-smi' Enter

# 驗證
echo ""
echo "=== 設定完成！驗證環境 ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""
echo "專案位置: /workspace/Yolov7fast"
tmux ls
EOF
```

### Step 4: 啟動 TensorBoard

```bash
ssh -p $VAST_PORT $VAST_HOST "mkdir -p /workspace/Yolov7fast/runs && nohup tensorboard --logdir /workspace/Yolov7fast/runs --port 6006 --bind_all &>/dev/null &"
```

### Step 5: 連線並開始訓練

```bash
# 連線 (含 TensorBoard port forwarding)
ssh -p $VAST_PORT $VAST_HOST -L 6006:localhost:6006

# 進入 tmux
tmux attach -t vast

# 開始訓練
cd /workspace/Yolov7fast
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100
```

TensorBoard: http://localhost:6006

---

## 套件版本（RTX 5090 專用）

| 套件 | 版本 | 說明 |
|------|------|------|
| Python | 3.12 | vast.ai 預裝 |
| PyTorch | 2.8.0+cu128 | 支援 Blackwell (sm_120) |
| torchvision | 0.23.0+cu128 | |
| torchaudio | 2.8.0+cu128 | |
| CUDA | 12.8 | PyTorch wheel 內建 |

**重要**：RTX 5090 使用 Blackwell 架構 (sm_120)，需要 PyTorch 2.8.0+ 和 CUDA 12.8+

---

## 主機規格參考

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| CPU | AMD Ryzen 9 7950X (32 核心) |
| RAM | 124GB+ |
| Disk | 100GB+ |

---

## Tmux 環境

### Session 結構
```
vast (session)
├── train     - 訓練任務
├── cpu       - htop CPU 監控
├── gpu       - nvidia-smi GPU 監控
└── terminal  - 一般操作
```

### 快捷鍵
| 按鍵 | 功能 |
|------|------|
| `Ctrl+b` → `n` | 下一個 window |
| `Ctrl+b` → `p` | 上一個 window |
| `Ctrl+b` → `0-3` | 跳到指定 window |
| `Ctrl+b` → `d` | Detach（離開但不關閉） |

---

## 常用指令

```bash
# 檢查 GPU 狀態
ssh -p $VAST_PORT $VAST_HOST "nvidia-smi"

# 檢查 tmux
ssh -p $VAST_PORT $VAST_HOST "tmux ls"

# 進入 tmux session
ssh -p $VAST_PORT $VAST_HOST -t "tmux attach -t vast"

# 查看訓練 window 輸出
ssh -p $VAST_PORT $VAST_HOST "tmux capture-pane -t vast:train -p | tail -50"
```

---

## 注意事項

1. **SSH Key**：每次租用新 instance 都需要重新添加 SSH key
2. **Instance 重啟**：tmux session 和 TensorBoard 會消失，需重新設定
3. **費用**：記得用完要停止 instance
4. **資料集**：需另外下載或上傳 COCO 資料集到 `/workspace/Yolov7fast/coco320/`

---

## 資料集上傳

```bash
# 從本機上傳 coco320 (約 5.9GB)
rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    /Users/jimmy/Projects/Yolov7fast/coco320/ \
    $VAST_HOST:/workspace/Yolov7fast/coco320/
```

---

*最後更新：2025-11-27*
