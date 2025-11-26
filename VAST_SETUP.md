# Vast.ai 遠端訓練環境設定指南

## 主機資訊

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| CPU | AMD Ryzen 9 7950X (32 核心) |
| RAM | 188GB |
| Disk | 200GB |
| CUDA | 13.0 |
| Driver | 580.95.05 |

## SSH 連線

### 連線指令
```bash
ssh -p 60002 root@153.198.29.53
```

### SSH Key 設定

1. **本機 public key 位置**：`~/.ssh/id_ed25519.pub`

2. **vast.ai 設定**：
   - 登入 vast.ai → Account → Keys → SSH Keys
   - 加入 public key

3. **重要**：若 vast.ai 同步有問題，手動在遠端加入：
   ```bash
   echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys
   ```

## Tmux 環境

### Session 結構
```
vast (session)
├── train     - 訓練任務
├── cpu       - htop CPU 監控
├── gpu       - nvidia-smi GPU 監控
└── terminal  - 一般操作
```

### 建立指令
```bash
# 建立 session 和 windows
ssh -p 60002 root@153.198.29.53 "tmux new -d -s vast -n train"
ssh -p 60002 root@153.198.29.53 "tmux new-window -t vast -n cpu"
ssh -p 60002 root@153.198.29.53 "tmux new-window -t vast -n gpu"
ssh -p 60002 root@153.198.29.53 "tmux new-window -t vast -n terminal"

# 啟動監控
ssh -p 60002 root@153.198.29.53 "tmux send-keys -t vast:cpu 'htop' Enter"
ssh -p 60002 root@153.198.29.53 "tmux send-keys -t vast:gpu 'watch -n 1 nvidia-smi' Enter"
```

### 常用操作
```bash
# 進入 tmux session
ssh -p 60002 root@153.198.29.53 -t "tmux attach -t vast"

# 查看特定 window 輸出
ssh -p 60002 root@153.198.29.53 "tmux capture-pane -t vast:gpu -p"
ssh -p 60002 root@153.198.29.53 "tmux capture-pane -t vast:cpu -p"

# 在特定 window 執行指令
ssh -p 60002 root@153.198.29.53 "tmux send-keys -t vast:terminal 'ls -la' Enter"
```

### Tmux 快捷鍵（進入 session 後）
| 按鍵 | 功能 |
|------|------|
| `Ctrl+b` → `n` | 下一個 window |
| `Ctrl+b` → `p` | 上一個 window |
| `Ctrl+b` → `0-3` | 跳到指定 window |
| `Ctrl+b` → `d` | Detach（離開但不關閉） |

## 背景監控系統

### 監控內容
| 欄位 | 說明 |
|------|------|
| timestamp | ISO 格式時間戳 |
| cpu_pct | CPU 使用率 % |
| ram_gb | RAM 使用量 GB |
| gpu_pct | GPU 使用率 % |
| vram_mb | VRAM 使用量 MB |
| gpu_temp | GPU 溫度 °C |
| gpu_power | GPU 功耗 W |

### Log 檔案位置
```
/workspace/monitor.csv
```

### 啟動監控腳本
```bash
ssh -p 60002 root@153.198.29.53 'echo "timestamp,cpu_pct,ram_gb,gpu_pct,vram_mb,gpu_temp,gpu_power" > /workspace/monitor.csv && nohup bash -c "while true; do
  CPU=\$(top -bn1 | grep \"Cpu(s)\" | awk \"{print \\\$2}\" | cut -d\"%\" -f1)
  RAM=\$(free -g | awk \"/Mem:/{print \\\$3}\")
  GPU_INFO=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits)
  GPU_PCT=\$(echo \$GPU_INFO | cut -d, -f1 | tr -d \" \")
  VRAM=\$(echo \$GPU_INFO | cut -d, -f2 | tr -d \" \")
  TEMP=\$(echo \$GPU_INFO | cut -d, -f3 | tr -d \" \")
  POWER=\$(echo \$GPU_INFO | cut -d, -f4 | tr -d \" \")
  echo \"\$(date +%Y-%m-%dT%H:%M:%S),\$CPU,\$RAM,\$GPU_PCT,\$VRAM,\$TEMP,\$POWER\" >> /workspace/monitor.csv
  sleep 1
done" &>/dev/null &'
```

### 查看監控資料
```bash
# 最近 10 筆
ssh -p 60002 root@153.198.29.53 "tail -10 /workspace/monitor.csv"

# 下載到本地分析
scp -P 60002 root@153.198.29.53:/workspace/monitor.csv ./
```

## 快速檢查指令

```bash
# 檢查 GPU 狀態
ssh -p 60002 root@153.198.29.53 "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv"

# 檢查 CPU 和記憶體
ssh -p 60002 root@153.198.29.53 "free -h && nproc"

# 檢查 tmux sessions
ssh -p 60002 root@153.198.29.53 "tmux ls"

# 即時監控 (10 秒)
ssh -p 60002 root@153.198.29.53 'for i in {1..10}; do
  echo "$(date +%H:%M:%S) CPU: $(top -bn1 | grep "Cpu(s)" | awk "{print \$2}")% GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
  sleep 1
done'
```

## 注意事項

1. **SSH Key 同步問題**：vast.ai 有時無法正確同步 SSH key 到 instance，需手動加入 `authorized_keys`

2. **Instance 重啟**：重啟後 tmux session 會消失，需重新建立

3. **監控腳本**：nohup 背景執行，instance 重啟後需重新啟動

4. **費用**：$0.472/hr，記得用完要停止 instance

---

*建立日期：2025-11-25*
