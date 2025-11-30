# YOLOv7 1B4H 系統設計書 (SDD)



版本: v1.1 (Phase 1 + Weight Transfer)

日期: 2025-11-30

參考文件: YOLOv7 1B4H PRD v0.5

專案網址: https://github.com/jimmychintw/Yolov7fast

------



## 1. 系統架構概述 (System Architecture)





### 1.1 設計目標



在不破壞 YOLOv7 原有架構的前提下，實作 1B4H (One Backbone, Four Heads) Strategy B。

Phase 1 僅支援 標準語意分類 (Standard Grouping)，並確保訓練與推論流程的正確性。



### 1.2 設計原則 (Design Principles)



1. **開放封閉原則 (OCP)**: 核心邏輯透過繼承或掛勾 (Hook) 擴充，盡量不修改 `models/yolo.py` 與 `utils/loss.py` 的深層邏輯。
2. **設定檔驅動**: 所有的 Head 分配、權重與參數皆由外部 YAML 控制。
3. **策略模式**: Loss 計算與 Head 架構採用模組化設計，支援未來擴充 (如幾何分類、Attention)。



### 1.3 資料流 (Data Flow)



- 訓練階段 (Training):

  Input -> Backbone -> Neck -> MultiHeadDetect (輸出 4 組 Raw Tensors) -> ComputeLossRouter (動態分配標籤並計算總 Loss) -> Backprop

- 推論階段 (Inference):

  Input -> Backbone -> Neck -> MultiHeadDetect (輸出 4 組 Raw Tensors) -> Global Concatenation (拼接為單一張量) -> Standard NMS -> Output

------



## 2. 模組詳細設計 (Module Design)





### 2.1 設定檔解析模組 (Configuration Manager)



**檔案**: `utils/head_config.py`

負責解析定義了 Head 分配規則的 YAML 檔案。

- **類別**: `HeadConfig`
- **屬性**:
  - `head_map (dict)`: `global_class_id` -> `head_id` 的映射表。
  - `local_id_map (dict)`: `global_class_id` -> `local_class_id` (0~19) 的映射表。
  - `weights (list)`: 每個 Head 的 Loss 權重列表。
- **方法**:
  - `__init__(config_path)`: 讀取 YAML 並建立映射表。驗證是否所有 80 類都有被分配，且無重複。
  - `get_head_info(global_id)`: 返回 `(head_id, local_id)`。



### 2.2 多頭檢測模組 (Multi-Head Detector)



**檔案**: `models/multihead.py`

核心檢測層，包含 4 個獨立的檢測頭結構。

- **類別**: `MultiHeadDetect(nn.Module)`
- **初始化參數**:
  - `nc`: 總類別數 (80)。
  - `anchors`: 錨框列表。
  - `ch`: 輸入通道列表 (來自 Neck)。
  - `config_obj`: `HeadConfig` 實例。
- **結構**:
  - `self.heads`: `nn.ModuleList`，長度為 4。每個元素包含一組獨立的 `nn.Conv2d` (如同原生的 `Detect` 模組)。
- **核心方法 `forward(x)`**:
  1. 遍歷 `self.heads`，對輸入特徵 `x` 進行卷積運算。
  2. **訓練模式**: 返回一個列表 `[z0, z1, z2, z3]`，每個 `zi` 是該 Head 的原始輸出。
  3. **推論模式**:
     - 對每個 Head 的輸出進行後處理 (Sigmoid, Grid 敏感度處理)。
     - **關鍵**: 執行 `torch.cat([z0, z1, z2, z3], dim=1)` 將所有預測框在 Anchors 維度合併。
     - 返回 `(concatenated_output, x)`。



### 2.3 損失路由器 (Loss Router)



**檔案**: `utils/loss_router.py`

負責實現「隱式負樣本挖掘」，將標籤分配給正確的 Head，並將其他 Head 視為背景。

- **類別**: `ComputeLossRouter`
- **輸入**: `MultiHeadDetect` 的輸出、Targets (標籤)。
- **邏輯**:
  1. 初始化總 Loss = 0。
  2. 遍歷 4 個 Head：
     - **Filter**: 從 Targets 中篩選出 `HeadConfig` 指定給當前 Head 的標籤。
     - **Remap**: 將篩選出的標籤 `global_class_id` 轉換為 `local_class_id`。
     - **Compute**: 呼叫標準 YOLO Loss 計算邏輯 (Box/Obj/Cls)。
       - *注意*: 若某 Head 無對應標籤，其 Obj Loss 仍需計算 (作為全背景抑制)。
     - **Weighted Sum**: `total_loss += head_loss * head_weight`。
  3. 返回 `total_loss`。



### 2.4 模型建構器擴充 (Model Builder Extension)



**檔案**: `models/yolo.py`

- **修改 `parse_model`**:
  - 新增對模組名稱 `'MultiHeadDetect'` 的支援。
  - 當 YAML 設定檔中出現此模組時，傳入 `head_config` 參數進行實例化。



### 2.5 訓練腳本整合 (Training Script)



**檔案**: `train.py`

- **CLI 參數**:
  - `--head-config` (必要): 指定 Head 分配設定檔路徑
  - `--heads` (選用，預設 4): 指定檢測頭數量
  - `--test-batch-size` (選用): 驗證階段使用的 batch size，解決 1B4H 推論 OOM 問題
- **Loss Factory**:
  - 檢查模型最後一層是否為 `MultiHeadDetect`。
  - 若是，初始化 `ComputeLossRouter` 取代原有的 `ComputeLoss`。

#### 2.5.1 `--test-batch-size` 參數說明

**問題背景**:
1B4H 架構在推論階段需要將 4 個 Head 的輸出進行 `torch.cat` 拼接，記憶體需求約為單 Head 的 4 倍。
當使用大 batch size (如 384) 訓練時，驗證階段會發生 CUDA OOM。

**解決方案**:
新增 `--test-batch-size` 參數，允許訓練和驗證使用不同的 batch size：
- 訓練: 使用 `--batch-size` (如 384)
- 驗證: 使用 `--test-batch-size` (如 64)

**實作位置** (`train.py`):
```python
# 變數提取 (line 47)
test_batch_size = opt.test_batch_size if opt.test_batch_size else batch_size * 2

# testloader 創建 (line 264)
testloader = create_dataloader(..., test_batch_size, ...)

# test.test() 呼叫 (line 441)
results, maps, times = test.test(..., batch_size=test_batch_size, ...)
```

**使用範例**:
```bash
python train.py \
    --batch-size 384 \
    --test-batch-size 64 \
    --heads 4 \
    --head-config data/coco_320_1b4h_standard.yaml \
    ...
```

------



### 2.6 權重遷移模組 (Weight Transfer Utility)



**檔案**: `utils/weight_transfer.py`

負責處理從單頭模型到多頭模型的權重載入。

- **函式**: `load_transfer_weights(model, weights_path, device)`

- **邏輯**:

```python
def load_transfer_weights(model, weights_path, device):
    # 1. 載入 checkpoint
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt['model'].float().state_dict()

    # 2. 獲取當前模型結構
    model_state_dict = model.state_dict()

    # 3. 過濾權重 (Intersect)
    # 只保留名稱相同且形狀完全一致的權重
    # 這會自動過濾掉 Detect Head (因為通道數不同)
    intersect_dict = {k: v for k, v in state_dict.items()
                      if k in model_state_dict and v.shape == model_state_dict[k].shape}

    # 4. 載入
    model.load_state_dict(intersect_dict, strict=False)
    return len(intersect_dict), len(model_state_dict)
```

- **輸出訊息**: `Transferred X/Y items (Backbone + Neck loaded, Head initialized)`



### 2.7 訓練腳本整合 (Training Script) - 權重遷移



**檔案**: `train.py`

- **新增 CLI 參數**: `--transfer-weights` (flag)
- **修改點**:
  - 在模型初始化 (`Model()`) 之後，標準權重載入之前
  - 檢查 `opt.transfer_weights` 參數
  - 若為 True，呼叫 `load_transfer_weights` 取代標準載入流程

------



## 3. 介面與資料格式 (Interfaces & Data)





### 3.1 設定檔格式 (Standard Grouping)



**檔案**: `data/coco_320_1b4h_standard.yaml`

YAML

```
nc: 80
heads: 4
head_assignments:
  head_0:
    classes: [0, 24, 25, ...] # 20 個 ID
    weight: 1.0
  # ... 其他 heads
```

------



## 4. Phase 1 驗收測試計畫 (Verification Plan)



本階段結束前，必須通過以下測試。



### 4.1 單元測試 (Unit Tests)



- **UT-01: Config Mapping**
  - 驗證 Global ID 到 Head ID/Local ID 的映射準確性。
- **UT-02: Router Logic**
  - 驗證標籤遮罩 (Masking) 是否正確，確保 Head 0 不會訓練 Head 1 的標籤。
- **UT-03: Inference Concatenation**
  - 驗證推論時 Tensor 是否正確拼接，總維度是否正確。



### 4.2 整合測試 (Integration Tests)



- **IT-01: 訓練啟動測試**
  - 指令: `python train.py --cfg cfg/training/yolov7-tiny-1b4h-strategy-b.yaml --head-config data/coco_320_1b4h_standard.yaml`
  - 標準: 成功啟動訓練迴圈，無 Crash，Loss 開始下降。
- **IT-02: 推論測試**
  - 使用 IT-01 產出的 `best.pt` 執行 `detect.py`，驗證 NMS 是否正常過濾重疊框。
- **IT-03: 權重遷移載入測試**
  - 輸入: `--weights runs/train/noota_100ep2/weights/best.pt --transfer-weights`
  - 驗證:
    - 控制台顯示 "Transferred X/Y items"
    - X 應接近 Y（約少 10-20 個項目，即 Head 的權重數）
    - 訓練第 1 個 Epoch 的 mAP 應顯著大於 0（例如 > 0.05），證明 Backbone 有效

------



## 附錄 A: 開發者實作指令包 (Developer Implementation Prompt)



*請複製以下內容提供給 Claude Code 或 AI 助手，以執行 Phase 1 的程式碼開發。*

Markdown

```
# Role
You are a Senior Computer Vision Engineer. Your task is to implement **Phase 1** of the "YOLOv7 1B4H (One Backbone Four Heads)" architecture upgrade based on the provided System Design Document (SDD).

# Constraints (CRITICAL)
1. **Open-Closed Principle**: Do NOT modify existing logic in `models/yolo.py` or `utils/loss.py` unless absolutely necessary for hooking. Prefer extending or adding new files.
2. **Compatibility**: The implementation must work with the existing YOLOv7 codebase structure.
3. **Phase 1 Scope**: Implement "Standard Grouping" only. Do not implement Geometry Grouping or RL Augment yet.

# Implementation Tasks

## Task 1: Configuration Manager
Create `utils/head_config.py`.
- **Class**: `HeadConfig`
- **Functionality**: Load a YAML config (e.g., `data/coco_320_1b4h_standard.yaml`) to map COCO 80 classes to 4 specific Heads.
- **Methods**: `get_head_id(class_id)`, `get_head_weight(head_id)`, `get_local_id(global_id)`.

## Task 2: Multi-Head Detector
Create `models/multihead.py`.
- **Class**: `MultiHeadDetect` (inherits from `nn.Module`)
- **Structure**: Must contain `self.heads` (a ModuleList of 4 independent detection heads).
- **Forward Logic**:
    - **Training**: Return a list of raw tensors `[z0, z1, z2, z3]`.
    - **Inference**:
        1. Process outputs (sigmoid, grid sensitivity).
        2. **Crucial**: Concatenate all outputs along dimension 1: `torch.cat([z0, z1, z2, z3], 1)`.
        3. Return `(concatenated_output, x)` to allow standard NMS to handle cross-head duplicates automatically.

## Task 3: Loss Router
Create `utils/loss_router.py`.
- **Class**: `ComputeLossRouter`
- **Logic**:
    - Iterate through the 4 heads.
    - For each head, use `HeadConfig` to create a mask for `targets`.
    - **Implicit Negative Mining**: If a target belongs to Head 1, it is a negative sample (background) for Head 0. Head 0's Obj Loss must still be calculated, but Cls Loss should be skipped or 0 for that target.
    - Remap Global Class IDs (0-79) to Local Class IDs (0-19) before computing loss.
    - Sum weighted losses from all heads.

## Task 4: Integration
1. **Modify `models/yolo.py`**: Update `parse_model` to recognize the module name `'MultiHeadDetect'` and instantiate it using `HeadConfig`.
2. **Modify `train.py`**:
    - Add CLI args: `--heads`, `--head-config`.
    - In the loss instantiation section: If model type is `MultiHeadDetect`, use `ComputeLossRouter` instead of `ComputeLoss`.

## Task 5: Config File
Create `data/coco_320_1b4h_standard.yaml` with the standard grouping (Person, Vehicle, Animal, Indoor) as defined in PRD v0.3.

# Goal
After implementation, I should be able to run the following command to verify Phase 1:
`python train.py --cfg cfg/training/yolov7-tiny-1b4h-strategy-b.yaml --head-config data/coco_320_1b4h_standard.yaml --heads 4`
```