

------





# **YOLOv7-Tiny 1B4H Strategy A 後續兩階段解凍微調 SDD**







## **1. 背景與目的**





你已完成 YOLOv7-Tiny、COCO 320x320、1B4H 的 **Strategy A** 長訓（500 epoch），並從分析圖確認：



- 1B4H 的 Obj loss 呈現高平台且趨於水平，mAP 後段增益很小
- 需要在 **不破壞既有訓練流程** 的前提下，提供「續跑 + 分階段解凍」能力
- 目標是讓 Obj loss 重新具備下降斜率、並推動 mAP 突破既有 plateau





本設計將在原 repo 上新增最小侵入的 stage 控制，讓你可以由外部參數指定：



- Stage1：只訓練 **NECK+HEAD**（Backbone 全凍結，BN 保守固定）
- Stage2：再解凍 **LATE BACKBONE**（小 LR + BN 分段策略）





------





## **2. 不可破壞原有功能（硬性需求）**







### **2.1 Backward Compatibility 原則**





- 若 **未指定** **--stage**，train.py 行為必須與原 repo 完全一致（包含 optimizer 分組、freeze、scheduler、resume、DDP 等）。
- 若指定 --stage none（或預設空字串），也必須等價於原始行為。
- 所有新增功能必須「**可插拔**」：只在 --stage 為指定值時作用。
- --resume、--weights、原有 --freeze、原有 --hyp、原有 log、tensorboard/wandb（若有）都必須可照常使用。







### **2.2 不改動核心演算法**





- 不改 Strategy A 的 loss、target matching、資料增強策略（你要延續原先 str A 的結果）
- 本次只處理「requires_grad / BN 模式 / optimizer param groups / warmup restart」這類工程控制





------





## **3. YOLOv7-Tiny 結構分組（本設計固定依此）**





依你實際盤點結果（Detect = model.77，IDetect），定義 top-level index scopes：



- **HEAD**：{74, 75, 76, 77}（最後三個 conv + IDetect）
- **NECK**：{38..73}
- **BACKBONE_LATE**：{23..37}
- **BACKBONE_EARLY**：{0..22}





> 設計上「Stage1 不動 backbone；Stage2 只動 late backbone」。



------





## **4. 功能需求（PRD）**







### **4.1 Stage1：**

### **stage1_neck_tune**





**目標**：只訓練 NECK+HEAD，Backbone 全凍結；Backbone 的 BN 固定 eval。



- Trainable scope：NECK + HEAD

- Frozen scope：BACKBONE_EARLY + BACKBONE_LATE

- BN 策略：Backbone 的 BN eval()（不更新 running stats），並固定 BN 參數（gamma/beta）

- Optimizer param groups：

  

  - head lr = lr0 * 1.0
  - neck lr = lr0 * 0.3
  - backbone 不進 optimizer（requires_grad=False）

  

- Warmup restart：Stage1 續跑時重新 warmup（預設 5 ep）





驗收：



- 510 ep 內確認 trainable params 只在 model.3877
- obj loss 曲線開始出現下降斜率或至少比過去水平更有變化
- mAP50 至少穩定不大退







### **4.2 Stage2：**

### **stage2_late_backbone_tune**





**目標**：解凍 BACKBONE_LATE(23..37) + NECK + HEAD；EARLY backbone 仍凍結；使用小 LR；BN 分兩段策略。



- Trainable scope：HEAD + NECK + BACKBONE_LATE

- Frozen scope：BACKBONE_EARLY

- BN Phase：

  

  - Phase1（前 N epoch，預設 20）：EARLY BN eval、LATE BN eval（求穩）
  - Phase2（N epoch 後）：EARLY BN eval、LATE BN **可選**切回 train（允許適配）

  

- Optimizer param groups：

  

  - head lr = lr0 * 1.0
  - neck lr = lr0 * 0.3
  - late_backbone lr = lr0 * 0.05（保守預設，可調）

  

- Warmup restart：Stage2 續跑時重新 warmup（預設 10 ep）





驗收：



- 5 ep 內確認 trainable 包含 model.23..37
- 30~50 ep 內 obj loss 是否開始往 1B1H 方向下降
- mAP50 是否突破既有 1B4H best（~0.431）





風險控制（若 mAP 跳水）：



1. late_backbone lr_mult: 0.05 → 0.02
2. bn_phase1_epochs: 20 → 40
3. 解凍範圍縮小（先解 31..37）





------





## **5. 系統設計（SDD）**







### **5.1 檔案與模組新增/修改清單**







#### **新增：**

#### **tools/print_trainable.py**



用途：安全檢查與驗收（Stage1/2 必跑）



- 載入模型與 weights
- 印出每個 model.{i} 的 trainable 參數量、總參數量
- 印出 BN 模組數量與目前是否處於 train/eval（可選）





CLI 建議：

```
python tools/print_trainable.py --weights /path/to/weights.pt --device 0
```



#### **新增：**

#### **utils/freeze_by_index.py**



提供最小侵入且可重用的工具：



- set_requires_grad_by_top_indices(model, idx_set, requires_grad: bool)

  

  - 以 named_parameters() 的 name 判斷是否以 model.{idx}. 開頭（需同時兼容 model.model.{idx}.）

  

- set_bn_eval_by_top_indices(model, idx_set, freeze_affine: bool = True)

  

  - 對 named_modules() 找出 BN2d 且 module name 位於指定 top indices 範圍
  - m.eval() 固定 running stats
  - 若 freeze_affine=True：BN 的 weight/bias requires_grad=False

  





兼容性要求：



- 若 repo 的 model 容器不是 model.{idx} 而是 model.model.{idx}，需同時支持兩種 prefix。







#### **新增：**

#### **utils/optimizer_groups.py**



新增一個「以 scope 分組」的 param groups builder，避免大改原 optimizer 形成方式。



建議函式：



- build_stage_param_groups(model, head_idx, neck_idx, late_idx, lr0, weight_decay, mult_dict)

  回傳格式：



```
[
  {"params": head_params, "lr": lr0*mult_head, "weight_decay": wd, "name": "head"},
  {"params": neck_params, "lr": lr0*mult_neck, "weight_decay": wd, "name": "neck"},
  {"params": late_params, "lr": lr0*mult_backbone, "weight_decay": wd, "name": "late_backbone"},
]
```

並附帶印出：



- 每組參數量（numel）與 tensor 數量
- 每組 lr





> 注意：Stage 啟用時用此 builder；Stage 不啟用時沿用原 repo 的 optimizer 建法。





#### **修改：**

#### **train.py**



只允許「最小侵入」改法：



1. **新增 CLI 參數**







- --stage：str，預設 ""（空字串代表不啟用）

  

  - 可選值：stage1_neck_tune, stage2_late_backbone_tune

  

- --lr-mult-head（float, default 1.0）

- --lr-mult-neck（float, default 0.3）

- --lr-mult-backbone（float, default 0.05，Stage2 才用）

- --bn-phase1-epochs（int, default 20，Stage2 才用）

- --bn-unfreeze-late（bool, default False；若 True，Phase2 允許 LATE BN 回到 train）

- --warmup-restart-epochs（int, default：Stage1=5、Stage2=10；可由 stage 自動給預設，也允許 CLI 覆寫）

- --print-stage-summary（bool, default True）：每次開始印 scope、trainable 統計、optimizer groups







1. **在建立 optimizer 前插入 stage hook（僅在 stage 指定時生效）**







- stage1：

  

  - 先將全 model requires_grad=True（避免繼承上次狀態）
  - 凍結 BACKBONE（0..37）
  - 固定 BACKBONE BN eval

  

- stage2：

  

  - EARLY（0..22）凍結
  - LATE（23..37）解凍
  - Phase1：EARLY BN eval + LATE BN eval
  - Phase2：EARLY BN eval +（如果 bn_unfreeze_late=True）LATE BN train

  







1. **stage 模式下的 optimizer 建立**







- stage 模式下，使用 utils/optimizer_groups.py 建立 param groups（head/neck/late）
- 非 stage 模式下，完全沿用原本 optimizer 建立方式







1. **Warmup restart（續跑時重新 warmup）**







- 目標：避免「續跑/解凍」一開始 LR 太大造成崩潰

- 實作建議（最不破壞原本 scheduler）：

  

  - 新增一個係數 warmup_scale(epoch_in_stage)：

    

    - 若 epoch_in_stage < warmup_restart_epochs：線性從 0 → 1
    - 否則 1

    

  - 把每個 param group 的 lr 乘上這個 scale（只在 stage 模式）

  

- epoch_in_stage 的定義：

  

  - 若你是 resume 繼續跑，建議 epoch_in_stage = current_epoch - start_epoch（start_epoch 為本次 run 的起點）
  - 或更簡單：只要 stage 模式，從 0 開始計算（不依賴 resume epoch），以「本次新增訓練段」為準

  







1. **Log 輸出（stage 模式必備）**

   在訓練開始時印出：







- stage 名稱
- scopes index 範圍（HEAD/NECK/LATE/EARLY）
- trainable params 統計（總數、各 scope）
- optimizer groups：name、lr、params numel
- BN 狀態摘要：EARLY/LATE 是否 eval







### **5.2 測試與驗收設計（防止破壞原功能）**







#### **必做：零回歸測試（只要你 repo 有基本跑法）**





1. **不加** **--stage** 跑 1 epoch（或 --epochs 1）







- 驗證：train.py 行為與原本一致（loss、log、optimizer groups 形式不變）







1. **Stage1 dry run**：--stage stage1_neck_tune --epochs 1







- 驗證：只有 model.38~77 trainable







1. **Stage2 dry run**：--stage stage2_late_backbone_tune --epochs 1







- 驗證：model.2377 trainable，model.022 frozen







#### **必做：print_trainable 驗收**





- Stage1/Stage2 開跑前先跑一次 tools/print_trainable.py，把輸出存檔（便於回溯）





------





## **6. 使用方式（你控制 Claude Code 分段實作）**







### **6.1 你要 Claude Code 先做 Stage1 的實作順序**





**第一段實作（只做 Stage1 能跑通，先不做 Stage2）**



- 新增：tools/print_trainable.py
- 新增：utils/freeze_by_index.py
- 新增：utils/optimizer_groups.py（先支援 head/neck 兩組也行）
- train.py 加 --stage stage1_neck_tune 與對應 hook
- Stage1 warmup restart（5 ep）
- 通過 Stage1 dry run







### **6.2 第二段再做 Stage2**





**第二段實作（加入 Stage2 + BN phase）**



- train.py 增加 stage2 hook
- optimizer_groups 支援 late_backbone 第三組
- BN phase 切換（bn_phase1_epochs + bn_unfreeze_late）
- Stage2 warmup restart（10 ep）
- 通過 Stage2 dry run





------





## **7. 你實際要跑的指令範本（延續 Strategy A 的 best.pt）**





> 你說「依據原先 str A 的結果繼續往下做」：因此 weights 要指向 **StrategyA 的 best.pt**（或你指定的 checkpoint）。





### **Stage1（從 StrategyA best.pt 續跑）**



```
python train.py \
  --weights /path/to/StrategyA/best.pt \
  --resume \
  --data data/coco.yaml \
  --img 320 \
  --batch 64 \
  --hyp data/hyp.scratch.tiny.yaml \
  --epochs 50 \
  --stage stage1_neck_tune \
  --lr-mult-head 1.0 \
  --lr-mult-neck 0.3 \
  --warmup-restart-epochs 5 \
  --print-stage-summary
```



### **Stage2（從 Stage1 best.pt 續跑）**



```
python train.py \
  --weights /path/to/stage1/best.pt \
  --resume \
  --data data/coco.yaml \
  --img 320 \
  --batch 64 \
  --hyp data/hyp.scratch.tiny.yaml \
  --epochs 150 \
  --stage stage2_late_backbone_tune \
  --lr-mult-head 1.0 \
  --lr-mult-neck 0.3 \
  --lr-mult-backbone 0.05 \
  --bn-phase1-epochs 20 \
  --bn-unfreeze-late False \
  --warmup-restart-epochs 10 \
  --print-stage-summary
```



------





## **8. Claude Code 要注意的實作細節（避免坑）**





1. **只改 requires_grad 不代表 BN 不動**：一定要明確 BN.eval() 才能固定 running stats
2. named_parameters() 前綴可能是 model.{i}. 或 model.model.{i}.：工具要同時支持
3. stage 模式下 optimizer 只放入 trainable params，避免 frozen params 混入造成誤判
4. warmup restart 不要破壞原 scheduler：用乘法係數最安全
5. --stage 缺省時務必走原邏輯（不要不小心套用新分組）





------



