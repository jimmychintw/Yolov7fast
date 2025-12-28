# **PRD：Person-only（1B+1N1H）與 Person-only Half-Width（Neck/Head 0.5x）**







## **1) 目的**





你要一次把兩個模式做進 train.py，用來量化比較：



- **A. Person-only（Full width）**：只訓練 person 一個類別；Backbone 凍結；訓練 500 epoch（由 --epochs 500 控制）
- **B. Person-only + 半頻寬（0.5x Neck/Head）**：同樣只訓練 person，但 **Neck1 & Head1 的通道寬度砍一半**，用來估計「為了未來多專家推理速度」會掉多少 mAP





------





## **2) 你問的「半頻寬要加一層什麼？」**





因為 Backbone 仍輸出原本通道數（例如 P3/P4/P5 各有固定 C），你要讓後面 Neck/Head 變窄，**最穩的做法**是在 Backbone→Neck 的每個輸出 feature map 前面加一個 **1×1 Conv 投影層（Transition / Adapter / Channel Projection）**：



- 對每個尺度特徵（例如 P3、P4、P5）各加一層：

  

  - Conv1x1(C_in → C_out)，其中 C_out = round(C_in * 0.5)

  

- 這些 Adapter 是新加的層（隨機初始化），Backbone 權重可照載、照 freeze

- 然後 Neck/Head 以「半寬通道」運作





> 簡單講：**半頻寬 = 每個 Backbone 輸出前加 1×1 降通道**（必要），再讓 Neck/Head 走縮窄通道。



------





## **3) CLI 需求（train.py 新增選項）**







### **3.1 Person-only 開關**





- --focus-class person（或 0）

  

  - 功能：只保留該 class 的 GT，其餘 class 丟棄
  - 並將該 class **remap 成 class 0**，使模型 **nc=1**

  







### **3.2 Half-Width 開關（只影響 Neck1 & Head1）**





- --nh-width-mult 1.0|0.5（預設 1.0）

  

  - 1.0：person-only full width（A）
  - 0.5：person-only half width（B）

  







### **3.3 Adapter 層開關（只在 nh-width-mult < 1 時生效）**





- --nh-adapter 1x1（預設：自動啟用於 --nh-width-mult 0.5）

  

  - 行為：在每個 Backbone→Neck 的輸入 feature map 前插入 1×1 Conv 降通道

  





> Freeze 仍沿用你原本的 --freeze 行為，不新增 freeze 的語意。



------





## **4) 功能行為（你要量的兩個實驗模式）**







### **Mode A：Person-only（Full width）**





- 啟用條件：--focus-class person 且 --nh-width-mult 1.0

- 行為：

  

  - Dataset labels：只保留 person、並 remap 成 class 0
  - Model：nc=1
  - Backbone：依原本 --freeze 凍結
  - Neck/Head：原通道寬度（不加 adapter）

  







### **Mode B：Person-only + Half-Width（0.5x Neck/Head）**





- 啟用條件：--focus-class person 且 --nh-width-mult 0.5

- 行為：

  

  - Dataset labels：同 Mode A
  - Model：nc=1
  - Backbone：同 Mode A（freeze）
  - **在 Backbone→Neck 的 P3/P4/P5（或你實際用到的尺度）前，各插入 1×1 adapter**
  - Neck/Head：以半寬通道運作（等價只縮窄 Neck1/Head1）

  





------





## **5) 你要的評估輸出（只針對 person）**





每次訓練至少要輸出這些（都只針對 person / class 0）：



- **person AP50**
- **person AP50:95**
- （建議加）AP_small / AP_medium / AP_large（因為縮寬最常傷 small）





此外你要量「為了速度」的代價，所以要附：



- 推論速度：FPS 或 latency（同 batch=1、同 img size、同 device）
- 參數量 / FLOPs（可選，但速度以實測為準）





------





## **6) 你要回答的兩個問題，對應實驗設計**





你要比較的其實就兩個差分：



1. **「只開 person（Mode A）」相對於你原本多類別（baseline）**：

   

   - person AP 提升多少？

   

2. **「person + 半頻寬（Mode B）」相對於 Mode A**：

   

   - 速度提升多少？
   - person AP 掉多少？

   





> PRD 要求：Mode A、Mode B 的訓練設定除 nh-width-mult 外，其餘完全一致（data/imgsz/augment/epochs/freeze/seed）。



------





## **7) 驗收條件（最小）**





- 不開 --focus-class：行為完全不變

- 開 --focus-class person：

  

  - labels 只剩 class 0
  - nc=1 正常訓練/驗證

  

- 開 --nh-width-mult 0.5：

  

  - adapter（1×1）確實插入且通道數減半
  - 訓練/驗證不 crash
  - 可輸出速度與 person AP 指標，能完成 Mode A vs Mode B 比較

  





------