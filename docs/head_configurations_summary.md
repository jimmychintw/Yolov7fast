# YOLOv7 1B4H Head 分類配置總結

## 概述

本專案有三種 Head 分類方式，將 COCO 80 類分配到 4 個 Detection Head：

| 配置 | 檔案 | 分類依據 | 特點 |
|------|------|----------|------|
| Standard | `coco_320_1b4h_standard.yaml` | 語意類別 | 每 Head 20 類，平均分配 |
| AntiConfusion | `coco_320_1b4h_anticonfusion.yaml` | 混淆矩陣 | 高混淆對分開，Person 獨立 |
| Geometry | `coco_320_1b4h_geometry.yaml` | 長寬比 K-means | 按物件幾何形狀分組 |

---

## 1. Standard（標準語意分類）

**設計依據**：按物件語意類別分組，每個 Head 負責 20 類

### Head 分配

| Head | 名稱 | 類別數 | 類別 ID |
|------|------|--------|---------|
| Head 0 | 人物與配件 | 20 | 0, 24-28, 31-44 |
| Head 1 | 交通工具與日常物品 | 20 | 1-12, 72-79 |
| Head 2 | 動物與食物 | 20 | 13-23, 45-53 |
| Head 3 | 家具與電子產品 | 20 | 29-30, 54-71 |

### 詳細內容

**Head 0 - 人物與配件 (20 類)**
```
person, backpack, umbrella, handbag, tie, suitcase,
snowboard, sports ball, kite, baseball bat, baseball glove,
skateboard, surfboard, tennis racket, bottle, wine glass,
cup, fork, knife, spoon
```
- 特點：複雜姿態、小物件、紋理多變
- 預期樣本比例：~35%

**Head 1 - 交通工具與日常物品 (20 類)**
```
bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter,
refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush
```
- 特點：剛性物體、規則形狀、金屬質感
- 預期樣本比例：~25%

**Head 2 - 動物與食物 (20 類)**
```
bench, bird, cat, dog, horse, sheep, cow, elephant,
bear, zebra, giraffe, bowl, banana, apple, sandwich,
orange, broccoli, carrot, hot dog, pizza
```
- 特點：可變形、有生命、毛髮紋理
- 預期樣本比例：~20%

**Head 3 - 家具與電子產品 (20 類)**
```
frisbee, skis, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote,
keyboard, cell phone, microwave, oven, toaster, sink
```
- 特點：室內場景、靜態物體、規則形狀
- 預期樣本比例：~20%

---

## 2. AntiConfusion（反混淆分類）

**設計依據**：基於 1B1H 500ep 混淆矩陣分析，將高混淆類別對分到不同 Head

### 設計原則
1. 反混淆優先
2. 樣本數平衡
3. 類別數平衡

### Head 分配

| Head | 名稱 | 類別數 | 樣本數 | 樣本比例 |
|------|------|--------|--------|----------|
| Head 0 | Person_Specialist | **1** | 262,465 | 30.5% |
| Head 1 | AntiConfusion_Group_1 | 26 | 199,806 | 23.2% |
| Head 2 | AntiConfusion_Group_2 | 26 | 198,854 | 23.1% |
| Head 3 | AntiConfusion_Group_3 | 27 | 198,876 | 23.1% |

### 詳細內容

**Head 0 - Person_Specialist (1 類)**
```
person
```
- 獨立處理，佔 30.5% 樣本 (262,465 bbox)
- 避免壓制其他類別

**Head 1 - AntiConfusion_Group_1 (26 類)**
```
car, motorcycle, airplane, parking meter, bench, bird,
horse, elephant, bear, handbag, skis, wine glass,
spoon, sandwich, orange, hot dog, pizza, couch,
potted plant, dining table, laptop, oven, refrigerator,
clock, teddy bear, toothbrush
```

**Head 2 - AntiConfusion_Group_2 (26 類)**
```
bus, boat, traffic light, stop sign, cat, sheep,
zebra, giraffe, snowboard, sports ball, kite,
baseball glove, skateboard, surfboard, cup, knife,
carrot, donut, chair, bed, mouse, cell phone,
microwave, sink, vase, hair drier
```

**Head 3 - AntiConfusion_Group_3 (27 類)**
```
bicycle, train, truck, fire hydrant, dog, cow,
backpack, umbrella, tie, suitcase, frisbee,
baseball bat, tennis racket, bottle, fork, bowl,
banana, apple, broccoli, cake, toilet, tv,
remote, keyboard, toaster, book, scissors
```

### 已驗證的反混淆對

所有高混淆對都已分到不同 Head：

| 混淆對 | Head 分配 |
|--------|-----------|
| car ↔ truck | H1 ↔ H3 ✓ |
| bicycle ↔ motorcycle | H3 ↔ H1 ✓ |
| cat ↔ dog | H2 ↔ H3 ✓ |
| sheep ↔ cow | H2 ↔ H3 ✓ |
| fork ↔ knife | H3 ↔ H2 ✓ |
| knife ↔ spoon | H2 ↔ H1 ✓ |
| wine glass ↔ cup | H1 ↔ H2 ✓ |
| apple ↔ orange | H3 ↔ H1 ✓ |
| chair ↔ couch | H2 ↔ H1 ✓ |
| mouse ↔ remote | H2 ↔ H3 ✓ |
| cell phone ↔ remote | H2 ↔ H3 ✓ |
| microwave ↔ oven | H2 ↔ H1 ✓ |
| skis ↔ snowboard | H1 ↔ H2 ✓ |
| backpack ↔ handbag | H3 ↔ H1 ✓ |
| bus ↔ truck | H2 ↔ H3 ✓ |
| couch ↔ bed | H1 ↔ H2 ✓ |
| horse ↔ cow | H1 ↔ H3 ✓ |

---

## 3. Geometry（幾何形狀分類）

**設計依據**：使用 K-means 依物件長寬比 (aspect ratio) 分組

### Head 分配

| Head | 名稱 | 類別數 | 平均長寬比 | 形狀特徵 |
|------|------|--------|------------|----------|
| Head 0 | Tall (Vertical) | 12 | 0.56 | 高瘦 (垂直) |
| Head 1 | Square (Central) | 28 | 0.93 | 近方形 |
| Head 2 | Square (Central) | 26 | 1.26 | 近方形 |
| Head 3 | Wide (Horizontal) | 14 | 1.92 | 扁寬 (水平) |

### 詳細內容

**Head 0 - Tall/Vertical Group (12 類, Avg Ratio: 0.56)**
```
person, traffic light, fire hydrant, parking meter, giraffe,
handbag, tie, bottle, wine glass, refrigerator, vase, toothbrush
```

**Head 1 - Square/Central Group 1 (28 類, Avg Ratio: 0.93)**
```
bicycle, motorcycle, stop sign, dog, horse, elephant, zebra,
backpack, suitcase, sports ball, baseball bat, baseball glove,
tennis racket, cup, knife, spoon, banana, chair, potted plant,
toilet, tv, cell phone, oven, book, clock, scissors, teddy bear, hair drier
```

**Head 2 - Square/Central Group 2 (26 類, Avg Ratio: 1.26)**
```
car, bus, truck, bird, cat, sheep, cow, bear, kite, skateboard,
fork, bowl, apple, sandwich, orange, broccoli, carrot, hot dog,
donut, cake, couch, laptop, mouse, remote, microwave, toaster
```

**Head 3 - Wide/Horizontal Group (14 類, Avg Ratio: 1.92)**
```
airplane, train, boat, bench, umbrella, frisbee, skis, snowboard,
surfboard, pizza, bed, dining table, keyboard, sink
```

---

## 配置比較

| 面向 | Standard | AntiConfusion | Geometry |
|------|----------|---------------|----------|
| 分類依據 | 語意類別 | 混淆矩陣 | 長寬比 |
| Head 0 類別數 | 20 | 1 (person only) | 12 |
| Head 1 類別數 | 20 | 26 | 28 |
| Head 2 類別數 | 20 | 26 | 26 |
| Head 3 類別數 | 20 | 27 | 14 |
| 平衡性 | 類別數平均 | 樣本數平均 | 形狀特徵 |
| 特殊處理 | 無 | Person 獨立 | 無 |

---

## 實驗結果 (目前使用 AntiConfusion)

AntiConfusion 配置在 1B4H 實驗中表現最佳：
- mAP@0.5: 0.4353 (與 1B1H 持平)
- 成功消除類別間梯度衝突
- Person 獨立 Head 避免樣本不平衡問題

---

## 檔案位置

```
data/
├── coco_320_1b4h_standard.yaml       # 標準語意分類
├── coco_320_1b4h_anticonfusion.yaml  # 反混淆分類 (推薦)
└── coco_320_1b4h_geometry.yaml       # 幾何形狀分類
```
