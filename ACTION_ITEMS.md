# Action Items - 待執行任務清單

## ⭐ 優先度 1：實作策略 A 的 0.1 軟抑制版本

### 📋 背景說明

**當前實作**（策略 A v1.0）：
- **位置**：`utils/loss_router.py` Line 260-269
- **行為**：完全無視（0.0）其他 Head 的物體位置
- **實作方式**：使用 bool mask 過濾，ignore 的位置完全不計算 obj loss
- **效果**：
  - ✅ Obj Loss 降低約 4%（0.0511 vs 0.0533 @ Epoch 30）
  - ⚠️ mAP 提升很小（+0.0008 @ Epoch 30）
  - ⚠️ 可能導致 Precision 震盪較久

### 🎯 改進方向：0.1 軟抑制

**概念**：
- 不是「完全無視」其他 Head 的物體
- 而是給予「微弱的背景壓力」(target = 0.1)
- 告訴模型：「這裡有東西，但不是我的，稍微壓一下」

**優勢**：
1. **更快穩定 Precision**：避免在這些位置過於自信
2. **減少 False Positive**：10% 的背景壓力抑制誤檢
3. **加速 mAP 成長**：Precision 更快回穩 → mAP 更快噴發

### 🔧 程式碼修改

**檔案**：`utils/loss_router.py`

**當前程式碼**（Line 260-269）：
```python
if self.ignore_other_heads and all_targets is not None:
    # 策略 A：忽略其他 Head 的物體位置
    ignore_mask = self._get_ignore_mask(i, head_preds[i], head_id, all_targets)
    # 只計算「正樣本」和「真背景」的 loss，忽略「其他 Head 的物體」
    valid_positions = ~ignore_mask  # [bs, na, ny, nx]
    obji = self.BCEobj(pi[..., 4][valid_positions], tobj[valid_positions])
else:
    # 原有行為：計算所有位置的 obj loss
    obji = self.BCEobj(pi[..., 4], tobj)
lobj += obji * self.balance[i]
```

**修改為**（0.1 軟抑制版本）：
```python
if self.ignore_other_heads and all_targets is not None:
    # 策略 A v2.0：0.1 軟抑制（取代完全無視）
    ignore_mask = self._get_ignore_mask(i, head_preds[i], head_id, all_targets)

    # 方案 1：使用超參數控制抑制係數（推薦）
    soft_ratio = self.hyp.get('ignore_soft_ratio', 0.1)  # 預設 0.1
    tobj_soft = tobj.clone()
    tobj_soft[ignore_mask] = soft_ratio  # 軟背景壓力
    obji = self.BCEobj(pi[..., 4], tobj_soft)

    # 方案 2：硬編碼 0.1（快速測試用）
    # tobj_soft = tobj.clone()
    # tobj_soft[ignore_mask] = 0.1
    # obji = self.BCEobj(pi[..., 4], tobj_soft)
else:
    # 原有行為：計算所有位置的 obj loss
    obji = self.BCEobj(pi[..., 4], tobj)
lobj += obji * self.balance[i]
```

**新增超參數**（`data/hyp.scratch.tiny.noota.yaml`）：
```yaml
# 策略 A v2.0：軟抑制係數
ignore_soft_ratio: 0.1  # 其他 Head 物體的 objectness target（0.0=完全無視, 0.1=軟抑制, 1.0=完全當負樣本）
```

### 📊 實驗對比方案

建議跑三組對比實驗（各 100 epochs）：

| 實驗 | 策略 | 配置 | 預期 mAP@50 (ep100) |
|------|------|------|---------------------|
| **對照組** | 無策略 A | `--ignore-other-heads` 不開 | 0.4235 (已知) |
| **實驗組 A** | 策略 A v1.0 (0.0) | `--ignore-other-heads` | 0.424-0.425 ❓ |
| **實驗組 B** | 策略 A v2.0 (0.1) | `--ignore-other-heads`<br>`ignore_soft_ratio: 0.1` | 0.425-0.430 ❓ |

### 🎯 成功標準

**策略 A v2.0 成功條件**：
1. Epoch 100 mAP > 0.425（超越對照組 0.4235）
2. Obj Loss 仍然低於對照組（證明停火協議有效）
3. Precision 震盪週期更短（< 20 epochs 穩定）
4. 最終能突破 plateau，達到 0.43+

### 📝 執行檢查清單

執行此 action item 時，請確認：

- [ ] 修改 `utils/loss_router.py` Line 260-269
- [ ] 在 `data/hyp.scratch.tiny.noota.yaml` 新增 `ignore_soft_ratio: 0.1`
- [ ] 測試修改後的程式碼（單元測試）
- [ ] Git commit（標註為策略 A v2.0）
- [ ] 啟動三組對比實驗（對照組、v1.0、v2.0）
- [ ] 記錄實驗結果到 progress.md

### 🔗 相關檔案

- 實作：`utils/loss_router.py`
- 配置：`data/hyp.scratch.tiny.noota.yaml`
- 訓練：`train.py` (已支援 `--ignore-other-heads` 參數)
- 分析：`temp/compare_30epochs_fixed.py`

### 📅 時間記錄

- **提出日期**：2025-12-23
- **當前訓練**：策略 A v1.0 (0.0) 進行中 @ Epoch 30+
- **下次執行時機**：當前訓練完成後，或需要啟動新實驗時

---

## 其他待辦事項

（尚無其他項目）
