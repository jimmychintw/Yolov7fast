"""
Focus Class - 單類別專注訓練模組

功能：
1. 解析 --focus-class 參數 (class name 或 ID)
2. 過濾 labels 只保留指定類別
3. Remap 類別 ID 為 0

使用場景: Person-only 訓練
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_focus_class(focus_class: str, class_names: list) -> dict:
    """
    解析 focus_class 參數

    Args:
        focus_class: "person" 或 "0" (class ID)
        class_names: 完整類別名稱列表

    Returns:
        dict: {
            'original_id': int,  # 原始類別 ID
            'name': str,         # 類別名稱
            'remap_id': 0        # 永遠 remap 到 0
        }

    Raises:
        ValueError: 如果類別名稱或 ID 無效
    """
    # 嘗試解析為數字
    try:
        original_id = int(focus_class)
        if original_id < 0 or original_id >= len(class_names):
            raise ValueError(f"Invalid class ID: {original_id}. Must be 0-{len(class_names)-1}")
        name = class_names[original_id]
    except ValueError as e:
        if "Invalid class ID" in str(e):
            raise
        # 解析為類別名稱
        name_lower = focus_class.lower()
        class_names_lower = [n.lower() for n in class_names]
        if name_lower not in class_names_lower:
            raise ValueError(f"Unknown class name: '{focus_class}'. Available: {class_names[:10]}...")
        original_id = class_names_lower.index(name_lower)
        name = class_names[original_id]

    return {
        'original_id': original_id,
        'name': name,
        'remap_id': 0
    }


def filter_labels_by_class(labels: list, focus_class_info: dict) -> list:
    """
    過濾 labels 只保留指定類別，並 remap 到 class 0

    Args:
        labels: 原始 labels 列表，每個元素是 ndarray [N, 5] (class_id, x, y, w, h)
        focus_class_info: parse_focus_class 返回的 dict

    Returns:
        filtered_labels: 過濾後的 labels，class 已 remap 為 0
    """
    original_id = focus_class_info['original_id']
    filtered = []

    total_before = 0
    total_after = 0

    for label in labels:
        total_before += len(label)

        if len(label) == 0:
            filtered.append(np.zeros((0, 5), dtype=np.float32))
            continue

        # 只保留目標類別
        mask = label[:, 0] == original_id
        if mask.sum() == 0:
            filtered.append(np.zeros((0, 5), dtype=np.float32))
            continue

        kept = label[mask].copy()
        kept[:, 0] = 0  # Remap to class 0
        filtered.append(kept)
        total_after += len(kept)

    logger.info(f"[Focus-Class] Filtered labels: {total_before} -> {total_after} "
                f"({focus_class_info['name']} only, id {original_id} -> 0)")

    return filtered


def parse_focus_head(head_idx: int, head_config_path: str, class_names: list) -> dict:
    """
    解析 focus_head 參數，從 head_config 讀取類別列表

    Args:
        head_idx: Head 索引 (0-3)
        head_config_path: head config YAML 檔案路徑
        class_names: 完整 COCO 類別名稱列表 (80 類)

    Returns:
        dict: {
            'head_idx': int,         # Head 索引
            'head_name': str,        # Head 名稱
            'original_ids': list,    # 原始 COCO 類別 ID 列表 (保持不變)
            'names': list,           # 類別名稱列表
            'class_count': int,      # 此 Head 的類別數量
        }

    Raises:
        ValueError: 如果 head_idx 無效或配置檔不存在
    """
    import yaml
    from pathlib import Path

    config_path = Path(head_config_path)
    if not config_path.exists():
        raise ValueError(f"Head config file not found: {head_config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 檢查 head_assignments
    if 'head_assignments' not in config:
        raise ValueError(f"head_assignments not found in {head_config_path}")

    head_key = f'head_{head_idx}'
    if head_key not in config['head_assignments']:
        raise ValueError(f"Head {head_idx} not found in config. Available: {list(config['head_assignments'].keys())}")

    head_config = config['head_assignments'][head_key]
    original_ids = head_config['classes']
    head_name = head_config.get('name', f'Head_{head_idx}')

    # 取得類別名稱
    names = [class_names[cid] for cid in original_ids]

    return {
        'head_idx': head_idx,
        'head_name': head_name,
        'original_ids': original_ids,
        'names': names,
        'class_count': len(original_ids),
    }


def filter_labels_by_head(labels: list, focus_head_info: dict) -> list:
    """
    過濾 labels 只保留指定 Head 的類別
    **保持原始 COCO ID**，不做 remap

    Args:
        labels: 原始 labels 列表，每個元素是 ndarray [N, 5] (class_id, x, y, w, h)
        focus_head_info: parse_focus_head 返回的 dict

    Returns:
        filtered_labels: 過濾後的 labels，class ID 保持原始值
    """
    original_ids = set(focus_head_info['original_ids'])
    filtered = []

    total_before = 0
    total_after = 0

    for label in labels:
        total_before += len(label)

        if len(label) == 0:
            filtered.append(np.zeros((0, 5), dtype=np.float32))
            continue

        # 只保留屬於此 Head 的類別 (使用 isin 語意)
        mask = np.array([int(cid) in original_ids for cid in label[:, 0]])
        if mask.sum() == 0:
            filtered.append(np.zeros((0, 5), dtype=np.float32))
            continue

        kept = label[mask].copy()
        # **保持原始 COCO ID，不做 remap**
        filtered.append(kept)
        total_after += len(kept)

    logger.info(f"[Focus-Head] Filtered labels: {total_before} -> {total_after} "
                f"(Head {focus_head_info['head_idx']}: {focus_head_info['head_name']}, "
                f"{focus_head_info['class_count']} classes, original IDs preserved)")

    return filtered


def get_images_with_class(img_files: list, labels: list, focus_class_info: dict) -> tuple:
    """
    獲取包含指定類別的圖像列表

    Args:
        img_files: 圖像檔案路徑列表
        labels: 對應的 labels 列表
        focus_class_info: parse_focus_class 返回的 dict

    Returns:
        tuple: (filtered_img_files, filtered_labels)
    """
    original_id = focus_class_info['original_id']

    filtered_img_files = []
    filtered_labels = []

    for img_file, label in zip(img_files, labels):
        if len(label) == 0:
            continue

        # 檢查是否包含目標類別
        mask = label[:, 0] == original_id
        if mask.sum() > 0:
            kept = label[mask].copy()
            kept[:, 0] = 0  # Remap to class 0
            filtered_img_files.append(img_file)
            filtered_labels.append(kept)

    logger.info(f"[Focus-Class] Images with '{focus_class_info['name']}': "
                f"{len(filtered_img_files)}/{len(img_files)}")

    return filtered_img_files, filtered_labels
