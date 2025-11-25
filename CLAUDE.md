# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要規定

1. **所有回覆必須使用繁體中文**
2. **嚴格模組化**：每個模組是完全獨立的 `.py` 檔案
3. **絕對禁止修改已完成模組**：一旦模組完成，不可回頭修改
4. **獨立可測試**：每個模組都必須可以單獨執行和測試
5. **禁止重新寫程式**：所有程式都必須基於現有內容進行優化，不得重新編寫

## Project Overview

YOLOv7 fast implementation project (Python 3.12).

## GitHub

- **使用者名稱**：jimmychintw
- **遠端倉庫**：https://github.com/jimmychintw/Yolov7fast

## Development Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt
```
