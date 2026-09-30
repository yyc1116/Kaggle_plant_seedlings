# Plant Seedlings Classification (2018)

[![Python - uv](https://img.shields.io/badge/environment-uv-261230?style=flat&logo=python)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.98614-blue)](https://www.kaggle.com/c/plant-seedlings-classification)
[![Ranking](https://img.shields.io/badge/Estimated%20Rank-Top%205%25--6%25-brightgreen)](https://www.kaggle.com/c/plant-seedlings-classification)

## 專案簡介

本專案以 **PyTorch** 解決 Kaggle 於 2018 年舉辦的 [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) 影像分類競賽（共 12 種植物幼苗類別）。

- **Kaggle Score**：`0.98614`
- **估計排名 (Estimated Ranking)**：約第 **40–53 名 / 833 隊（Top 5%–6%）**
  > *註：排名係對照 2018 年競賽歷史公開排行榜分數推估，非競賽當期官方排名。*
- **核心架構**：`EfficientNetV2-S` 與 `ResNet50` 之多尺度特徵融合（Multi-scale modeling）。
- **關鍵技術**：Data Augmentation（包含 Random Erasing）、Pretrained Backbones、Warmup + Cosine Decay 學習率排程。

---

##  程式碼架構導覽

如果想快速檢視實作細節，可直接點閱對應檔案：

| 檔案 | 說明 | 關鍵實作 |
| :--- | :--- | :--- |
| [`model.py`](./model.py) | 模型架構定義 | `MultiScaleResNet`（384/224/112 三尺度輸入與特徵拼接） |
| [`dataloader.py`](./dataloader.py) | 資料管線與增強 | `transforms.Compose`、RandomErasing、`pin_memory` 傳輸優化 |
| [`train.py`](./train.py) | 訓練迴圈與優化 | Warmup + Cosine Decay 排程器、Checkpoint 定期保存 |
| [`test.py`](./test.py) | 推論與提交檔案生成 | 讀取最佳權重並生成 `predictions.csv` |

---

## 快速開始 (Quick Start)

專案採用 [uv](https://github.com/astral-sh/uv) 進行環境與依賴管理。

### 1. 環境準備
```bash
git clone [https://github.com/yyc1116/Kaggle_plant_seedlings.git](https://github.com/yyc1116/Kaggle_plant_seedlings.git)
cd Kaggle_plant_seedlings
uv sync
```
