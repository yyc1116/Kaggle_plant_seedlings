# Plant Seedlings Classification (2018)
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

## Quick Start

專案使用 [uv](https://github.com/astral-sh/uv) 進行環境管理。

### 下載資料集

1. 依照 [Kaggle API 文件](https://www.kaggle.com/docs/api) 完成登入與 API 設定。
2. 下載競賽資料集：  
`kaggle competitions download -c plant-seedlings-classification`
3. 解壓縮資料集：  
`unzip plant-seedlings-classification.zip -d plant-seedlings-classification`

### 訓練與推論

1. 訓練模型：`uv run train.py`
2. 產生預測結果：`uv run test.py`
