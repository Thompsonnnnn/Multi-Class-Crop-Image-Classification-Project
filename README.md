# Multi-Class Crop Image Classification Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 專案簡介
本專案旨在利用深度學習技術進行**多類別作物影像分類**。透過 **ResNet-18** 預訓練模型進行遷移學習 (Transfer Learning) [cite: 2]，並整合了多種先進的訓練策略以提升模型的泛化能力與穩定性。此專案不僅包含核心訓練邏輯，也涵蓋了從原始資料切分到訓練結果視覺化的完整流水線。

## 核心技術亮點
* **資料增強 (Data Augmentation)**: 除了基礎的隨機旋轉與翻轉外，額外實作了 **Mixup** 與 **CutMix** 技術 [cite: 2]，有效抑制過擬合並提升模型對複雜影像的辨識率。
* **混合精度訓練 (AMP)**: 支援 `torch.cuda.amp` [cite: 2]，在保持精度的同時大幅加速訓練速度並節省顯存。
* **完善的評估機制**: 訓練過程會自動產生 **Loss、Accuracy 與 F1-score 曲線圖**，並在測試集上生成 **混淆矩陣 (Confusion Matrix)** 與詳細的分類報告 [cite: 2]。
* **靈活的資料管理**: 內建自動化腳本，將原始影像依 **8:1:1** 比例精確切分為訓練、驗證與測試集 [cite: 1]。

## 專案結構
```text
.
├── src/
│   ├── prepare_dataset.py   # 資料預處理與切分腳本
│   └── train.py             # 核心訓練與評估腳本
├── dataset/                 # (自動生成) 存放分類後的影像資料 [cite: 1]
├── outputs/                 # (自動生成) 存放模型權重與結果圖表 [cite: 2]
├── requirements.txt         # 專案依賴套件
└── README.md
```

## 安裝與環境建置
請確保已安裝 Python 3.8+ 與 CUDA 環境，接著執行：
```bash
pip install -r requirements.txt
```

## 使用說明

### 1. 資料準備
在執行前，請檢查 `src/prepare_dataset.py` 中的路徑設定 [cite: 1]。接著將原始影像放在指定的源目錄，然後執行：
```bash
python src/prepare_dataset.py
```
這會建立符合 `torchvision.datasets.ImageFolder` 格式的資料夾結構 [cite: 1, 2]。

### 2. 模型訓練
執行以下指令開始訓練，可調整參數如 Epochs、Batch Size 與是否開啟 Mixup：
```bash
python src/train.py --data_root ./dataset/images --pretrained --use_amp --use_mixup --epochs 30
```

## 訓練結果視覺化
訓練完成後，`outputs/` 資料夾中將會生成圖表以供分析，包含 Loss 曲線、Accuracy 曲線與混淆矩陣 [cite: 2]。
