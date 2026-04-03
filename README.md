# 🏎️ RaceOutcomePred

> **Can we predict who stands on the podium?**

![F1 Car](https://media.tenor.com/HAQL59Z7D_wAAAAi/formula-racing.gif)
 
Machine learning models that predict Formula 1 podium finishes (P1–P3) using only pre-race data
 
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
 
---

## 🔮 How Does It Predict?
 
Before a race starts, we feed the model features we know *ahead of time* — grid position, qualifying result, and how the driver and constructor have been performing so far that season. The model outputs a probability that the driver finishes in the top 3.

> **What are we predicting?** Given pre-race data from any race in the 
> 2014–2024 era, the model predicts whether a driver will finish on the 
> podium. The model is validated on 2023–2024 seasons it has never seen 
> during training.
 
**Example:** 2022 Bahrain GP, predicting Leclerc:
 
| Feature | Value |
|---|---|
| Grid position | 1 (pole) |
| Qualifying position | 1 |
| Driver podium rate (season so far) | — (first race, uses season mean) |
| Constructor podium rate (season so far) | — (first race, uses season mean) |
 
→ Model outputs: **high podium probability** ✅ *(he won)*

---
 
## 📦 Dataset
 
- **Source:** [Formula 1 World Championship — Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Scope:** 2014–2024 (hybrid-turbo era only)
- **Why not earlier?** Pre-2014 regulation differences make constructor/driver strength non-comparable across eras. We stop at 2024 because the 2026 regulation overhaul completely reshuffled the power unit and aerodynamic landscape, making historical patterns unreliable for the current season.
- **Label:** Binary — `1` = podium (P1–P3), `0` = non-podium
- **Class balance:** ~14.8% podium rate (3 podiums per race × ~20 drivers)
 
---

## ⚙️ Features
 
| Feature | Description |
|---|---|
| `grid` | Starting grid position |
| `quali_position` | Qualifying classification position |
| `driver_season_podium_rate` | Driver's podium rate across all prior races this season |
| `constructor_season_podium_rate` | Constructor's podium rate across all prior races this season |
| `driver_season_avg_grid` | Driver's average grid position across all prior races this season |
| `teammate_podium_rate_diff` | Driver's podium rate minus their teammate's this season |
 
> All rolling stats use a **shift(1) expanding window** — the current race is never included in its own features to prevent data leakage.
 
---

## 🤖 Models
 
| Model | Type | Notes |
|---|---|---|
| Logistic Regression (L2) | Linear | Baseline classifier |
| Logistic Regression (L1) | Linear | Sparse — good for feature selection |
| SVM Linear | Linear | Maximum margin classifier |
| SVM RBF | Nonlinear | Captures complex decision boundaries |
| Decision Tree | Nonlinear | Interpretable structure |
 
---
 
## 📊 Results
 
**Validation Set (2022 season)**
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression L2 | 0.8705 | 0.5391 | 0.9394 | 0.6851 | 0.9444 |
| Logistic Regression L1 | 0.8705 | 0.5391 | 0.9394 | 0.6851 | 0.9441 |
| **SVM Linear** | **0.8773** | **0.5508** | **0.9848** | **0.7065** | **0.9461** |
| SVM RBF | 0.8591 | 0.5161 | 0.9697 | 0.6737 | 0.9383 |
| Decision Tree | 0.8432 | 0.4885 | 0.9697 | 0.6497 | 0.9282 |
 
**Test Set (2023–2024 seasons)**
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression L2 | 0.8368 | 0.4769 | 0.8986 | 0.6231 | 0.9285 |
| Logistic Regression L1 | 0.8368 | 0.4769 | 0.8986 | 0.6231 | 0.9285 |
| **SVM Linear** | 0.8303 | 0.4669 | 0.9203 | 0.6195 | **0.9299** |
| SVM RBF | 0.8281 | 0.4640 | 0.9348 | 0.6202 | 0.9118 |
| Decision Tree | 0.8150 | 0.4429 | 0.8986 | 0.5933 | 0.9055 |
 
> High recall (~0.93–0.98) means the models catch almost every real podium. Lower precision reflects the inherent difficulty of the class imbalance — only 3 of ~20 drivers podium per race.

### Feature Correlation Heatmap
![Feature Correlation Heatmap](results/feature_correlation_heatmap.png)

### Feature Importance
![Decision Tree Importance](results/importance_decision_tree.png)
![Logistic Regression Importance](results/importance_logreg_l2.png)
 
---

## 🗂️ Repo Structure
 
```
RaceOutcomePred/
├── data/
│   └── raw/          # Kaggle CSVs here (not tracked in git)
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── heatmap.py
├── models/           # saved model files (not tracked in git)
├── results/          # metrics, plots
├── requirements.txt
└── README.md
```
 
---
 
## 🚀 Setup & Usage
 
```bash
# 1. Clone the repo
git clone https://github.com/Sandeeptha-NotAbot/RaceOutcomePred.git
cd RaceOutcomePred
 
# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
Download the Kaggle dataset and place all CSV files in `data/raw/`, then:
 
```bash
# Train all models
python -m src.train
 
# Evaluate and generate plots
python -m src.evaluate
 
# Generate feature correlation heatmap
python -m src.heatmap
```
 
All outputs saved to `results/`.
 
---
 
## 👩‍💻 Team
 
| Name | GitHub |
|---|---|
| Sandeeptha Madan | [@Sandeeptha-NotAbot](https://github.com/Sandeeptha-NotAbot) |
| Evan Sivets | [@boots99](https://github.com/boots99) |
 
*Iowa State University — COMS 474: Introduction to Machine Learning*