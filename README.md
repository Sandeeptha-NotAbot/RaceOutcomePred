# 🏎️ RaceOutcomePred

> **Can we predict who stands on the podium before the race even starts?**

![F1 Car](https://media.tenor.com/HAQL59Z7D_wAAAAi/formula-racing.gif)
 
Machine learning models that predict Formula 1 podium finishes (P1–P3) using only pre-race data — no live telemetry, no lap times, no cheating.
 
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
 
---

## 🔮 How Does It Predict?
 
Before a race starts, we feed the model features we know *ahead of time* — grid position, qualifying result, and how the driver and constructor have been performing so far that season. The model outputs a probability that the driver finishes in the top 3.
 
**Example:** 2022 Bahrain GP, predicting Leclerc:
 
| Feature | Value |
|---|---|
| Grid position | 1 (pole) |
| Qualifying position | 1 |
| Driver podium rate (season so far) | — (first race, uses season mean) |
| Constructor podium rate (season so far) | — (first race, uses season mean) |
 
→ Model outputs: **high podium probability** ✅ *(he won)*
 
> **Why it's not cheating:** Every feature is available before lights out. No lap times, no live positions, no DNF data — just what teams and analysts would actually know before the race.

---
 
## 📦 Dataset
 
- **Source:** [Formula 1 World Championship — Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **Scope:** 2014–2024 (hybrid-turbo era only)
- **Why not earlier?** Pre-2014 regulation differences make constructor/driver strength non-comparable across eras. The 2026 regulation overhaul is the same reason we stop at 2024.
- **Label:** Binary — `1` = podium (P1–P3), `0` = non-podium
- **Class balance:** ~14.8% podium rate (3 podiums per race × ~20 drivers)
 
---
 