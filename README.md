# 🏎️ RaceOutcomePred

> **Can we predict who stands on the podium before the race even starts?**

![F1 Car](https://media1.tenor.com/m/YxQ7dqEdZjIAAAAd/ferrari-sf23.gif)
 
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