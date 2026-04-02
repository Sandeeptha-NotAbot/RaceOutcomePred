```
RaceOutcomePred/
├── data/
│   └── raw/          # Kaggle CSVs here
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/           # saved model files
├── results/          # metrics, plots
├── requirements.txt
└── README.md
```
```
- Era filtering happens first, so none of the pre-2014 data ever touches the features
- The label (podium) is derived from positionOrder rather than position — this 
matters because position is null for 
drivers who DNF, but positionOrder always has a value
- Qualifying times (q1/q2/q3) are kept as raw strings for now — converting them to 
milliseconds is a job for features.py, 
which is the next script to write
```
```
All models hit ~0.93+ ROC-AUC on the val set
SVM Linear is the best overall model on val (F1: 0.707, ROC-AUC: 0.946)
Logistic Regression L1 and L2 are basically identical, which means L1's sparsity isn't helping
worth noting
```
What the model actually does
Before a race starts, you feed it the features we engineered — things we know ahead of time like grid position, qualifying position, and how the driver and constructor have been performing so far that season. The model outputs a probability: "this driver has a 72% chance of finishing on the podium."
Concrete example
Say it's the 2022 Bahrain GP and you want to predict whether Leclerc will podium:

grid = 1 (he qualified P1)
quali_position = 1
driver_season_podium_rate = 0.0 (no prior races this season yet, filled with season mean)
constructor_season_podium_rate = 0.0 (same)
driver_season_avg_grid = 1.0
teammate_podium_rate_diff = 0.0

The model looks at those numbers, compares them against patterns it learned from 2014–2021 data, and says "historically, drivers with these numbers finish on the podium X% of the time."
Why it's not cheating
The key thing is we only use information available before the race starts — no lap times, no race positions, no DNF info. That's what makes it a genuine prediction rather than just a lookup.
What the model learned
From 2014–2021 it essentially learned things like:

Starting from pole → high podium probability
Constructor has been strong this season → higher probability
Driver has been consistently finishing well → higher probability