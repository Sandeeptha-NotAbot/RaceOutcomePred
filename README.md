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
- The label (podium) is derived from positionOrder rather than position — this matters because position is null for 
drivers who DNF, but positionOrder always has a value
- Qualifying times (q1/q2/q3) are kept as raw strings for now — converting them to milliseconds is a job for features.py, 
which is the next script to write
```
```
All models hit ~0.93+ ROC-AUC on the val set
SVM Linear is the best overall model on val (F1: 0.707, ROC-AUC: 0.946)
Logistic Regression L1 and L2 are basically identical, which means L1's sparsity isn't helping: worth noting
```