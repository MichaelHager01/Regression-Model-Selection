# Regression model selection

Selects a polynomial degree, then a ridge penalty, for predicting bearing wear rate.

## What it does

- `src/a2_part1.py` scales the training rows and compares polynomial degrees 1 through 6 with cross-validated RMSE.
- `src/a2_part2.py` models wear rate from rotational speed (RPM), bearing load (newtons), and material hardness (HB).
- The ridge search uses the same degrees and penalties of 0.001, 0.01, 0.1, and 1.
- Both scripts save the chosen pipeline with joblib.

## Layout

- `src/a2_part1.py` — degree selection on the scaled training set
- `src/a2_part2.py` — ridge selection for bearing wear
- `data/train_dataset.csv` — speed, load, hardness, and wear rate

## Run

From the repository root:

```bash
pip install -r requirements.txt
python src/a2_part1.py
python src/a2_part2.py
```

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, joblib. Written for MEEN 423.
