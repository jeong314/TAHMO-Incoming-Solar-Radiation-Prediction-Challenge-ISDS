# Environment

Tested environment:

```text
Python: 3.11.9
pandas: 3.0.2
numpy: 2.4.4
OS/shell: Windows PowerShell
```

Project command convention:

```powershell
.\.venv\Scripts\python.exe
```

The reproduction script does not use LightGBM, XGBoost, CatBoost, PyTorch, TensorFlow, or network APIs. It only needs `numpy` and `pandas`.

External-data preprocessing scripts have additional optional dependencies listed
in `../preprocessing/requirements-external.txt`. Those dependencies are not
needed for final-stage v28.0 reproduction.

## Determinism

The final-stage script is deterministic:

- no random sampling
- no model training
- no stochastic library calls
- no public-leaderboard target fitting
- no manual post-submission editing

The only expected outputs are determined by the CSV inputs in this folder.
