# DS Engineer Test Task

Solutions to a three-part Data Science Engineer test assignment: an algorithmic task, a regression task on tabular data, and an OOP interface design for digit classification.

## Repository structure

```
.
├── README.md
├── requirements.txt          # pinned Python dependencies
├── .gitignore                # ignores .venv/, __pycache__/, *.py[cod]
├── task1/
│   └── islands.py            # counting islands in a binary matrix
├── task2/
│   ├── data/
│   │   ├── train.csv         # training data (includes the "target" column)
│   │   └── hidden_test.csv   # data to generate predictions for
│   ├── notebooks/
│   │   └── EDA.ipynb         # exploratory data analysis
│   ├── models/
│   │   └── model.joblib      # trained model pipeline
│   ├── predictions/
│   │   └── predictions.csv   # model output for hidden_test.csv
│   ├── features.py           # feature engineering
│   ├── train.py              # training script
│   └── predict.py            # inference script
└── task3/
    └── interfaces.py         # digit classification interface + implementations
```

## Setup

Python 3 is required. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main libraries: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`.

---

## Task 1: Number of islands

`task1/islands.py` counts islands in an `M x N` binary matrix, where `1` is land and `0` is water. Land cells are connected horizontally and vertically (4-directional). The solution uses a recursive depth-first search with a `visited` array, so each island is flood-filled once and counted.

**Usage**

```bash
python task1/islands.py M N <M*N matrix values, row by row>
```

Example: the 2x2 matrix `[[1, 0], [1, 0]]` is passed as:

```bash
python task1/islands.py 2 2 1 0 1 0
```

Prints the number of islands (here, `1`).

Matrix values must be `0` or `1`.

---

## Task 2: Regression

A regression pipeline trained on `task2/data/train.csv` and applied to `task2/data/hidden_test.csv`.

**Approach**

- EDA (see `notebooks/EDA.ipynb`) showed that the target is approximately `col6² + col7`, while the remaining columns behave like noise.
- `features.py` builds exactly those two features: the square of column `"6"` and column `"7"`.
- `train.py` wraps the feature step and a `LinearRegression` model in a scikit-learn `Pipeline`. It reports RMSE on a hold-out split (default 20%, seed 42), then refits on the full data and saves the pipeline with `joblib`.
- `predict.py` loads the saved pipeline and writes predictions to a CSV with a single `target` column.

**Train**

```bash
cd task2
python train.py --train-path data/train.csv --model-path models/model.joblib
```

Options: `--val-size` (default `0.2`), `--seed` (default `42`).

**Predict**

```bash
cd task2
python predict.py --input data/hidden_test.csv \
    --model-path models/model.joblib \
    --output predictions/predictions.csv
```

Run both scripts from inside `task2/`, since `train.py` and `predict.py` import `features.py` and use paths relative to that folder. `predict.py` imports `features` so that `joblib` can load the saved pipeline.

---

## Task 3: Digit classification interfaces

`task3/interfaces.py` defines a common interface for classifying 28x28 grayscale digit images (MNIST-style, shape `(28, 28, 1)`), with several interchangeable implementations behind a single entry point.

**Classes**

- `DigitClassificationInterface`: abstract base class with `preprocess`, `train` and `predict` methods.
- `CNNModel`: wraps an already trained CNN (e.g. a Keras model). Scales pixels to 0-1, adds a batch dimension to get `(1, 28, 28, 1)`, and returns the class with the highest probability.
- `RandomForest`: wraps a scikit-learn `RandomForestClassifier`. Scales pixels to 0-1 and flattens the image to `(1, 784)`. Raises an error if the model has not been fitted.
- `RandomModel`: baseline that center-crops the image to 10x10 and returns a random digit from 0 to 9.
- `DigitClassifier`: single entry point that picks the implementation by name (`"cnn"`, `"rf"` or `"rand"`), validates that the input shape is `(28, 28, 1)`, and always returns an `int` digit.

Training is intentionally not implemented in the classes: `train` raises `NotImplementedError`, and the CNN and random forest are expected to be trained elsewhere and passed in.

**Usage**

```python
from interfaces import DigitClassifier

DigitClassifier("rand").predict(img)
DigitClassifier("rf", model=fitted_rf).predict(img)
DigitClassifier("cnn", model=trained_cnn).predict(img)
```

Running `python task3/interfaces.py` executes a small demo: a random prediction, and a random forest fitted on dummy data to show the `"rf"` path.
