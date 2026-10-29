# Diabetes Prediction with a Multilayer Perceptron

A binary classifier that predicts whether a patient has diabetes from eight diagnostic measurements, built with a Keras multilayer perceptron (MLP) and compared against Logistic Regression and Random Forest baselines.

## Dataset

[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 patients, 8 numeric features and a binary `Outcome` label (1 = diabetes).

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

## Approach

1. **EDA:** boxplots and class-conditional density plots for each feature. Glucose, BMI and Age show the clearest separation between the two classes.
2. **Missing data:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are physiologically impossible, so they are treated as missing. Mean imputation and KNN imputation were compared, and mean imputation was used (KNN overfit more).
3. **Preprocessing:** standardization (zero mean, unit variance).
4. **Class imbalance:** handled with balanced class weights computed on the training set only.
5. **Split:** 64% train / 16% validation / 20% test.
6. **Model:** MLP with two hidden layers (ReLU), dropout (0.3), L2 regularization and a sigmoid output. Units, learning rate and L2 strength are tuned with Keras Tuner (random search, 10 trials).
7. **Evaluation:** confusion matrix, classification report, ROC curve and precision-recall curve.
8. **Baselines:** Logistic Regression and Random Forest trained on the same split and evaluated on the same test set.

## Results

Test-set results from the final notebook run (fill in after running the notebook; results vary slightly between runs unless the seeds are fixed):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.633 |0.717 |0.673 |0.828 | 0.713|
| Random Forest | 0.766|0.649 |0.698 |0.673 |0.818 |0.654 |
| MLP |0.721 | 0.564| 0.830| 0.830| 0.828| 0.703|

## Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the data
#    Download diabetes.csv from the Kaggle link above and place it in data/

# 4. Launch the notebook
jupyter notebook Prediction_Diabetes_with_Multilayer_Perceptron.ipynb
```

The notebook reads the CSV from a path near the top. Set it to:

```python
df = pd.read_csv("data/diabetes.csv")
```

## Project structure

```
.
├── data/
│   └── diabetes.csv
├── Prediction_Diabetes_with_Multilayer_Perceptron.ipynb
├── requirements.txt
└── README.md
```

## Limitations

- The dataset is small (768 rows), so test-set metrics are noisy. Differences of a few points between models may not be meaningful.
- All patients in the dataset are women of Pima Indian heritage aged 21 or older, so the results should not be assumed to generalize to other populations.
- This is a learning project and not a clinical tool.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Keras Tuner, matplotlib, seaborn
