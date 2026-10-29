# Diabetes Prediction with a Multilayer Perceptron

A binary classifier that predicts whether a patient has diabetes from eight diagnostic measurements. A Keras multilayer perceptron (MLP) is compared against Logistic Regression and Random Forest baselines, first on a held-out test set and then with repeated stratified cross-validation.

## Dataset

[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 patients, 8 numeric features and a binary `Outcome` label (1 = diabetes). Check the license on the dataset page before redistributing the CSV.

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
2. **Missing data:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are physiologically impossible, so they are set to `NaN`. Imputation is done after splitting, inside a pipeline fitted on training data only, so no information from the test set leaks into training.
3. **Split:** 64% train / 16% validation / 20% test, stratified by outcome, with a fixed random seed.
4. **Preprocessing:** mean imputation and standardization, fitted on the training set only. Mean and KNN imputation are compared with cross-validation (see below).
5. **Class imbalance:** balanced class weights computed on the training set only.
6. **Model:** MLP with two hidden layers (ReLU), dropout (0.3), L2 regularization and a sigmoid output. Units, learning rate and L2 strength are tuned with Keras Tuner (random search, 10 trials).
7. **Evaluation:** confusion matrix, classification report, ROC curve and precision-recall curve on the test set.
8. **Baselines:** Logistic Regression and Random Forest trained on the same split and evaluated on the same test set.
9. **Cross-validation:** all three models evaluated with repeated stratified 5-fold cross-validation (5 folds x 3 repeats) on the full dataset, reported as mean ± standard deviation. Imputation and scaling are fitted inside each fold.

## Results

The numbers below come from running the notebook.

**Hold-out test set (154 patients)**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.734 | 0.603 | 0.704 | 0.650 | 0.814 | 0.678 |
| Random Forest | 0.740 | 0.659 | 0.537 | 0.592 | 0.837 | 0.715 |
| MLP | 0.773 | 0.673 | 0.685 | 0.679 | 0.832 | 0.689 |

**Repeated stratified 5-fold cross-validation (mean ± std over 15 folds)**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.754 ± 0.038 | 0.634 ± 0.059 | 0.724 ± 0.054 | 0.673 ± 0.04 | 0.836 ± 0.031 | 0.72 ± 0.057 |
| Random Forest | 0.764 ± 0.033 | 0.694 ± 0.068 | 0.591 ± 0.055 | 0.636 ± 0.049 | 0.831 ± 0.034 | 0.712 ± 0.063 |
| MLP | 0.75 ± 0.039 | 0.61 ± 0.048 | 0.797 ± 0.06 | 0.69 ± 0.042 | 0.833 ± 0.03 | 0.697 ± 0.054 |

**Mean vs KNN imputation (cross-validated):** the two imputers perform practically the same. With Logistic Regression, KNN is 0.001 to 0.003 higher on every metric, and with Random Forest mean imputation is slightly higher on most metrics. All differences are far smaller than the standard deviation (about 0.03 to 0.06), so mean imputation is used because it is simpler.

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

The cross-validation section retrains the MLP in every fold and takes a few minutes. Set `n_repeats=1` in that section for a quicker run.

## Reproducibility

All random seeds (NumPy, TensorFlow, train/test splits, cross-validation folds, Keras Tuner and the baselines) are fixed to 42. Results can still differ slightly across machines and TensorFlow versions.

## Project structure

```
.
├── data/
│   └── diabetes.csv
├── Prediction_Diabetes_with_Multilayer_Perceptron.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- The dataset is small (768 rows), so single-split metrics are noisy. Cross-validation reduces this, but differences of a few points between models may still not be meaningful.
- The MLP hyperparameters were tuned once on a single train/validation split that overlaps with some cross-validation test folds, so the MLP's cross-validated scores may be slightly optimistic. Nested cross-validation would remove this bias.
- All patients in the dataset are women of Pima Indian heritage aged 21 or older, so the results should not be assumed to generalize to other populations.
- This is a learning project and not a clinical tool.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Keras Tuner, matplotlib, seaborn
