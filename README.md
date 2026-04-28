# Predictive-maintenance-for-manufacturing
Predictive maintenance system using ML/DL models to identify high-risk machine failures from sensor data.

## Overview

This project focuses on predicting machine failures in advance using machine learning and deep learning techniques such as using Random Forest, XGBoost, and LSTM to detect potential machine failures from sensor data. By analyzing sensor data, the system identifies high-risk machines and helps prevent unexpected breakdowns.

## Dataset

The dataset contains machine sensor readings such as:

* Air temperature [K]
* Process temperature [K]
* Rotational speed [rpm]
* Torque [Nm]
* Tool wear [min]

Target variables:

* Failure (0 = No Failure, 1 = Failure)
* Failure Type

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* TensorFlow (LSTM)
* SHAP (Explainability)

## Models Implemented

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* Bi-LSTM (Deep Learning Model)

## Model Performance (Test Set - 20%)

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest       | 0.9680   | 0.5179    | 0.8529 | 0.6444   | 0.9829  |
| Gradient Boosting   | 0.9715   | 0.5545    | 0.8235 | 0.6627   | 0.9819  |
| XGBoost             | 0.9545   | 0.4207    | 0.8971 | 0.5728   | 0.9772  |
| Logistic Regression | 0.8710   | 0.1955    | 0.8971 | 0.3211   | 0.9375  |
| Bi-LSTM             | 0.0342   | 0.0342    | 1.0000 | 0.0661   | 0.4987  |

## High-Risk Machine Detection

Machines with failure probability ≥ 0.70 are flagged as high-risk.

* Enables proactive maintenance
* Reduces machine downtime
* Improves operational efficiency

## Key Insights

* Tree-based models (Random Forest, Gradient Boosting) provide the best overall performance
* High recall ensures most failure cases are detected
* Tool wear and torque are strong predictors of machine failure
* Deep learning model (Bi-LSTM) underperformed due to lack of temporal sequence structure in the dataset

## Outputs

All visualizations and results (confusion matrix, ROC curves, SHAP plots) are available in the `outputs/` folder.

## Project Structure

```bash
predictive-maintenance/
│
├── code/
├── dataset/
├── outputs/
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/predictive-maintenance-ml.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook
```

## Author

Aditi Srivastava
