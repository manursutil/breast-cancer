# Breast Cancer Classification – ML & API Project

This project uses machine learning and deep learning to classify breast tumors as **benign** or **malignant** based on medical imaging features. It includes comprehensive **EDA**, interpretable and high-performing **ML models**, a **TensorFlow neural network**, and a **REST API** to serve predictions.

## Dataset

- Source: [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- 30 numerical features derived from images of cell nuclei
- Target: `M` = Malignant, `B` = Benign

## Exploratory Data Analysis (EDA)

Key steps:
- Correlation heatmap to detect multicollinearity
- Pairplot to visualize class separability
- Feature selection based on correlation and importance

Insights:
- Strong predictors: `concave points_worst`, `radius_mean`, `area_worst`
- Redundant features removed using correlation threshold

## Machine Learning Models

### Logistic Regression
- High accuracy (96.5%) with reduced features
- Coefficient-based interpretation
- Linear decision boundary

### Random Forest
- Top performer (97.2% accuracy)
- Feature importance used for model transparency
- Handles non-linearity and interactions

## Neural Network (TensorFlow)

- Input: normalized features
- Architecture: Dense layers with ReLU + Dropout
- Output: Sigmoid (binary classification)
- Performance: Comparable to Random Forest

## API for Inference

The API allows real-time predictions using:
- Logistic Regression
- Random Forest
- Neural Network (TensorFlow)

### Endpoints:

POST /predict/logistic

POST /predict/randomforest

POST /predict/neuralnet

## Goals

- Build interpretable and performant models
- Deploy them for real-world usage via API
- Apply data science best practices: EDA → Modeling → Deployment