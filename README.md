# Bank-Prediction
Practical exercise #3
# Predicting Bank Term Deposit Subscriptions with Machine Learning

## Project Overview

This project uses real-world data from a Portuguese bank to build and evaluate machine learning models for predicting whether a client will subscribe to a long-term term deposit. The goal is to improve the efficiency of direct marketing campaigns by identifying likely responders.

Using the CRISP-DM framework, I followed a complete pipeline:
- Data cleaning and encoding
- Baseline performance evaluation
- Model training (KNN, Decision Tree, SVM, Logistic Regression)
- Hyperparameter tuning
- Model comparison using accuracy, recall, F1-score, and ROC AUC

---

## Business Objective

The business goal is to develop a predictive model that helps the bank:
- Reduce the number of ineffective marketing calls
- Identify clients who are most likely to subscribe to a term deposit
- Improve campaign efficiency and ROI

To achieve this, I compared multiple machine learning models using relevant performance metrics and training efficiency.

---

## Baseline Model

- The dataset is highly imbalanced: only **~11%** of clients subscribed to the deposit.
- A baseline classifier predicting all “no” yields **~89% accuracy**, but **0% recall** for the “yes” class.
- This highlights the need for better models that focus on identifying positive cases.

---

## Models Trained and Evaluated

| Model              | Training Time | Test Accuracy | Test Recall | Test F1 Score |
|-------------------|----------------|----------------|--------------|----------------|
| Logistic Regression | Fast           | Moderate       | Low–Moderate | Balanced        |
| KNN (Tuned)         | **Very Slow**  | Good           | Good         | Good           |
| Decision Tree (Tuned) | Fast        | Moderate–Good  | Moderate     | Balanced        |
| SVM (Tuned)         | Moderate       | **Best**       | **High**     | **Best**        |



---

## Hyperparameter Tuning

Used `GridSearchCV` with cross-validation to tune:
- KNN: `n_neighbors`, `weights`, `metric`
- Decision Tree: `max_depth`, `min_samples_split`, `criterion`
- SVM: `C`, `gamma`, `kernel`

Models were tuned using F1-score as the primary scoring metric due to class imbalance.

---

## Evaluation Metrics

I evaluated models on:
- Accuracy: overall correctness
- Recall: ability to identify positive (“yes”) cases
- F1-score: balance between precision and recall


> I recommend prioritizing F1-score and recall in future model selection due to the importance of capturing rare positive cases.

---

## Final Recommendations

- Remove features like `sex` that don’t meaningfully impact performance.
- Avoid relying solely on accuracy for evaluation in imbalanced datasets.
- Use SVM (or Decision Tree if interpretability is preferred) as a high-performing model for deployment.
- Use feature importance and sensitivity analysis to guide campaign strategies (e.g., optimal contact timing).

---

## Files Included

- `bank-additional-full.csv`: Dataset used for training and evaluation
- `prompt_III.ipynb`: Full analysis including data prep, modeling, tuning, and evaluation
- `README.md`: Summary of findings and methods

---



This project was completed as part of a machine learning assignment exploring supervised learning, model tuning, and evaluation.

