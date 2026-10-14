# Wine Predictor KNN Classifier

A machine learning pipeline utilizing the K-Nearest Neighbors (KNN) algorithm to classify wine cultivars based on chemical composition analysis.

## Author
* **Mangesh Thak**
* **Date:** 08/09/2026

---

## Overview

`WinePredictorKNN6.py` provides an end-to-end data science pipeline performing the following steps:
1. **Data Loading:** Imports wine dataset records from CSV.
2. **Data Cleaning:** Handles missing values and validates dataset dimensions.
3. **Feature Separation:** Separates independent feature attributes from the dependent target column (`Class`).
4. **Train-Test Split:** Performs a stratified 50-50 split for training and testing data.
5. **Feature Scaling:** Standardizes features using `StandardScaler`.
6. **Model Building:** Instantiates a `KNeighborsClassifier` model ($k=9$).
7. **Model Training:** Fits the KNN classifier on scaled training data.
8. **Model Testing:** Evaluates performance and outputs accuracy using `accuracy_score`.

---

## Requirements

The script depends on standard Python data science libraries. You can install them via pip:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

The project expects a CSV file named `WinePredictor.csv` located in the same directory, containing chemical property columns and a target label column titled `Class`.

---

## Usage

Run the script from your terminal:

```bash
python WinePredictorKNN6.py
```