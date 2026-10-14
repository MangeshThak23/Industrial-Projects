# Marvellous Iris Classification Case Study (Iris_4)

An end-to-end exploratory data analysis and machine learning pipeline utilizing a **Decision Tree Classifier** to categorize Iris flower species based on sepal and petal measurements.

## 📜 Author
* **Mangesh Thak**
* **Date:** 08/09/2026

---

## 📌 Project Overview

`Iris_4.py` executes a complete 9-step machine learning workflow:

1. **Load the Dataset:** Reads the Iris dataset from `iris.csv` into a pandas DataFrame and previews initial entries.
2. **Data Analysis (EDA):** Inspects dataset dimensions, column names, missing values per column, class distribution (`species` count), and statistical summaries.
3. **Feature & Label Separation:** Defines the independent feature matrix (`X`) and dependent target vector (`Y`).
4. **Data Visualization:** Generates a labeled scatter plot comparing `petal length (cm)` vs. `petal width (cm)` grouped by species.
5. **Train-Test Split:** Splits the dataset into 50% training (75 samples) and 50% testing (75 samples) subsets using `random_state=42`.
6. **Build the Model:** Initializes a `DecisionTreeClassifier` with `max_depth=5`.
7. **Train the Model:** Fits the decision tree classifier on the training dataset (`X_train`, `Y_train`).
8. **Test the Model:** Predicts target species labels (`Y_pred`) for the test dataset (`X_test`).
9. **Evaluate Model Performance:** Calculates and prints the model accuracy percentage, confusion matrix, and full classification report.

---

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.x installed. Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Requirements

Place a CSV file named `iris.csv` in the same directory as `Iris_4.py`. The dataset should contain the following columns:
* `sepal length (cm)`
* `sepal width (cm)`
* `petal length (cm)`
* `petal width (cm)`
* `species` (Target variable)

---

## 🚀 Usage

Run the script from your terminal or command prompt:

```bash
python Iris_4.py
```