========================================================================
            Fraudulent Transaction Detection System
========================================================================

Author : Mangesh Thak
Date   : 02/09/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project provides a comprehensive classification benchmark pipeline to 
detect fraudulent financial transactions using multiple machine learning 
algorithms and ensemble methods[cite: 3].

Key models evaluated in the pipeline:
1. Decision Tree Classifier[cite: 3]
2. Random Forest Classifier[cite: 3]
3. Bagging Classifier (with Decision Tree base estimator)[cite: 3]
4. Hard Voting Classifier (combining Logistic Regression, Decision Tree, and KNN)[cite: 3]
5. AdaBoost Classifier[cite: 3]

The results are rendered in a clean comparative grid table showing Accuracy, 
Precision, Recall, F1 Score, and Confusion Matrix values (TN, FP, FN, TP)[cite: 3].

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Fraudulent_Transaction_Detection.py  # Main pipeline script[cite: 3]
├── Fraudulent_Transaction_Detection.csv # Input dataset file[cite: 3]
├── requirements.txt                     # Project dependencies[cite: 3]
└── readme.txt                           # Project documentation

------------------------------------------------------------------------
3. PIPELINE STEP BREAKDOWN
------------------------------------------------------------------------
- Step 1 : Load Dataset             - Reads 'Fraudulent_Transaction_Detection.csv'[cite: 3].
- Step 2 : Feature & Target Split   - Separates feature matrix (X) and target variable (Y: Fraud)[cite: 3].
- Step 3 : Train-Test Split         - Performs an 80/20 train-test split[cite: 3].
- Step 4 : Feature Scaling          - Standardizes features using StandardScaler[cite: 3].
- Step 5 : Model Evaluation         - Fits individual and ensemble classifiers, computing 
                                      Accuracy, Precision, Recall, F1 Score, and Confusion Matrix[cite: 3].
- Step 6 : Display Results          - Renders a comparative evaluation table using `tabulate`[cite: 3].

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8+ is installed on your system.
2. Open a terminal or command prompt in the project directory.
3. Install the required dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Run the main script using:

    python Fraudulent_Transaction_Detection.py

Ensure 'Fraudulent_Transaction_Detection.csv' is in the working directory before execution[cite: 3].
========================================================================