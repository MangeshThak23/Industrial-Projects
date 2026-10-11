========================================================================
             Loan Default Prediction System using MLP Neural Network
========================================================================

Author : Mangesh Thak
Date   : 16/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project provides an end-to-end Machine Learning pipeline using a 
Multi-Layer Perceptron (MLP) Neural Network (scikit-learn MLPClassifier) 
to predict customer loan default risks. 

Key features include:
- Data pre-processing, categorical encoding, and feature scaling.
- Stratified train-test dataset splitting.
- Base MLP neural network model training.
- Hyperparameter tuning experiments across activation functions, 
  hidden layer architectures, and learning rates.
- Model evaluation (overfitting/underfitting assessment, loss curve plotting, 
  confusion matrix, and classification report).
- Real-time inference capability on new applicant data.

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Loan_Default_4.py      # Main Python pipeline execution script
├── Loan_Default.csv        # Input dataset file
├── requirements.txt       # Python package dependencies
└── README.txt             # Project documentation

------------------------------------------------------------------------
3. PIPELINE FUNCTION BREAKDOWN
------------------------------------------------------------------------
- LoadDataset()            : Loads CSV data, performs binary encoding, 
                             and applies LabelEncoder on categorical features.
- CheckClassAreBalced()    : Verifies class balance ratio for the target variable.
- SplitDataset()           : Prepares features/target, performs stratified 
                             train-test split, and applies StandardScaler.
- ModelCreation()          : Fits a base MLPClassifier neural network.
- exprement()              : Executes grid search experiments for optimal 
                             activation, architecture, and learning rate parameters.
- EvaluateModel()          : Calculates train/test accuracy, checks fit status, 
                             outputs classification metrics, and plots loss curves.
- PredictLoanApplication() : Accepts new applicant data and outputs prediction 
                             probabilities and default verdicts.

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Make sure Python 3.8+ is installed on your system.
2. Open a terminal or command prompt in the project directory.
3. Install the required dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Run the main pipeline script using:

    python Loan_Default_4.py

Ensure 'Loan_Default.csv' is present in the same working directory 
prior to running the program.
========================================================================