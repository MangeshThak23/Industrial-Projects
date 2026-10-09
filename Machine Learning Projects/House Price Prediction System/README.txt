========================================================================
            California Housing Price Prediction System
========================================================================

Author : Mangesh Thak
Date   : 22/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project evaluates single decision tree models versus ensemble bagging 
regressor models to predict housing values using the California housing dataset[cite: 24, 25, 26].

Key implementations included:
1. Individual Decision Tree Regressor - Standalone baseline regression tree[cite: 24].
2. Ensemble Bagging Regressor - Bagging ensemble with 10 Decision Tree estimators[cite: 25, 26].

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Individual_California_DT_2.py      # Standalone Decision Tree script[cite: 24]
├── Ensemble_California_Bagging_2.py   # Bagging Regressor ensemble script[cite: 25]
├── Ensemble_California_Boosting_2.py  # Bagging Regressor ensemble pipeline[cite: 26]
├── california_housing.csv            # Input dataset file[cite: 24, 25, 26]
├── requirements.txt                  # Python dependency list[cite: 24, 25, 26]
└── readme.txt                        # Project documentation

------------------------------------------------------------------------
3. PIPELINE STAGE BREAKDOWN
------------------------------------------------------------------------
Each script runs through a consistent end-to-end Machine Learning pipeline:
- Step 1 : Load Dataset - Reads 'california_housing.csv' into a DataFrame[cite: 24, 25, 26].
- Step 2 : Separate Features & Labels - Splits input features (X) and target (Y)[cite: 24, 25, 26].
- Step 3 : Split Dataset - Splits data into training (80%) and testing (20%) sets[cite: 24, 25, 26].
- Step 4 : Create Model - Initializes base DecisionTreeRegressor or BaggingRegressor[cite: 24, 25, 26].
- Step 5 : Train Model - Fits the model using the training features and labels[cite: 24, 25, 26].
- Step 6 : Test Model - Makes regression predictions on test data[cite: 24, 25, 26].
- Step 7 : Evaluate Model - Measures and outputs Mean Squared Error (MSE) and R2 Score[cite: 24, 25, 26].

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8+ is installed on your machine.
2. Open a terminal or command prompt in the project directory.
3. Install required libraries:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Run any of the desired pipeline scripts from your terminal:

# Run Standalone Decision Tree model:
python Individual_California_DT_2.py

# Run Bagging Regressor model:
python Ensemble_California_Bagging_2.py

Note: Make sure 'california_housing.csv' is present in the working directory 
prior to running[cite: 24, 25, 26].
========================================================================