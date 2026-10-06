========================================================================
            Breast Cancer Classification - Ensemble Learning
========================================================================

Author : Mangesh Thak
Date   : 20/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project demonstrates and compares three different Ensemble Machine 
Learning strategies to classify breast cancer diagnostic data:

1. Bagging (Bootstrap Aggregating) - Utilizes BaggingClassifier with 
   DecisionTreeClassifier as the base estimator[cite: 21].
2. Boosting - Utilizes AdaBoostClassifier for adaptive boosting[cite: 22].
3. Soft Voting - Combines predictions from Logistic Regression, Decision 
   Tree, and K-Nearest Neighbors using probability-weighted Soft Voting 
   (VotingClassifier)[cite: 23].

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Ensemble_BreastCancer_Bagging_2.py      # Bagging implementation[cite: 21]
├── Ensemble_BreastCancer_Boosting_2.py     # AdaBoost implementation[cite: 22]
├── Ensemble_BreastCancer_Voting_Soft_2.py # Soft Voting Ensemble implementation[cite: 23]
├── breast_cancer.csv                      # Dataset file[cite: 21, 22, 23]
├── requirements.txt                       # Dependencies[cite: 21, 22, 23]
└── README.txt                             # Project documentation

------------------------------------------------------------------------
3. PIPELINE STAGE BREAKDOWN
------------------------------------------------------------------------
Each script follows a standardized ML pipeline execution workflow:
- Step 1 : Load Dataset - Reads 'breast_cancer.csv' into a DataFrame[cite: 21, 22, 23].
- Step 2 : Separate Features & Labels - Extracts features (X) and target (Y)[cite: 21, 22, 23].
- Step 3 : Split Dataset - Splits data (80% Train, 20% Test)[cite: 21, 22, 23].
- Step 4 : Scale Features - Standardizes features using StandardScaler[cite: 21, 22, 23].
- Step 5 : Create Ensemble Models - Configures Bagging, Boosting, or Voting models[cite: 21, 22, 23].
- Step 6 : Train Model - Fits the selected model on training data[cite: 21, 22, 23].
- Step 7 : Test Model - Makes predictions on test set features[cite: 21, 22, 23].
- Step 8 : Evaluate Model - Outputs Accuracy Score and Confusion Matrix[cite: 21, 22, 23].

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8+ is installed on your machine.
2. Open a terminal or command prompt in the project directory.
3. Install required libraries:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE SCRIPTS
------------------------------------------------------------------------
Run any of the ensemble models using Python:

# Run Bagging Classifier:
python Ensemble_BreastCancer_Bagging_2.py

# Run Boosting (AdaBoost) Classifier:
python Ensemble_BreastCancer_Boosting_2.py

# Run Soft Voting Ensemble Classifier:
python Ensemble_BreastCancer_Voting_Soft_2.py

Note: Ensure 'breast_cancer.csv' is present in the working directory 
before running any script[cite: 21, 22, 23].
========================================================================