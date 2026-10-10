========================================================================
                 Loan Approval Prediction System
========================================================================

Author : Mangesh Thak
Date   : 28/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project evaluates individual machine learning classification models 
alongside ensemble voting mechanisms (Hard and Soft Voting) to predict customer 
loan approvals[cite: 12].

Evaluated Machine Learning Models:
1. Logistic Regression[cite: 12]
2. Decision Tree Classifier[cite: 12]
3. K-Nearest Neighbors (KNN)[cite: 12]
4. Hard Voting Ensemble Classifier[cite: 12]
5. Soft Voting Ensemble Classifier[cite: 12]

The final results and accuracy metrics for each model are output in a 
structured grid format using the `tabulate` library[cite: 12].

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Loan_Approval_Prediction_System_2.py # Main machine learning pipeline[cite: 12]
├── Customer_Loan_Approval.csv           # Input customer loan dataset[cite: 12]
├── requirements.txt                     # Python dependencies list[cite: 12]
└── readme.txt                           # Project documentation

------------------------------------------------------------------------
3. PIPELINE STEP BREAKDOWN
------------------------------------------------------------------------
- Step 1 : Load Dataset            - Reads 'Customer_Loan_Approval.csv'[cite: 12].
- Step 2 : Feature & Target Split  - Separates feature matrix (X) and 
                                     target column (Y: LoanApproved)[cite: 12].
- Step 3 : Train-Test Split        - Splits dataset into 80% training and 
                                     20% testing sets[cite: 12].
- Step 4 : Feature Scaling         - Standardizes features using StandardScaler[cite: 12].
- Step 5 : Train Base Models       - Trains Logistic Regression, Decision Tree, 
                                     and KNN models, recording accuracy[cite: 12].
- Step 6 : Hard Voting Ensemble    - Builds and evaluates Hard VotingClassifier[cite: 12].
- Step 7 : Soft Voting Ensemble    - Builds and evaluates Soft VotingClassifier[cite: 12].
- Step 8 : Display Results         - Renders comparison grid table via `tabulate`[cite: 12].

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8 or higher is installed on your machine.
2. Open terminal/command prompt in the directory containing the files.
3. Install required libraries:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Run the main script using Python:

    python Loan_Approval_Prediction_System_2.py

Note: Make sure 'Customer_Loan_Approval.csv' is placed in the same folder 
prior to running the script[cite: 12].
========================================================================