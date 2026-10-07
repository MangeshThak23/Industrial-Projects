========================================================================
            Employee Attrition Prediction System (FNN / MLP)
========================================================================

Author : Mangesh Thak
Date   : 16/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project provides a Machine Learning pipeline using a Multi-Layer 
Perceptron (MLP) Neural Network to predict employee attrition risk 
based on workplace and demographic attributes (such as Age, Income, 
Job Satisfaction, and Overtime)[cite: 20].

Key stages of the pipeline:
- Data Ingestion & Exploratory Data Analysis (EDA)[cite: 20]
- Categorical Feature Encoding & Preprocessing[cite: 20]
- Stratified Train-Test Dataset Splitting[cite: 20]
- Feature Standard Scaling[cite: 20]
- MLPClassifier Model Creation & Training[cite: 20]
- Model Evaluation (Overfitting/Underfitting detection & Confusion Matrix)[cite: 20]
- Training Loss Curve Visualization[cite: 20]
- Prediction / Inference on unseen employee records[cite: 20]

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Employee_Attrition.py    # Main pipeline execution script[cite: 20]
├── Employee_Attrition.csv   # Input dataset file[cite: 20]
├── requirements.txt         # Project dependencies[cite: 20]
└── README.txt               # Project documentation[cite: 20]

------------------------------------------------------------------------
3. PIPELINE FUNCTION BREAKDOWN
------------------------------------------------------------------------
- load_data()           : Loads CSV dataset into a pandas DataFrame[cite: 20].
- explore_data()        : Prints dataset dimensions, summary stats, and column details[cite: 20].
- preprocess_data()     : Encodes binary features (OverTime) and labels (Attrition)[cite: 20].
- split_data()          : Splits data into training (80%) and testing (20%) sets[cite: 20].
- scale_features()      : Normalizes numerical features using StandardScaler[cite: 20].
- train_model()         : Configures and trains the MLPClassifier model[cite: 20].
- evaluate_model()      : Calculates training/testing accuracy and confusion matrix[cite: 20].
- plot_loss_curve()     : Displays iteration vs loss trajectory graph[cite: 20].
- predict_unseen_data() : Predicts attrition likelihood on new employee entries[cite: 20].

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
Execute the driver script using:

    python Employee_Attrition.py

Ensure 'Employee_Attrition.csv' is located in the working directory 
prior to running[cite: 20].
========================================================================