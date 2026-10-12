========================================================================
             Titanic Survival Prediction System
========================================================================

Author : Mangesh Thak
Date   : 20/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project builds a Machine Learning pipeline to predict passenger survival 
on the Titanic using Logistic Regression. The application cleans incoming 
data, performs feature preprocessing and one-hot encoding, trains a binary 
classification model, evaluates performance metrics, and serializes the 
trained model into a reusable `.pkl` file.

------------------------------------------------------------------------
2. DIRECTORY STRUCTURE
------------------------------------------------------------------------
.
├── TitanicLogistic5_PreserveModel_2.py  # Main pipeline execution script
├── MarvellousTitanicDataset.csv         # Input dataset
├── MarvellousTitanic.pkl                # Serialized model output
├── requirements.txt                     # Project dependencies
└── README.txt                           # Project documentation

------------------------------------------------------------------------
3. PIPELINE STAGE BREAKDOWN
------------------------------------------------------------------------
Step 1: LoadData        - Loads dataset from CSV file into a Pandas DataFrame.
Step 2: PreprocessData  - Drops irrelevant features, imputes missing values 
                          (Age/Fare median, Embarked mode), and performs 
                          one-hot encoding.
Step 3: SplitData       - Splits preprocessed data into training (80%) and 
                          testing (20%) sets.
Step 4: TrainModel      - Fits a Logistic Regression model on training data.
Step 5: EvaluateModel   - Computes Accuracy Score and Confusion Matrix.
Step 6: PreserveModel   - Saves trained model to disk as 'MarvellousTitanic.pkl'.

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8+ is installed on your system.
2. Open a terminal/command prompt in the project directory.
3. Install required dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Run the main script using the following command:

    python TitanicLogistic5_PreserveModel_2.py

Ensure 'MarvellousTitanicDataset.csv' is present in the working directory 
before execution.
========================================================================