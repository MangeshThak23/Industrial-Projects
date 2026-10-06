========================================================================
            Advertising Sales Prediction System (Multiple Linear Regression)
========================================================================

Author : Mangesh Thak
Date   : 24/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project demonstrates an end-to-end Multiple Linear Regression pipeline 
to predict product sales based on advertising expenditures across TV, Radio, 
and Newspaper media channels.

Key pipeline features:
- Data loading and column cleanup (removing index/unnamed columns)
- Missing value inspection and statistical summary generation
- Feature correlation analysis
- Train-test dataset splitting (80% training, 20% testing)
- Model training using Scikit-Learn's LinearRegression
- Performance evaluation using MSE, RMSE, and R2 Score metrics
- Extraction of individual feature coefficients and model intercept

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── LinearRegressionAdvertising8.py   # Main Python regression pipeline script
├── Advertising.csv                   # Input dataset file
├── requirements.txt                  # Required Python dependencies
└── readme.txt                        # Project documentation

------------------------------------------------------------------------
3. PIPELINE STEP BREAKDOWN
------------------------------------------------------------------------
- Step 1  : Load Dataset              - Reads 'Advertising.csv' into a DataFrame.
- Step 2  : Column Cleanup            - Drops unwanted index columns ('Unnamed: 0').
- Step 3  : Missing Value Analysis    - Checks for null/missing values across columns.
- Step 4  : Statistical Summary       - Displays descriptive statistics (describe()).
- Step 5  : Correlation Matrix        - Calculates pairwise correlations among features.
- Step 6  : Feature & Target Split    - Extracts independent (X: TV, radio, newspaper) 
                                        and dependent (Y: sales) variables.
- Step 7  : Train-Test Split          - Splits data into 80% train and 20% test sets.
- Step 8  : Model Fitting             - Trains LinearRegression on training data.
- Step 9  : Inference / Testing       - Generates sales predictions on test set features.
- Step 10 : Evaluation Metrics        - Computes MSE, RMSE, and R2 Score.
- Step 11 : Model Parameters          - Prints TV, Radio, and Newspaper coefficients 
                                        along with the y-intercept.

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
Run the main pipeline script using:

    python LinearRegressionAdvertising8.py

Ensure 'Advertising.csv' is present in the working directory prior to execution.
========================================================================