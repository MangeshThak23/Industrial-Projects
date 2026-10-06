========================================================================
         Customer Segmentation System (K-Means Clustering)
========================================================================

Author : Mangesh Thak
Date   : 25/08/2026

------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------
This project demonstrates an unsupervised machine learning pipeline using 
K-Means Clustering to perform customer segmentation based on annual income 
and spending scores[cite: 2].

Key pipeline features:
- Dataset loading and missing value verification[cite: 2].
- Feature selection (Annual Income and Spending Score)[cite: 2].
- Standard feature scaling using StandardScaler[cite: 2].
- Optimal cluster determination using the Elbow Method (WCSS calculation for k=1 to 10)[cite: 2].
- Elbow curve visualization using Matplotlib[cite: 2].
- Final K-Means model fitting (k=4) and cluster assignment back to the dataset[cite: 2].

------------------------------------------------------------------------
2. FILE STRUCTURE
------------------------------------------------------------------------
.
├── Customer_kmean6.py    # Main Python clustering pipeline script[cite: 2]
├── Mall_Customers.csv    # Input dataset file[cite: 2]
├── requirements.txt      # Project Python dependencies[cite: 2]
└── readme.txt            # Project documentation[cite: 2]

------------------------------------------------------------------------
3. PIPELINE STEP BREAKDOWN
------------------------------------------------------------------------
- Step 1 : Load Data         - Reads 'Mall_Customers.csv' and checks for missing values[cite: 2].
- Step 2 : Feature Selection - Selects 'AnnualIncome' and 'SpendingScore' columns[cite: 2].
- Step 3 : Scale Data        - Normalizes selected features using StandardScaler[cite: 2].
- Step 4 : Elbow Method      - Calculates Within-Cluster Sum of Square (WCSS) across k=1 to 10[cite: 2].
- Step 5 : Visualization     - Plots WCSS values to identify the optimal number of clusters[cite: 2].
- Step 6 : Final Model       - Fits KMeans with k=4 and assigns cluster labels to customer records[cite: 2].

------------------------------------------------------------------------
4. INSTALLATION & SETUP
------------------------------------------------------------------------
1. Ensure Python 3.8+ is installed on your system.
2. Open a terminal or command prompt in the project directory.
3. Install required libraries:

    pip install -r requirements.txt

------------------------------------------------------------------------
5. HOW TO RUN THE PROJECT
------------------------------------------------------------------------
Execute the main script using:

    python Customer_kmean6.py

Ensure 'Mall_Customers.csv' is present in the working directory before running[cite: 2].
========================================================================