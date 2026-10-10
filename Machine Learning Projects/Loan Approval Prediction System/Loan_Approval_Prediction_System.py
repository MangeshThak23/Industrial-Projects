import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from tabulate import tabulate

#---------------------------------------
# Step 1 : Load the dataset
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Load Dataset
#   Description : Read customer loan approval dataset from CSV
#   Input       : CSV File Name ("Customer_Loan_Approval.csv")
#   Output      : DataFrame (df)
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

df = pd.read_csv("Customer_Loan_Approval.csv")

print("Shape of dataset : ", df.shape)

print("First few records : ")
print(df.head())

#---------------------------------------
# Step 2 : Separate features and labels
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Separate Features and Labels
#   Description : Separate feature matrix and target column
#   Input       : DataFrame (df)
#   Output      : Features (X), Target (Y)
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

X = df.drop("LoanApproved", axis=1)
Y = df["LoanApproved"]

print("X shape : ", X.shape)
print("Y shape : ", Y.shape)

#---------------------------------------
# Step 3 : Split dataset for training and testing
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Split Dataset
#   Description : Perform train-test split (80% training, 20% testing)
#   Input       : Features (X), Target (Y)
#   Output      : X_train, X_test, Y_train, Y_test
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#---------------------------------------
# Step 4 : Scale the features
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Feature Scaling
#   Description : Standardize features using StandardScaler
#   Input       : Training features (X_train), Testing features (X_test)
#   Output      : Scaled features (X_train, X_test)
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)  # Use transform only on test data

#---------------------------------------
# Step 5 : Create Individual models and store results
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Train Base Models
#   Description : Train individual Logistic Regression, Decision Tree, and KNN models
#   Input       : X_train, Y_train, X_test, Y_test
#   Output      : Accuracy metrics stored in results list
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

results = []

# Logistic Regression
model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, Y_train)
Y_pred_log = model_log.predict(X_test)
acc_log = accuracy_score(Y_test, Y_pred_log)
results.append(["Logistic Regression",acc_log])

# Decision Tree
model_det = DecisionTreeClassifier(random_state=42)
model_det.fit(X_train, Y_train)
Y_pred_det = model_det.predict(X_test)
acc_det = accuracy_score(Y_test, Y_pred_det)
results.append(["Decision Tree", acc_det])

# KNN
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, Y_train)
Y_pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(Y_test, Y_pred_knn)
results.append(["KNN", acc_knn])

#---------------------------------------
# Hard Voting
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Hard Voting Ensemble
#   Description : Build and evaluate Hard VotingClassifier ensemble model
#   Input       : Base classifiers, X_train, Y_train, X_test
#   Output      : Hard Voting accuracy appended to results
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

modelh = VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression(max_iter=1000)),
        ('decision_tree', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='hard'
)
modelh.fit(X_train, Y_train)
Y_pred_h = modelh.predict(X_test)
acc_hard = accuracy_score(Y_test, Y_pred_h)
results.append(["Hard Voting (Ensemble)", acc_hard])

#---------------------------------------
# Soft Voting
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Soft Voting Ensemble
#   Description : Build and evaluate Soft VotingClassifier ensemble model
#   Input       : Base classifiers, X_train, Y_train, X_test
#   Output      : Soft Voting accuracy appended to results
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

models = VotingClassifier(
    estimators=[
        ('logistic', LogisticRegression(max_iter=1000)),
        ('decision_tree', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft'
)
models.fit(X_train, Y_train)
Y_pred_s = models.predict(X_test)
acc_soft = accuracy_score(Y_test, Y_pred_s)
results.append(["Soft Voting (Ensemble)",acc_soft])

#---------------------------------------
# Print Results in Tabular Format
#---------------------------------------
#-----------------------------------------------------
#   Block Name  : Display Results
#   Description : Render model accuracy comparison table
#   Input       : Results list
#   Output      : Formatted tabulate table in console
#   Author      : Mangesh Thak
#   Date        : 28/08/2026
#-----------------------------------------------------

print("\n")
print(tabulate(results, headers=["Model", "Accuracy"], tablefmt="grid"))