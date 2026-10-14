import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

Border = "-"*30

########################################
# Step 1 : Load the dataset
########################################
#-----------------------------------------------------
#   Block Name  : Load Dataset
#   Description : Read Iris dataset from CSV file and display preview
#   Input       : DataPath ("iris.csv")
#   Output      : DataFrame (df)
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 1 : Load the dataset")
print(Border)

DataPath = "iris.csv"

df = pd.read_csv(DataPath)

print("Dataset loaded succefully")
print("Initial entries from dataset are. : ")
print(df.head())

########################################
# Step 2 : Data Analysis (EDA)
########################################
#-----------------------------------------------------
#   Block Name  : Exploratory Data Analysis
#   Description : Check shape, column list, missing values, class counts, and summary statistics
#   Input       : DataFrame (df)
#   Output      : Statistical summaries printed to console
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 2 : Data Analysis (EDA)")
print(Border)

print("Shape of dataset : ",df.shape)

print("Column names : ",list(df.columns))

print("Mising values per column : ")
print(df.isnull().sum())

print("Class distribution (species count)")
print(df["species"].value_counts())

print("Statistical report of dataset : ")
print(df.describe())

########################################
# Step 3 : Decide Independent & Dependent variables
########################################
#-----------------------------------------------------
#   Block Name  : Feature and Label Separation
#   Description : Define feature column list and extract X (features) and Y (labels)
#   Input       : DataFrame (df)
#   Output      : Feature matrix (X), Label vector (Y)
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 3 : Decide Independent & Dependent variables")
print(Border)

# X : Independent Varibale / Fetures
# Y : Dependenet VAriable / Labels

feture_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
    ]

X = df[feture_cols]
Y = df["species"]

print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)

########################################
# Step 4 : Visualisation of Dataset
########################################
#-----------------------------------------------------
#   Block Name  : Data Visualization
#   Description : Generate scatter plot for petal length vs petal width grouped by species
#   Input       : DataFrame (df)
#   Output      : Scatter plot window
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 4 : Visualisation of Dataset")
print(Border)

# Scatter plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"],label = sp)

plt.title("Marvellous Iris Case Study")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()

########################################
# Step 5 : Split the dataset for training and testing
########################################
#-----------------------------------------------------
#   Block Name  : Train-Test Split
#   Description : Split features and labels into 50% training and 50% testing sets
#   Input       : Feature matrix (X), Label vector (Y)
#   Output      : X_train, X_test, Y_train, Y_test
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, random_state=42)

print("Dataset splitting activity done")

print("X : ",X.shape)       # (150,4)
print("Y : ",Y.shape)       # (150,)

print("X_train : ",X_train.shape)   # (75,4)
print("X_test : ",X_test.shape)     # (75,4)

print("Y_train : ",Y_train.shape)   #(75,)
print("Y_test : ",Y_test.shape)     #(75,)

########################################
# Step 6 : Build the model
########################################
#-----------------------------------------------------
#   Block Name  : Model Initialization
#   Description : Instantiate DecisionTreeClassifier model object with max depth of 5
#   Input       : None
#   Output      : DecisionTreeClassifier model instance
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 6 : Build the model")
print(Border)

model = DecisionTreeClassifier(max_depth=5)

print("Model gets created succesfully")

########################################
# Step 7 : Train the model
########################################
#-----------------------------------------------------
#   Block Name  : Model Training
#   Description : Fit DecisionTreeClassifier using training data
#   Input       : X_train, Y_train
#   Output      : Fitted DecisionTreeClassifier model
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)

print("Model trained succesfully")

########################################
# Step 8 : Test the model
########################################
#-----------------------------------------------------
#   Block Name  : Model Testing
#   Description : Generate predictions for testing dataset using trained model
#   Input       : X_test
#   Output      : Predicted species labels (Y_pred)
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 8 : Test the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model testing done")

print("Expected answers : ")
print(Y_test)

print("Predicted answers : ")
print(Y_pred)

########################################
# Step 9 : Evaluate the mdoel performance
########################################
#-----------------------------------------------------
#   Block Name  : Model Evaluation
#   Description : Compute and display accuracy percentage, confusion matrix, and classification report
#   Input       : Y_test, Y_pred
#   Output      : Console output for performance metrics
#   Author      : Mangesh Thak
#   Date        : 08/09/2026
#-----------------------------------------------------

print(Border)
print("Step 9 : Evaluate the mdoel performance")
print(Border)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model is : ",accuracy*100)

print("Confustion matrix")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

print("Classification Report")
print(classification_report(Y_test, Y_pred))