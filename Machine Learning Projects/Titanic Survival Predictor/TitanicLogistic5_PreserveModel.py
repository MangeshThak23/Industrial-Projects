import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1 : Load Data

#-----------------------------------------------------
#   Function Name : LoadData
#   Description   : Load the data from CSV
#   Input         : Name of csv file (filename)
#   Output        : DataFrame containing loaded data
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset loaded succesfully")
    print(df.head())

    return df

# Step 2 : Data Preprocessing

#-----------------------------------------------------
#   Function Name : PreprocessData
#   Description   : Clean data, fill missing values, and encode categorical variables
#   Input         : DataFrame (df)
#   Output        : Preprocessed DataFrame
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def PreprocessData(df):
    df = df.drop(
        columns=[
            "Passengerid",
            "zero",
            "name"
        ],
        errors="ignore"
    )

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Convert categorical to numeric data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print(df.head())

    print("Data prepsocessing completed")

    return df

# Step 3 : Split Data 

#-----------------------------------------------------
#   Function Name : SplitData
#   Description   : Split dataset into training and testing feature/target sets
#   Input         : DataFrame (df)
#   Output        : Tuple (X_train, X_test, Y_train, Y_test)
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def SplitData(df):
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Dataset Splitting completed succesfully")

    return X_train, X_test, Y_train, Y_test

# Step 4 : Train the model

#-----------------------------------------------------
#   Function Name : TrainModel
#   Description   : Train the LogisticRegression model
#   Input         : Training features (X_train), Training labels (Y_train)
#   Output        : Trained LogisticRegression model instance
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def TrainModel(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)

    model = model.fit(X_train, Y_train)

    print("Model trained succesfully")

    return model

# Step 5 : Evaluate model

#-----------------------------------------------------
#   Function Name : EvaluateModel
#   Description   : Evaluate model using accuracy score and confusion matrix
#   Input         : Trained model, Testing features (X_test), Testing labels (Y_test)
#   Output        : Console output (prints accuracy score and confusion matrix)
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def EvaluateModel(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy is : ", accuracy)

    print(confusion_matrix(Y_test, Y_pred))

# Step 6 : Preserve Model

#-----------------------------------------------------
#   Function Name : PreserveModel
#   Description   : Save trained model to disk as a pickle (.pkl) file
#   Input         : Trained model, Output filename string (filename)
#   Output        : Console output (prints saved filename confirmation)
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def PreserveModel(model, filename):
    joblib.dump(model, filename)

    print("Model preserved with name : ", filename)

#-----------------------------------------------------
#   Function Name : main
#   Description   : Main driver function to execute the full pipeline
#   Input         : None
#   Output        : Pipeline execution results
#   Author        : Mangesh Thak
#   Date          : 20/08/2026
#-----------------------------------------------------
def main():
    # Step 1 
    df = LoadData("MarvellousTitanicDataset.csv")

    # Step 2
    df = PreprocessData(df)

    # Step 3
    X_train, X_test, Y_train, Y_test = SplitData(df)

    # Step 4 : 
    model = TrainModel(X_train, Y_train)

    # Step 5 : 
    EvaluateModel(model, X_test, Y_test)

    # Step 6 : 
    PreserveModel(model, "MarvellousTitanic.pkl")

if __name__ == "__main__":
    main()