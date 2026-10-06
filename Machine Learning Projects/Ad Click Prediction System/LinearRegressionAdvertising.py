import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#-----------------------------------------------------
#   Function Name : MarvellousRegression
#   Description   : Performs complete multiple linear regression pipeline
#   Input         : CSV File Path (DataPath)
#   Output        : Model evaluation metrics and coefficients
#   Author        : Mangesh Thak
#   Date          : 24/08/2026
#-----------------------------------------------------
def MarvellousRegression(DataPath):
    Border = "-"*40

    #-----------------------------------------------------
    #   Step 1 : Load the data
    #   Description : Read advertising data from CSV file
    #   Input       : DataPath
    #   Output      : DataFrame (df)
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 1 : Load the data")
    print(Border)

    df = pd.read_csv(DataPath)

    print(df.head())

    #-----------------------------------------------------
    #   Step 2 : Remove unwanted columns
    #   Description : Drop index or unnamed columns if present
    #   Input       : DataFrame (df)
    #   Output      : Cleaned DataFrame (df)
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 2 : Remove unwanted columns")
    print(Border)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.head())

    #-----------------------------------------------------
    #   Step 3 : Check missing values
    #   Description : Verify presence of null values in features
    #   Input       : DataFrame (df)
    #   Output      : Null count per column
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 3 : Check missing values")
    print(Border)

    print("Total missing values : ")
    print(Border)
    print(df.isnull().sum())
    print(Border)

    #-----------------------------------------------------
    #   Step 4 : Statistical Summary
    #   Description : Display descriptive statistics of dataset
    #   Input       : DataFrame (df)
    #   Output      : Statistical summary table
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 4 : Statistical Summary")
    print(Border)

    print(df.describe())

    #-----------------------------------------------------
    #   Step 5 : Correlation
    #   Description : Compute pairwise correlation matrix
    #   Input       : DataFrame (df)
    #   Output      : Correlation matrix
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 5 : Correlation")
    print(Border)

    print(df.corr())

    #-----------------------------------------------------
    #   Step 6 : Separate independent and dependent variables
    #   Description : Split features (X) and target variable (Y)
    #   Input       : DataFrame (df)
    #   Output      : Features (X), Target (Y)
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 6 : Separate independent and dependent variables")
    print(Border)

    X = df[["TV","radio","newspaper"]]
    Y = df["sales"]

    print("Independent Variables : ")
    print(X.head())

    print("Dependent Variables : ")
    print(Y.head())

    #-----------------------------------------------------
    #   Step 7 : Split the dataset
    #   Description : Train-test split (80% training, 20% testing)
    #   Input       : Features (X), Target (Y)
    #   Output      : X_train, X_test, Y_train, Y_test
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 7 : Split the dataset")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Training Data : ",X_train.shape)
    print("Testing Data : ",X_test.shape)

    #-----------------------------------------------------
    #   Step 8 : Create & train the model
    #   Description : Fit Multiple Linear Regression model
    #   Input       : Training features (X_train), Target (Y_train)
    #   Output      : Trained LinearRegression model
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 8 : Create & train the model")
    print(Border)

    model = LinearRegression()

    model = model.fit(X_train,Y_train)

    print("Model trained succesfully...")

    #-----------------------------------------------------
    #   Step 9 : Test the model
    #   Description : Make predictions on testing dataset
    #   Input       : Test features (X_test)
    #   Output      : Predicted sales values (Y_pred)
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 9 : Test the model")
    print(Border)

    Y_pred = model.predict(X_test)

    print("Expected answers : ")
    print(Y_test[:3])

    print("Predicted answers : ")
    print(Y_pred[:3])

    #-----------------------------------------------------
    #   Step 10 : Evaluate the model
    #   Description : Calculate MSE, RMSE, and R2 score metrics
    #   Input       : Actual labels (Y_test), Predictions (Y_pred)
    #   Output      : Console output for MSE, RMSE, and R2 score
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 10 : Evaluate the model")
    print(Border)

    MSE = mean_squared_error(Y_test, Y_pred)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(Y_test, Y_pred)

    print("MSE : ",MSE)
    print("RMSE : ",RMSE)
    print("R2 : ",R2)

    #-----------------------------------------------------
    #   Step 11 : Display Coefficient
    #   Description : Print feature coefficients and intercept
    #   Input       : Trained model
    #   Output      : Model coefficients and y-intercept
    #   Author      : Mangesh Thak
    #   Date        : 24/08/2026
    #-----------------------------------------------------
    print(Border)
    print("Step 11 : Display Coefficient")
    print(Border)

    print("TV coefficient : ",model.coef_[0])
    print("Radio coefficient : ",model.coef_[1])
    print("Newspaper coefficient : ",model.coef_[2])

    print("Intercept : ",model.intercept_)

#-----------------------------------------------------
#   Function Name : main
#   Description   : Entry point function
#   Input         : None
#   Output        : None
#   Author        : Mangesh Thak
#   Date          : 24/08/2026
#-----------------------------------------------------
def main():
    MarvellousRegression("Advertising.csv")

if __name__ == "__main__":
    main()