import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#-----------------------------------------------------
#   Function Name : MarvellousClassifier
#   Description   : Pipeline for wine dataset classification using K-Nearest Neighbors
#   Input         : DataPath (string path to CSV file)
#   Output        : Model accuracy printed to console
#   Author        : Mangesh Thak
#   Date          : 08/09/2026
#-----------------------------------------------------
def MarvellousClassifier(DataPath):
    border = "-"*40

    #-----------------------------------------------------
    #   Step 1 : Load the dataset from CSV file
    #   Description : Read CSV data into a pandas DataFrame and preview records
    #   Input       : DataPath ("WinePredictor.csv")
    #   Output      : DataFrame (df)
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------
    print(border)
    print("Step 1 : Load the dataset from CSV file")
    print(border)

    df = pd.read_csv(DataPath)

    print(border)
    print("Some entries from dataset : ")
    print(df.head())
    print(border)

    #-----------------------------------------------------
    #   Step 2 : Clean the dataset
    #   Description : Remove missing values in place and report dataset dimensions
    #   Input       : DataFrame (df)
    #   Output      : Cleaned DataFrame (df)
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------
    print(border)
    print("Step 2 : Clean the dataset")
    print(border)

    df.dropna(inplace= True)

    print("Shape of dataset : ",df.shape)
    
    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])

    print(border)

    #-----------------------------------------------------
    #   Step 3 : Separate independent and dependent variables
    #   Description : Split dataset into feature matrix (X) and target vector (Y)
    #   Input       : DataFrame (df)
    #   Output      : Features (X), Target (Y)
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------

    print(border)
    print("Step 3 : Separate independent and dependent variables")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(border)
    print("Input columns : ",X.columns.tolist())
    print("Output column : Class")
    print(border)

    #-----------------------------------------------------
    #   Step 4 : Split the dataset for tarining and testing
    #   Description : Stratified train-test split (50% training, 50% testing)
    #   Input       : Features (X), Target (Y)
    #   Output      : X_train, X_test, Y_train, Y_test
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------

    print(border)
    print("Step 4 : Split the dataset for tarining and testing")
    print(border)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5,random_state=42,stratify=Y)

    print(border)
    print("Details of training and testing data")

    print("Shape of X_train : ",X_train.shape)
    print("Shape of X_test : ",X_test.shape)

    print("Shape of Y_train : ",Y_train.shape)
    print("Shape of Y_test : ",Y_test.shape)

    print(border)    

    #-----------------------------------------------------
    #   Step 5 : Feture Scaling
    #   Description : Scale training and test features using StandardScaler
    #   Input       : X_train, X_test
    #   Output      : X_train_scaled, X_test_scaled
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------

    print(border)
    print("Step 5 : Feture Scaling") 
    print(border)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feture Scaling done")

    print(border)

    #-----------------------------------------------------
    #   Step 6 : Build the model
    #   Description : Instantiate KNeighborsClassifier model object with n_neighbors=9
    #   Input       : None
    #   Output      : KNeighborsClassifier instance
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------
    print(border)
    print("Step 6 : Build the model")
    print(border)

    model = KNeighborsClassifier(n_neighbors=9)

    print("Classification model is created")

    #-----------------------------------------------------
    #   Step 7 : Train the model
    #   Description : Train KNN model using scaled training data
    #   Input       : X_train_scaled, Y_train
    #   Output      : Trained model object
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------
    model = model.fit(X_train_scaled,Y_train)

    print(border)
    print("Step 7 : Train the model")
    print(border)

    print("Model training completed")

    print(border)

    #-----------------------------------------------------
    #   Step 8 : Test the model
    #   Description : Predict labels on scaled test data and calculate classification accuracy
    #   Input       : X_test_scaled, Y_test
    #   Output      : Printed accuracy percentage
    #   Author      : Mangesh Thak
    #   Date        : 08/09/2026
    #-----------------------------------------------------

    print(border)
    print("Step 8 : Test the model")
    print(border)

    Y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Model accuracy is : ",accuracy*100)

#-----------------------------------------------------
#   Function Name : main
#   Description   : Entry point function
#   Input         : None
#   Output        : None
#   Author        : Mangesh Thak
#   Date          : 08/09/2026
#-----------------------------------------------------
def main():
    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()