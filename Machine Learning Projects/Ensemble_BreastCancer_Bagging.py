import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix

#-----------------------------------------------------
#   Step 1 : Load the dataset
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Load Dataset
#   Description : Load the breast cancer dataset from CSV
#   Input       : CSV File Name ("breast_cancer.csv")
#   Output      : DataFrame (df)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

df = pd.read_csv("breast_cancer.csv")

print("Shape of dataset : ", df.shape)

print("First few records : ")
print(df.head())

#-----------------------------------------------------
#   Step 2 : Separate fetures and labels
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Separate Features and Labels
#   Description : Separate independent features and target label
#   Input       : DataFrame (df)
#   Output      : Feature matrix (X), Target vector (Y)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

X = df.drop("target", axis=1)
Y = df["target"]

print("X shape : ",X.shape)
print("Y shape : ",Y.shape)

#-----------------------------------------------------
#   Step 3 : Split dataset for traiing and testing
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Split Dataset
#   Description : Split features and labels into training and testing sets
#   Input       : Feature matrix (X), Target vector (Y)
#   Output      : Splitted data (X_train, X_test, Y_train, Y_test)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
                                            X,
                                            Y,
                                            test_size=0.2,
                                            random_state=42
                                            )

#-----------------------------------------------------
#   Step 4 : Scale the fetures
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Feature Scaling
#   Description : Standardize features using StandardScaler
#   Input       : Training features (X_train), Testing features (X_test)
#   Output      : Scaled features (X_train, X_test)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

#-----------------------------------------------------
#   Step 5.1 : Create the Base model
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Create Base Model
#   Description : Initialize base DecisionTreeClassifier model
#   Input       : Hyperparameters (random_state)
#   Output      : DecisionTreeClassifier instance (base_model)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

base_model = DecisionTreeClassifier(random_state=42)

#-----------------------------------------------------
#   Step 5.2 : Create the Bagging model
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Create Bagging Model
#   Description : Create BaggingClassifier ensemble model using base estimator
#   Input       : Base estimator, n_estimators, random_state
#   Output      : BaggingClassifier instance (model)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

model = BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)

#-----------------------------------------------------
#   Step 6 : Train the model
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Train Model
#   Description : Train BaggingClassifier model using scaled training set
#   Input       : Scaled training features (X_train), Training labels (Y_train)
#   Output      : Fitted BaggingClassifier model
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

model = model.fit(X_train,Y_train)

#-----------------------------------------------------
#   Step 7 : Test the model
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Test Model
#   Description : Predict target classes on test dataset using trained model
#   Input       : Scaled test features (X_test)
#   Output      : Predicted values (Y_pred)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

Y_pred = model.predict(X_test)

#-----------------------------------------------------
#   Step 8 : Evalaute the mdoel
#-----------------------------------------------------
#-----------------------------------------------------
#   Block Name  : Evaluate Model
#   Description : Calculate accuracy score and display confusion matrix
#   Input       : Actual labels (Y_test), Predicted labels (Y_pred)
#   Output      : Console output (Accuracy score and Confusion Matrix)
#   Author      : Mangesh Thak
#   Date        : 20/08/2026
#-----------------------------------------------------

print("Accuracy : ",accuracy_score(Y_test,Y_pred))

print("Confustion matrix : ")
print(confusion_matrix(Y_test,Y_pred))