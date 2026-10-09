import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------------------------------------
# Stpe 1 : Load the data 
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Load Dataset
#   Description : Load the California housing dataset from CSV
#   Input       : CSV File Name ("california_housing.csv")
#   Output      : DataFrame (df)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

df = pd.read_csv("california_housing.csv")
print("Shape of Dataset : ",df.shape)
print("First few records : ",df.head())

#-------------------------------------
# Step 2 : Separate Fetures and Labels
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Separate Features and Labels
#   Description : Separate independent features and target label
#   Input       : DataFrame (df)
#   Output      : Feature matrix (X), Target vector (Y)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

X = df.drop("target",axis=1)
Y = df["target"]

print("Shape of X : ",X.shape)
print("Shape of Y : ",Y.shape)

#-------------------------------------
# Step 3 : Split dataset for traiing and testing
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Split Dataset
#   Description : Split features and labels into training and testing sets
#   Input       : Feature matrix (X), Target vector (Y)
#   Output      : Splitted data (X_train, X_test, Y_train, Y_test)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

#-------------------------------------
# Step 4.1 : Create the base model
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Create Base Model
#   Description : Initialize base DecisionTreeRegressor model
#   Input       : Hyperparameters (random_state)
#   Output      : DecisionTreeRegressor instance (base_model)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

base_model = DecisionTreeRegressor(random_state=42)

#-------------------------------------
# Step 4.2 : Create the Bagging model
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Create Bagging Model
#   Description : Create BaggingRegressor ensemble model using base estimator
#   Input       : Base estimator, n_estimators, random_state
#   Output      : BaggingRegressor instance (model)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

model = BaggingRegressor(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)

#-------------------------------------
# Step 5 : Train the model
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Train Model
#   Description : Train BaggingRegressor model using training dataset
#   Input       : Training features (X_train), Training labels (Y_train)
#   Output      : Fitted BaggingRegressor model
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

model = model.fit(X_train,Y_train)

#-------------------------------------
# Step 6 : Test the model
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Test Model
#   Description : Make predictions on test set using trained model
#   Input       : Test features (X_test)
#   Output      : Predicted values (Y_pred)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

Y_pred = model.predict(X_test)

#-------------------------------------
# Step 7 : Evaluate the model
#-------------------------------------
#-----------------------------------------------------
#   Block Name  : Evaluate Model
#   Description : Calculate Mean Squared Error (MSE) and R2 Score
#   Input       : Actual labels (Y_test), Predicted labels (Y_pred)
#   Output      : Console output (MSE and R2 Score)
#   Author      : Mangesh Thak
#   Date        : 22/08/2026
#-----------------------------------------------------

print("MSE : ",mean_squared_error(Y_test, Y_pred))
print("R2 : ",r2_score(Y_test, Y_pred))