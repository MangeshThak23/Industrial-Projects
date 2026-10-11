import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------------------
#   Function Name : LoadDataset
#   Description   : Load dataset from CSV, inspect it, and encode binary/categorical variables
#   Input         : Path of CSV file (Datapath)
#   Output        : Tuple (Processed DataFrame, Fitted LabelEncoder instance)
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def LoadDataset(Datapath):

    df = pd.read_csv(Datapath)

    print("Shape of dataset: ", df.shape)
    print("First Five Data from a dataset: \n", df.head())

    print(df.columns)

    print(df.isnull().sum())

    df["PreviousDefault"] = (df["PreviousDefault"] == "Yes").astype(int)

    le = LabelEncoder()
    df["HomeOwnership"] = le.fit_transform(df["HomeOwnership"])

    print(df.head())

    return df, le


#-----------------------------------------------------
#   Function Name : CheckClassAreBalced
#   Description   : Check distribution and ratio of target classes to verify balance
#   Input         : DataFrame (df)
#   Output        : Console output (prints class counts, ratios, and balance condition)
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def CheckClassAreBalced(df):
    class_counts = df["Default"].value_counts()
    print("target class Distrution ")
    print(class_counts)
    print("target class percentage")
    print(df["Default"].value_counts(normalize=True) * 100)
    class_ration = class_counts.min() / class_counts.max()
    if class_ration >= 0.8:
        print("class are balanced")
    else:
        print("class are not balanced")


#-----------------------------------------------------
#   Function Name : SplitDataset
#   Description   : Separate features/target, split train-test data, and scale features
#   Input         : DataFrame (df)
#   Output        : Tuple (X_train, X_test, Y_train, Y_test, fitted StandardScaler instance)
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def SplitDataset(df):

    X = df.drop("Default", axis=1)
    Y = df["Default"]

    print("Independent variables: ", X.shape)
    print("Dependent variables: ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, Y_train, Y_test, scalar


#-----------------------------------------------------
#   Function Name : ModelCreation
#   Description   : Initialize and train the base MLPClassifier model
#   Input         : X_train, Y_train
#   Output        : Trained MLPClassifier model instance
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def ModelCreation(X_train, Y_train):
   
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    model = model.fit(X_train, Y_train)

    return model


#-----------------------------------------------------
#   Function Name : exprement
#   Description   : Run hyperparameter experiments on activation functions, hidden layer architectures, and learning rates
#   Input         : X_train, Y_train, X_test, Y_test
#   Output        : Last trained MLPClassifier model instance
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def exprement(X_train, Y_train, X_test, Y_test):
    # experiment 1 
    activation_reslut = {}
    Activation = [
        "identity",
        "logistic",
        "tanh",
        "relu"
    ]

    for act in Activation:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation=act,
            solver="adam",
            max_iter=1000,
            random_state=42
        )
        model = model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy_test = accuracy_score(Y_test, Y_pred) * 100
        activation_reslut[act] = accuracy_test

    for avtivation, accurecy in activation_reslut.items():
        print(avtivation, ":", accurecy)

    ####################################################
    hidden_reslut = {}
    hiddenlayers = [
        (10),
        (20, 10),
        (50, 25),
        (100, 50, 25)
    ]
   
    for layer in hiddenlayers:
        model = MLPClassifier(
            hidden_layer_sizes=layer,
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
        model = model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy_test = accuracy_score(Y_test, Y_pred) * 100
        hidden_reslut[layer] = accuracy_test
   
    for hidden, accurecy in hidden_reslut.items():
        print(hidden, ":", accurecy)

    #############################################################
    learning_reslut = {}
    learninglayers = [
        0.0001,
        0.001,
        0.01,
        0.1
    ]
   
    for layer in learninglayers:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=layer,
            max_iter=1000,
            random_state=42
        )
        model = model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy_test = accuracy_score(Y_test, Y_pred) * 100
        learning_reslut[layer] = accuracy_test
   
    for lerlayer, accurecy in learning_reslut.items():
        print(lerlayer, ":", accurecy)

    return model


#-----------------------------------------------------
#   Function Name : EvaluateModel
#   Description   : Evaluate accuracy, fit condition, confusion matrix, classification report, and loss curve plot
#   Input         : model, X_train, X_test, Y_train, Y_test
#   Output        : Console output and Matplotlib Loss Curve plot
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def EvaluateModel(model, X_train, X_test, Y_train, Y_test):

    Y_pred = model.predict(X_test)

    accuracy_test = accuracy_score(Y_test, Y_pred) * 100
    print(f"Accuracy of testing is: {accuracy_test:.2f}%")

    Y_train_pred = model.predict(X_train)
    accuracy_training = accuracy_score(Y_train, Y_train_pred) * 100
    print(f"Accuracy of training is:{accuracy_training:.2f}%")

    gap = accuracy_training - accuracy_test
   
    if gap > 10.0:
        print("Model is: Overfitting")
    elif accuracy_training < 70.0 and accuracy_test < 70.0:
        print("Model is: Underfitting")
    else:
        print("Model is: Best fit")

    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix is: \n", cm)

    print("Classification report: ")
    print(classification_report(Y_test, Y_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, color="blue", linewidth=2)
    plt.title("Training losss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


#-----------------------------------------------------
#   Function Name : Whystratify
#   Description   : Print explanation of stratified train-test splitting
#   Input         : None
#   Output        : Console output (prints explanation string)
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def Whystratify():
    print("stratify spliting keeps approxmilitly the same propration  of Default zero and default 1 in both the traning and testinfg data set")


#-----------------------------------------------------
#   Function Name : PredictLoanApplication
#   Description   : Preprocess new application data and predict loan default outcomes
#   Input         : scalar (StandardScaler), model (MLPClassifier), le (LabelEncoder)
#   Output        : Console output (prints prediction probabilities and default verdicts)
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def PredictLoanApplication(scalar, model, le):

    new_app = pd.DataFrame([[37, 45000, 15000, 720, 5, 1, 1200, 36, "No", "Rent"],
                            [24, 22000, 25000, 580, 1, 3, 1100, 60, "Yes", "Rent"]],
                           columns=['Age', 'Income', 'LoanAmount', 
                                    'CreditScore', 'EmploymentYears',
                                    'ExistingLoans', 'MonthlyDebt', 'LoanTerm', 'PreviousDefault',
                                    'HomeOwnership'])

    new_app["PreviousDefault"] = (new_app["PreviousDefault"] == "Yes").astype(int)
    new_app["HomeOwnership"] = le.transform(new_app["HomeOwnership"])

    new_app_scaled = scalar.transform(new_app)

    new_pred = model.predict(new_app_scaled)

    new_prob = model.predict_proba(new_app_scaled)

    print("New loan application data: ")
    print(new_app)

    print("Prediction probability: \n", new_prob)

    count = 1
    for pred in new_pred:
        if pred == 1:
            print("Applicant no.: ", count, "-> is defaultor")
        else:
            print("Applicant no.: ", count, "-> is Not defaultor")
        count += 1


#-----------------------------------------------------
#   Function Name : main
#   Description   : Main driver function to execute the full pipeline
#   Input         : None
#   Output        : Pipeline execution results
#   Author        : Mangesh Thak
#   Date          : 16/08/2026
#-----------------------------------------------------
def main():

    df, le = LoadDataset("Loan_Default.csv")
    CheckClassAreBalced(df)
    X_train, X_test, Y_train, Y_test, scalar = SplitDataset(df)
    model = ModelCreation(X_train, Y_train)
    Whystratify()
    EvaluateModel(model, X_train, X_test, Y_train, Y_test)
    PredictLoanApplication(scalar, model, le)
    exprement(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()