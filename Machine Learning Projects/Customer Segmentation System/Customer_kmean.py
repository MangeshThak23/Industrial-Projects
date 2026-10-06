import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#-----------------------------------------------------
#   Function Name : main
#   Description   : Entry point function for Customer Segmentation using K-Means
#   Input         : None
#   Output        : Visualizations and clustered DataFrame
#   Author        : Mangesh Thak
#   Date          : 25/08/2026
#-----------------------------------------------------
def main():
    #-----------------------------------------------------
    #   Step 1 : Load the data
    #   Description : Read customer dataset from CSV and check missing values
    #   Input       : CSV File Name ("Mall_Customers.csv")
    #   Output      : DataFrame (df)
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    df = pd.read_csv("Mall_Customers.csv")

    print("Dataset loaded with values")
    print(df.head())

    print("Missing values : ")
    print(df.isnull().sum())  

    #-----------------------------------------------------
    #   Step 2 : Feature selection
    #   Description : Select relevant features for clustering
    #   Input       : DataFrame (df)
    #   Output      : Selected feature matrix (X)
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    X = df[["AnnualIncome", "SpendingScore"]]

    print("Selected fetures : ")
    print(X.head())

    #-----------------------------------------------------
    #   Step 3 : Scale the data
    #   Description : Standardize feature values using StandardScaler
    #   Input       : Feature matrix (X)
    #   Output      : Scaled array (X_scaled)
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    scalar = StandardScaler()

    X_scaled = scalar.fit_transform(X)

    print("Scaled data : ")
    print(X_scaled[:5])

    #-----------------------------------------------------
    #   Step 4 : Elbow method
    #   Description : Calculate Within-Cluster Sum of Square (WCSS) for K from 1 to 10
    #   Input       : Scaled feature matrix (X_scaled)
    #   Output      : List of WCSS values
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    WCSS = []

    for k in range(1,11):
        model = KMeans(
            n_clusters = k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        WCSS.append(model.inertia_)

    print("Values of WCSS : ")
    for i in range(len(WCSS)):
        print(f"{i+1} : {WCSS[i]}")

    #-----------------------------------------------------
    #   Step 5 : Visualisation
    #   Description : Plot Elbow curve to determine optimal number of clusters
    #   Input       : WCSS list, K range (1 to 10)
    #   Output      : Line plot chart
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    plt.plot(range(1,11),WCSS, marker = "o")
    plt.xlabel("Number of clusters : k")
    plt.ylabel("WCSS")
    plt.title("Marvellous Elbow method")
    plt.grid(True)
    plt.show()

    #-----------------------------------------------------
    #   Step 6 : Final model
    #   Description : Train KMeans with optimal clusters and assign cluster labels
    #   Input       : Scaled dataset (X_scaled), n_clusters = 4
    #   Output      : DataFrame (df) with appended Cluster column
    #   Author      : Mangesh Thak
    #   Date        : 25/08/2026
    #-----------------------------------------------------
    model = KMeans(
                n_clusters = 4,
                random_state=42,
                n_init=10
            )

    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    print("Dataset with clusters : ")
    print(df.head(100))

if __name__ == "__main__":
    main()