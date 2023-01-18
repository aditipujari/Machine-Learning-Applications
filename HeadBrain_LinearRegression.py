import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def MarvellousHeadBrainPredictor():
    # Load data
    data = pd.read_csv('MarvellousHeadBrain (1) (1).csv')
    print("Size of dataset : ",data.shape)

    X = data["Head Size(cm^3)"].values
    Y = data["Brain Weight(grams)"].values

    X = X.reshape((-1,1))

    n = len(X)

    reg = LinearRegression()
    reg = reg.fit(X,Y)

    y_pred = reg.predict(X)
    r2 = reg.score(X,Y)

    print(r2)

def main():
    print("Supervised Machine Learning")
    print("Linear Regression on head and brain size dataset")

    MarvellousHeadBrainPredictor()

if __name__ == "__main__":
    main()