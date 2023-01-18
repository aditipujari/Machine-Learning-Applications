import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MarvellousHeadBrainPredictor():
    # Load data
    data = pd.read_csv('MarvellousHeadBrain (1) (1).csv')
    print("Size of dataset : ",data.shape)

    X = data["Head Size(cm^3)"].values
    Y = data["Brain Weight(grams)"].values

    # Least Square Method
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)

    numerator = 0
    denominator = 0

    # Equation of line is y = mx + c
    for i in range(n):
        numerator += (X[i]-mean_x)*(Y[i]-mean_y)
        denominator += (X[i]-mean_x)**2

    m = numerator/denominator

    c= mean_y - (m*mean_x)

    print("Slope of Regression line is : ",m)
    print("Y intercept of Regression line is : ",c)

    max_x = np.max(X)+100
    min_x = np.min(X)-100

    # Display plotting of above points
    x = np.linspace(min_x,max_x,n)

    y = c + m*x

    plt.plot(x,y,color = '#58b970',label = "Regression Line")

    plt.scatter(X,Y,color = '#ef5423', label = "Scatter Plot")

    plt.xlabel("Head size in cm3")

    plt.ylabel("Brain weight in gram")

    plt.legend()
    plt.show()

def main():
    print("Supervised Machine Learning")
    print("Linear Regression on head and brain size dataset")

    MarvellousHeadBrainPredictor()

if __name__ == "__main__":
    main()