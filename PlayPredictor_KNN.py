import numpy as nop
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    # Step 1 :  Load data
    data = pd.read_csv(data_path,index_col = 0)

    # Step 2 : Clean, Prepare and Manipulate data
    feature_names = ['Whether','Temperature']
    print("Name of features ",feature_names)

    weather = data.Whether
    temperature = data.Temperature
    play = data.Play

    # Creating label encoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers
    weather_encoded = le.fit_transform(weather)
    temp_encoded = le.fit_transform(temperature)
    label = le.fit_transform(play)

    # Combining weather and temperature into single list of tuples
    features = list(zip(weather_encoded,temp_encoded))

    # Step 3 : Train data
    model = KNeighborsClassifier(n_neighbors = 3)

    # Train he model using the training set
    model.fit(features,label)

    # Step 4 : Test Data
    predicted = model.predict([[0,2]])
    print(predicted)

def main():
    print("Supervised Machine Learning")
    print("K Nearest Neighbor on play predictor dataset")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()