import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def CheckAccuracy(features, label):

    Data_train, Data_test, Target_train, Target_test = train_test_split(features, label, test_size=0.5)

    Classifier = KNeighborsClassifier(n_neighbors=6) # K is chosen such that it is near to square root of N where N is total number of samples

    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)

    print("Accuracy is: ",Accuracy*100)

def PlayPrediction():
    # Import data
    df = pd.read_csv('PlayPredictor.csv')

    # Apply Encoding technique
    le = LabelEncoder()

    weather = le.fit_transform(df['Whether'])
    temperature = le.fit_transform(df['Temperature'])
    label = le.fit_transform(df['Play'])

    # Features Combine
    features = list(zip(weather, temperature))

    Classifier = KNeighborsClassifier(n_neighbors=3)

    Classifier.fit(features, label)             # Whole dataset is given for training

    Predictions = Classifier.predict([[1, 2]])  # 1 - Rainy 2 - Mild

    if Predictions == 1:
        print("All can play in this weather.")  # This is the output for the given input
    else:
        print("Cannot play in this weather.")

    CheckAccuracy(features,label)               # Method to calculate Accuracy

def main():

    print("-------------Play Predictor Case Study--------------------")

    PlayPrediction()

if __name__ == "__main__":
    main()