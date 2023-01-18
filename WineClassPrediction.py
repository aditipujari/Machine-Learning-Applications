import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def NaiveBayesClassifierFunction(Data_train, Data_test, Target_train, Target_test):

    Classifier = GaussianNB()
    
    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def KNearestNeighborFunction(Data_train, Data_test, Target_train, Target_test):

    Classifier = KNeighborsClassifier(n_neighbors=3) 

    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def DecisionTreeClassifierFunction(Data_train, Data_test, Target_train, Target_test):
    
    Classifier = tree.DecisionTreeClassifier()

    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)
    
    return Accuracy

def WineClassPrediction():
    # Import data
    df = pd.read_csv('WinePredictor (1).csv')

    features = df.drop(['Class'],axis = 1)
    label = df['Class']

    Data_train, Data_test, Target_train, Target_test = train_test_split(features, label, test_size=0.2)

    D_Accuracy = DecisionTreeClassifierFunction(Data_train, Data_test, Target_train, Target_test)
    print("Accuracy using Decision Tree Classifier is : ",D_Accuracy*100)

    K_Accuracy = KNearestNeighborFunction(Data_train, Data_test, Target_train, Target_test)
    print("Accuracy using K Nearest Neighbor Classifier is : ",K_Accuracy*100)

    N_Accuracy = NaiveBayesClassifierFunction(Data_train, Data_test, Target_train, Target_test)
    print("Accuracy using Naive Bayes Classifier is : ", N_Accuracy * 100)

def main():

    print("-------------Wine Predictor Case Study--------------------")

    WineClassPrediction()

if __name__ == "__main__":
    main()