from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    # Load dataset
    wine = datasets.load_wine()

    # Print the names of the features
    print(wine.feature_names)

    # Print the label names(class_0,class_1,class_2)
    print(wine.target_names)

    # Print the wind data(top 5 records)
    print(wine.data[0:5])

    # Print the wine labels(0: class_0, 1:class_1, 2:class_2)
    print(wine.target)

    # Split the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size = 0.3) # 70% training and 30% testing

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 3)

    # Train the model using training sets
    knn.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # Model Accuracy, how often is the classifier correct
    print("Accuracy : ", metrics.accuracy_score(y_test,y_pred))
    
def main():
    print("Machine Learning Application")
    print("Wine Predictor Application using K Nearest Neighbor Algorithm")

    WinePredictor()

if __name__ == "__main__":
    main()
