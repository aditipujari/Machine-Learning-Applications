from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def SVMAlgorithm():
    # Load dataset
    cancer = datasets.load_breast_cancer()

    # Print the names of the 13 features
    print("Features of the cancer dataset : ", cancer.feature_names)

    # Print the label type of cancer('malignant','benign')
    print("Labels of the cancer dataset : ",cancer.target_names)

    # Print data(feature) shape
    print("Shape of the dataset is : ", cancer.data.shape)

    # Print the cancer data features (top 5 records)
    print("First 5 records are  :")
    print(cancer.data[0:5])

    # Print the cancer labels (0 : malignant, 1 : benign)
    print("Target of dataset : ",cancer.target)

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 109)

    # Create a SVM classifier
    clf = svm.SVC(kernel = 'linear') # Linear kernel

    # Train the model using the training sets
    clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy : how often is the classifier correct ?
    print("Accuracy of the model is  :",metrics.accuracy_score(y_test,y_pred))

def main(): 
    print("Support Vector Machine")

    SVMAlgorithm()

if __name__ == "__main__":
    main()