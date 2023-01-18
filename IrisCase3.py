# Iris Versicolor, Setosa, Virginica
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

# k = 1
def euc(a,b):
    return distance.euclidean(a,b)

class MarvellousKNeighborsClassifier:

    def fit(self, trainingdata,trainingtarget):
        self.TrainingData = trainingdata
        self.TrainingTarget = trainingtarget

    def closest(self,row):
        minimumdistance = euc(row, self.TrainingData[0])
        minimumindex = 0

        for i in range(1, len(self.TrainingData)):
            Distance = euc(row,self.TrainingData[i])
            if Distance < minimumdistance:
                minimumdistance = Distance
                minimumindex = i

        return self.TrainingTarget[minimumindex]

    def predict(self, TestData):
        predictions = []
        for value in TestData:
            result = self.closest(value)
            predictions.append(result)

        return predictions

def MarvellousML():
    Dataset = load_iris()               # load the daata

    Data = Dataset.data                 # Features,data,attributes
    Target = Dataset.target             # Target,label,result

    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5) # Return multiple lists that too multidimensional

    Classifier = MarvellousKNeighborsClassifier()

    Classifier.fit(Data_train, Target_train)

    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def main():
    Ret = MarvellousML()
    print("Accuracy of Iris Dataset with KNN is : ",Ret * 100)

if __name__ == "__main__":
    main()