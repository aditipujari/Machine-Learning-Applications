from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42, test_size = 0.85)

# Create classifier object
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()

vot_clf = VotingClassifier(estimators = [('lr',log_clf),('rnd',rnd_clf),('knn',knn_clf)],voting = 'hard')

vot_clf.fit(X_train,y_train)

pred = vot_clf.predict(X_test)

print("Accuracy : ", accuracy_score(y_test,pred )*100)