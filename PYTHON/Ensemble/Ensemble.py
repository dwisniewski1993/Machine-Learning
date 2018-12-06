from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


class EnsembleVotingClassifier:
    def __init__(self, trainfile):
        self.sample_setosa = None
        self.sample_versicolor = None
        self.sample_virginica = None

        self.trainfile = trainfile
        train_df = pd.read_csv(self.trainfile)
        train_array = train_df.values

        self.X = train_array[:, 0:4]
        self.Y = train_array[:, 4]

        self.Y = self.map_labels(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)
        seed = 4
        self.kfold = KFold(n_splits=10, random_state=seed)

        self.estimators = []
        self.model_1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=200)
        self.model_2 = GaussianNB()
        self.model_3 = SVC(probability=True, kernel='rbf', C=1.0, gamma='auto')
        self.model_4 = RandomForestClassifier(n_estimators=50, random_state=0)
        self.model_5 = KNeighborsClassifier(n_neighbors=3)
        self.model_6 = GradientBoostingClassifier(n_estimators=100, random_state=4)
        self.model_7 = MLPClassifier(hidden_layer_sizes=1, activation='relu', solver='adam', alpha=0.01,
                                     learning_rate='constant', learning_rate_init=0.01, max_iter=1500)

        self.estimators.append(('logistic', self.model_1))
        self.estimators.append(('naiveB', self.model_2))
        self.estimators.append(('svm', self.model_3))
        self.estimators.append(('forest', self.model_4))
        self.estimators.append(('knn', self.model_5))
        self.estimators.append(('GBC', self.model_6))
        self.estimators.append(('MLP', self.model_7))

        self.ensemble = VotingClassifier(estimators=self.estimators, voting='hard')

    def set_samples(self, setosa, versicolor, virginica):
        sample = np.array(setosa)
        self.sample_setosa = sample.reshape(1, -1)

        sample = np.array(versicolor)
        self.sample_versicolor = sample.reshape(1, -1)

        sample = np.array(virginica)
        self.sample_virginica = sample.reshape(1, -1)

    @staticmethod
    def map_labels(labels):
        maped = []
        for each in labels:
            if each == 'Iris-setosa':
                maped.append(0.0)
            elif each == 'Iris-versicolor':
                maped.append(1.0)
            else:
                maped.append(2.0)
        return maped

    def get_val_results(self):
        result = cross_val_score(self.ensemble, self.X, self.Y, cv=self.kfold)
        return result

    def train_model(self):
        self.ensemble.fit(self.X_train, self.Y_train)

    def output_score(self):
        print("Accuracy: {:.2f}".format(self.ensemble.score(self.X_test, self.Y_test)))

    def output_predictions(self):
        print("Sample 1 --------------SETOSA___0___---------------------")
        print("Voting prediction: ", self.ensemble.predict(self.sample_setosa))

        print("Logistic Regression prediction: ", self.ensemble.named_estimators_.logistic.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.logistic.score(self.X_test, self.Y_test)))

        print("Naive Bayes predictions: ", self.ensemble.named_estimators_.naiveB.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.naiveB.score(self.X_test, self.Y_test)))

        print("SVM prediction: ", self.ensemble.named_estimators_.svm.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.svm.score(self.X_test, self.Y_test)))

        print("Forest prediction: ", self.ensemble.named_estimators_.forest.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.forest.score(self.X_test, self.Y_test)))

        print("KNN prediction: ", self.ensemble.named_estimators_.knn.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.knn.score(self.X_test, self.Y_test)))

        print("Gradient Boosting Classifier prediction: ", self.ensemble.named_estimators_.GBC.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.GBC.score(self.X_test, self.Y_test)))

        print("Multi Layer Perceptron prediction: ", self.ensemble.named_estimators_.MLP.predict(self.sample_setosa))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.MLP.score(self.X_test, self.Y_test)))

        print("Sample 2 --------------SETOSA___1___---------------------")
        print("Voting prediction: ", self.ensemble.predict(self.sample_versicolor))

        print("Logistic Regression prediction: ", self.ensemble.named_estimators_.logistic.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.logistic.score(self.X_test, self.Y_test)))

        print("Naive Bayes predictions: ", self.ensemble.named_estimators_.naiveB.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.naiveB.score(self.X_test, self.Y_test)))

        print("SVM prediction: ", self.ensemble.named_estimators_.svm.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.svm.score(self.X_test, self.Y_test)))

        print("Forest prediction: ", self.ensemble.named_estimators_.forest.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.forest.score(self.X_test, self.Y_test)))

        print("KNN prediction: ", self.ensemble.named_estimators_.knn.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.knn.score(self.X_test, self.Y_test)))

        print("Gradient Boosting Classifier prediction: ", self.ensemble.named_estimators_.GBC.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.GBC.score(self.X_test, self.Y_test)))

        print("Multi Layer Perceptron prediction: ", self.ensemble.named_estimators_.MLP.predict(self.sample_versicolor))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.MLP.score(self.X_test, self.Y_test)))

        print("Sample 3 --------------SETOSA___2___---------------------")
        print("Voting prediction: ", self.ensemble.predict(self.sample_virginica))

        print("Logistic Regression prediction: ", self.ensemble.named_estimators_.logistic.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.logistic.score(self.X_test, self.Y_test)))

        print("Naive Bayes predictions: ", self.ensemble.named_estimators_.naiveB.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.naiveB.score(self.X_test, self.Y_test)))

        print("SVM prediction: ", self.ensemble.named_estimators_.svm.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.svm.score(self.X_test, self.Y_test)))

        print("Forest prediction: ", self.ensemble.named_estimators_.forest.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.forest.score(self.X_test, self.Y_test)))

        print("KNN prediction: ", self.ensemble.named_estimators_.knn.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.knn.score(self.X_test, self.Y_test)))

        print("Gradient Boosting Classifier prediction: ", self.ensemble.named_estimators_.GBC.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.GBC.score(self.X_test, self.Y_test)))

        print("Multi Layer Perceptron prediction: ", self.ensemble.named_estimators_.MLP.predict(self.sample_virginica))
        print("Accuracy: {:.2f}".format(self.ensemble.named_estimators_.MLP.score(self.X_test, self.Y_test)))
