from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
import logging as log


class EnsembleVotingClassifier:
    """
    Ensemble Voting Classifier
    """

    def __init__(self, trainfile: str):
        """
        Ensemble Classifier Constructor
        :param trainfile: iris data csv path
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Ensemble Classifier')

        self.sample_setosa = None
        self.sample_versicolor = None
        self.sample_virginica = None

        # Load set
        self.trainfile = trainfile
        train_df = pd.read_csv(self.trainfile)
        train_array = train_df.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy.Arrays
        self.X = train_array[:, 0:4]
        self.Y = train_array[:, 4]

        # Map string labels to numeric
        self.Y = self.map_labels(self.Y)

        # Split to train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)
        seed = 4
        self.kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

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

    def __str__(self):
        """
        Printing data
        :return: None
        """
        print("Features: {}, Labels: {}".format(self.X, self.Y))

    def set_samples(self, setosa, versicolor, virginica):
        """
        Set samples
        :param setosa: setosa flower
        :param versicolor: versicolor flower
        :param virginica: virginica flower
        :return: None
        """
        sample = np.array(setosa)
        self.sample_setosa = sample.reshape(1, -1)

        sample = np.array(versicolor)
        self.sample_versicolor = sample.reshape(1, -1)

        sample = np.array(virginica)
        self.sample_virginica = sample.reshape(1, -1)

    @staticmethod
    def map_labels(labels: np.ndarray) -> list:
        """
        Mapping iris data labels to numeric
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        maped = [0.0 if x == 'Iris-setosa' else 1.0 if x == 'Iris-versicolor' else 2.0 for x in labels]
        return maped

    def rescale(self) -> None:
        """
        Rescaling data in dataset to [0,1]
        :return: None
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def normalize(self) -> None:
        """
        Normalizing data in dataset
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def standardize(self) -> None:
        """
        Standardizing data in dataset
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def get_val_results(self) -> np.ndarray:
        """
        Cross validation classification score
        :return: array of scores
        """
        return cross_val_score(self.ensemble, self.X, self.Y, cv=self.kfold)

    def train_model(self) -> None:
        """
        Training models
        :return: None
        """
        self.ensemble.fit(self.X_train, self.Y_train)

    def output_score(self) -> None:
        """
        Calculating and logging accuracy score
        :return: None
        """
        log.info(f"Accuracy: {self.ensemble.score(self.X_test, self.Y_test):.2f}")
        log.info(f"F1 Score: {f1_score(self.Y_test, self.ensemble.predict(self.X_test), average='weighted'):.2f}")

    def output_predictions(self) -> None:
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
