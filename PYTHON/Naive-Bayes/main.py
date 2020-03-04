from NB.Classifier import NaiveBayesClassifier


def main():
    """
    Main function. Run Gaussian Naive Bayes classification task.
    train.csv: iris dataset
    :return: None
    """
    # Naive Bayes Classification
    train_file_location = r'train.csv'
    nb = NaiveBayesClassifier(train_file=train_file_location)
    nb.normalize()
    nb.train_model()
    nb.score()
