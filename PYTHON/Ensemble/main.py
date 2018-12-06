from Ensemble import EnsembleVotingClassifier


def main():
    """
    Main function. Run Ensemble Voting classification task.
    :return:
    """
    # Set flowers samples
    flower = [5.1, 3.5, 1.4, 0.2]
    flower2 = [7.0, 3.2, 4.7, 1.4]
    flower3 = [5.9, 3.0, 5.1, 1.8]

    # Ensemble Classification
    train_file_location = r'train.csv'
    ec = EnsembleVotingClassifier(trainfile=train_file_location)
    ec.set_samples(setosa=flower, versicolor=flower2, virginica=flower3)
    ec.normalize()
    ec.train_model()
    ec.output_predictions()
    ec.output_score()
