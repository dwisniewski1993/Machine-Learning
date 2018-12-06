from Ensemble import EnsembleVotingClassifier


def main():

    flower = [5.1, 3.5, 1.4, 0.2]
    flower2 = [7.0, 3.2, 4.7, 1.4]
    flower3 = [5.9, 3.0, 5.1, 1.8]

    trainfilelocation = r'train.csv'
    ec = EnsembleVotingClassifier(trainfile=trainfilelocation)
    ec.set_samples(setosa=flower, versicolor=flower2, virginica=flower3)
    print(ec.get_val_results().mean())
    ec.train_model()
    ec.output_score()
    ec.output_predictions()


if __name__ == '__main__':
    main()
