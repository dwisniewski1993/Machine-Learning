from SentenceClassifier.SentenceClassifier import Sentencer


def main():
    """
    Main function to train the sentence classifier.
    """
    # Define paths to positive and negative data files
    positives_path = 'pos.txt'
    negatives_path = 'neg.txt'

    # Initialize and train the Sentencer classifier
    sentence = Sentencer(positive_data=positives_path, negative_data=negatives_path)
    sentence.train_model(num_epochs=100)


if __name__ == '__main__':
    main()
