from SentenceClassifier.SentenceClassifier import Sentencer


def main():
    positives_path = 'pos.txt'
    negatives_path = 'neg.txt'

    sentence = Sentencer(positive_data=positives_path, negative_data=negatives_path)
    sentence.train_model(100)
