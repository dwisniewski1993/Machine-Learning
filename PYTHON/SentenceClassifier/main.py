from SentenceClassifier.SentenceClassifier import Sentencer


def main():
    possitives_path = 'pos.txt'
    negatives_path = 'neg.txt'

    sentencer = Sentencer(possitive_data=possitives_path, negative_data=negatives_path)
    sentencer.train_model(100)
