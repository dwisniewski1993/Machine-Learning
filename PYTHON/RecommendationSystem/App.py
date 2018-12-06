from RecSys.Sys import RecSystem


def main():
    print("Recommendation System")

    datasetLocation = r'ml-latest-small/ratings.csv'

    rc = RecSystem(datasetloc=datasetLocation)
    rc.train_network()
    scores = rc.give_first_ten_pred()

    print(scores)


if __name__ == '__main__':
    main()
