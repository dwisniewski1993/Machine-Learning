from RecSys.Sys import RecommendationSystem


def main():
    dataset_location = r'ml-latest-small/ratings.csv'

    rc = RecommendationSystem(dataset_path=dataset_location)
    rc.train_model()
    rc.model_evaluation()


if __name__ == '__main__':
    main()
