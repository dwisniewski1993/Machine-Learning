from Functions import coeffs_sgd, data_load, str_column_to_float, dataset_minmax, normalize_dataset, evaluate_algorithm, linear_regression_sgd, cross_validation_split
from random import seed
#Predict Price
def main():
    print('Self Made Linear Regression')

    seed(1)

    filename = 'train.tsv'
    dataset = data_load(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

    n_folds = 5
    l_rate = 0.01
    n_epochs = 50
    scores = evaluate_algorithm(dataset, linear_regression_sgd(), n_folds, l_rate, n_epochs)
    print("Scores: ", scores)
    print("Mean RMSE: ", (sum(scores)/float(len(scores))))




if __name__ == "__main__":
    main()