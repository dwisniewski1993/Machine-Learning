from Functions import data_load, str_column_to_float, dataset_minmax, normalize_dataset, linear_regression_sgd, price_column, del_mark1, del_mark2
from random import seed
#Predict Price
def main():
    print('Self Made Linear Regression')
    out = open(r'out.txt', 'w')

    seed(1)

    filename = 'train.tsv'
    filename2 = 'in.tsv'

    dataset = data_load(filename)
    dataset2 = data_load(filename2)

    del_mark1(dataset)
    del_mark2(dataset2)
    price_column(dataset)

    for each in dataset:
        print(each)
    print("DATASET 2")
    for each in dataset2:
        print(each)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    for i in range(len(dataset2[0])):
        str_column_to_float(dataset2, i)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    minmax2 = dataset_minmax(dataset2)
    normalize_dataset(dataset2, minmax2)
    for each in dataset:
        print(each)


    l_rate = 0.01
    n_epoch = 50

    scores = linear_regression_sgd(dataset, dataset2, l_rate, n_epoch)


    for each in scores:
        out.write(str(each) + '\n')
        print(each)



if __name__ == "__main__":
    main()