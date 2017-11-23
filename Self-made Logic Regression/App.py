from DataPreprocessing import FixingShityInSet, FixingShityTrainSet
from Functions import data_load, str_column_to_float, dataset_minmax, normalize_dataset, logistic_regression
from random import seed
'''
Koniugacje
M:łem,łbym, ny
F:łam,łabym, na
'''

def main():
    print('Self-Made logic Regression')
    #FixingShityInSet()
    #FixingShityTrainSet()

    seed(1)

    out = open(r'out.tsv', 'w')

    filename = 'train_data_file.tsv'
    filename2 = 'in_data_file.tsv'

    dataset = data_load(filename)
    dataset2 = data_load(filename2)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    for i in range(len(dataset2[0])):
        str_column_to_float(dataset2, i)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset,minmax)

    l_rate = 0.01
    n_epoch = 100

    scores = logistic_regression(dataset, dataset2, l_rate, n_epoch)

    for each in scores:
        if each == 0:
            each = 'M'
        elif each == 1:
            each = 'F'
        out.write(each + '\n')
        print(each)

    out.close()



if __name__ == "__main__":
    main()