from DataPreprocessing import FixingShityInSet, FixingShityTrainSet
from Functions import data_load, str_column_to_float, dataset_minmax, normalize_dataset, logistic_regression
from random import seed
'''
Koniugacje
M:łem,łbym, ny
F:łam,łabym, na
'''

def main():
    print('Self-Made Logical Regression')
    #FixingShityInSet()
    #FixingShityTrainSet()

    seed(1)

    out1 = open(r'dev-0/out.tsv', 'w')
    out2 = open(r'dev-1/out.tsv', 'w')
    out3 = open(r'test-A/out.tsv', 'w')

    filename = 'train_data_file.tsv'
    filename2 = 'in_data_file_dev0.tsv'
    filename3 = 'in_data_file_dev1.tsv'
    filename4 = 'in_data_file_testA.tsv'

    dataset = data_load(filename)
    dataset2 = data_load(filename2)
    dataset3 = data_load(filename3)
    dataset4 = data_load(filename4)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    for i in range(len(dataset2[0])):
        str_column_to_float(dataset2, i)

    for i in range(len(dataset3[0])):
        str_column_to_float(dataset3, i)

    for i in range(len(dataset4[0])):
        str_column_to_float(dataset4, i)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

    l_rate = 0.01
    n_epoch = 100

    print("DEV-0")
    scores1 = logistic_regression(dataset, dataset2, l_rate, n_epoch)
    print("DEV-1")
    scores2 = logistic_regression(dataset, dataset3, l_rate, n_epoch)
    print("TEST-A")
    scores3 = logistic_regression(dataset, dataset4, l_rate, n_epoch)

    for each in scores1:
        if each == 0:
            each = 'M'
        elif each == 1:
            each = 'F'
        out1.write(each + '\n')
        print(each)

    for each in scores2:
        if each == 0:
            each = 'M'
        elif each == 1:
            each = 'F'
        out1.write(each + '\n')
        print(each)

    for each in scores3:
        if each == 0:
            each = 'M'
        elif each == 1:
            each = 'F'
        out1.write(each + '\n')
        print(each)

    out1.close()
    out2.close()
    out3.close()
    print("DONE!!!")



if __name__ == "__main__":
    main()