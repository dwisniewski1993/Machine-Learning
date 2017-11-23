from DataPreprocessing import FixingShityInSet, FixingShityTrainSet
'''
Koniugacje
M:łem,łbym, ny
F:łam,łabym, na
'''

'''
NEW Dataset LOOK:
M/W | counted men con | counted women con
'''

def main():
    print('Ready-made logic Regression')
    #FixingShityInSet()
    #FixingShityTrainSet()

    train_data = open(r'train_data_file.tsv', 'r')
    in_data = open(r'in_data_file.tsv', 'r')

    for each in in_data:
        print(each)
    for each in train_data:
        print(each)




    train_data.close()
    in_data.close()




if __name__ == "__main__":
    main()