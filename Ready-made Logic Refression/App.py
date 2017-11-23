from DataPreprocessing import FixingShityInSet
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
    in_data = FixingShityInSet()

    print("Main func")

    print(in_data)

if __name__ == "__main__":
    main()