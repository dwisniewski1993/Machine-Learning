from sklearn import linear_model
import numpy as np
from ManyArgs import many_arg
from OneArgs import one_arg

file = open(r'train.tsv')
file2 = open(r'in.tsv')

def main():
    one_arg()
    many_arg()

if __name__ == "__main__":
    main()