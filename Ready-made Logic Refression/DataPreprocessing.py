import re

in_set = open(r'../in.tsv', 'r')
train_set = open(r'../train.tsv', 'r')

lem_pattern = re.compile(r'łem') #M
lam_pattern = re.compile(r'łam') #F
lbym_pattern = re.compile(r'łbym') #M
labym_pattern = re.compile(r'łabym') #F
ny_pattern = re.compile(r'ny') #M
na_patrern = re.compile(r'na') #F

def FixingShityInSet():
    dataset = list()
    for each in in_set:
        count_men = 0
        count_women = 0
        data = list()
        each = each.split('\t')

        if re.search(lem_pattern,each[0]):
            #print("Znalazlem lem w :", each[0])
            count_men += 1

        if re.search(lam_pattern, each[0]):
            #print("Znalazlem lam w: ", each[0])
            count_women += 1

        if re.search(lbym_pattern, each[0]):
            #print("znalazlem lbym w: ", each[0])
            count_men += 1

        if re.search(labym_pattern, each[0]):
            #print("znalazlem labym w: ", each[0])
            count_women += 1

        if re.search(ny_pattern, each[0]):
            #print("znalazlem ny w: ", each[0])
            count_men += 1

        if re.search(na_patrern, each[0]):
            #print("znalazlem na w: ", each[0])
            count_women += 1

        #print("Dla tej lini MEN: ", count_men, " i WOMEN: ", count_women)
        data.append(count_men)
        data.append(count_women)
        #print(data)

        dataset.append(data)

    return dataset

def FixingShityTrainSet():
    pass