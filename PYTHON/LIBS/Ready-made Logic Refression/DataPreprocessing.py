import re

def FixingShityInSet():
    in_set = open(r'../in.tsv', 'r', encoding='utf8')
    in_data_file = open(r'in_data_file.tsv', 'w', encoding='utf8')

    lem_pattern = re.compile(r'łem')  # M
    lam_pattern = re.compile(r'łam')  # F
    lbym_pattern = re.compile(r'łbym')  # M
    labym_pattern = re.compile(r'łabym')  # F
    ny_pattern = re.compile(r'ny')  # M
    na_patrern = re.compile(r'na')  # F

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
    for each in dataset:
        in_data_file.write(str(each[0]) + '\t' + str(each[1]) + '\n')
    in_data_file.close()

    return dataset

def FixingShityTrainSet():
    train_set = open(r'../train.tsv', 'r', encoding='utf8')
    train_data_file = open(r'train_data_file.tsv', 'w', encoding='utf8')

    lem_pattern = re.compile(r'łem')  # M
    lam_pattern = re.compile(r'łam')  # F
    lbym_pattern = re.compile(r'łbym')  # M
    labym_pattern = re.compile(r'łabym')  # F
    ny_pattern = re.compile(r'ny')  # M
    na_patrern = re.compile(r'na')  # F

    dataset = list()
    for each in train_set:
        count_men = 0
        count_women = 0
        data = list()
        each = each.split('\t')
        print(each)
        try:

            if re.search(lem_pattern, each[1]):
                # print("Znalazlem lem w :", each[0])
                count_men += 1

            if re.search(lam_pattern, each[1]):
                # print("Znalazlem lam w: ", each[0])
                count_women += 1

            if re.search(lbym_pattern, each[1]):
                # print("znalazlem lbym w: ", each[0])
                count_men += 1

            if re.search(labym_pattern, each[1]):
                # print("znalazlem labym w: ", each[0])
                count_women += 1

            if re.search(ny_pattern, each[1]):
                # print("znalazlem ny w: ", each[0])
                count_men += 1

            if re.search(na_patrern, each[1]):
                # print("znalazlem na w: ", each[0])
                count_women += 1

            # print("Dla tej lini MEN: ", count_men, " i WOMEN: ", count_women)
            data.append(count_men)
            data.append(count_women)
            data.append(each[0])
            # print(data)

            dataset.append(data)
        except:
            pass
    for each in dataset:
        train_data_file.write(str(each[0]) + '\t' + str(each[1]) + '\t' + str(each[2]) + '\n')
    train_data_file.close()

    return dataset