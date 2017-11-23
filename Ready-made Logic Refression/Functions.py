from csv import reader

#Load Dataset
def data_load(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter='\t')
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        try:
            row[column] = float(row[column].strip())
        except:
            if row[column]=='M':
                row[column] = float(10)
            elif row[column]=='F':
                row[column] = float(20)
            else:
                s = row[column]
                q = ''.join(str(ord(c)) for c in s)
                row[column] = float(q)