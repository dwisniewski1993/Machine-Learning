#String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        row[column] = float(ord(row[column]))
