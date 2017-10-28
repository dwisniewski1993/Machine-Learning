from random import randrange
from math import sqrt
from csv import reader

#Load dataset
def data_load(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter='\t')
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Convert string to float
def str_column_to_float(dataset, column):
    for row in dataset:
        col = row[column]
        try:
            row[column] = float(row[column].strip())
        except:
            print("Vallue Error")
            s = row[column]
            q = ''.join(str(ord(c)) for c in s)
            row[column] = float(q)

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		rmse = rmse_metric(actual, predicted)
		scores.append(rmse)
	return scores

#Make a prediction
def predict(row, coeffs):
    yhat = coeffs[0]
    for i in range(len(row)-1):
        yhat += coeffs[i + 1] * row[i]
    return yhat

#Estimate Linear Regression with stoch, grand, desc
def coeffs_sgd(train, l_rate, n_epochs):
    coeff = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epochs):
        sum_error = 0
        for row in train:
            yhat = predict(row, coeff)
            error = yhat - row[-1]
            sum_error += error**2
            coeff[0] = coeff[0] - l_rate * error
            for i in range(len(row)-1):
                coeff[i + 1] = coeff[i + 1] - l_rate * error * row[i]
        print("Epoch= ", epoch, ", lrate=", l_rate, ", error=", sum_error)
    return coeff

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coeffs_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)