from csv import reader


class LR:
    def __init__(self, trainset, inset):
        self.trainFile = trainset
        self.inFile = inset

        self.dataset = self.load_dataset(self.trainFile)
        self.dataset2 = self.load_dataset(self.inFile)

        self.del_mark1(self.dataset)
        self.del_mark2(self.dataset2)

        for i in range(len(self.dataset[0])):
            self.str_column_to_float(self.dataset, i)
        for j in range(len(self.dataset2[0])):
            self.str_column_to_float(self.dataset2, j)

        minmax = self.dataset_minmax(self.dataset)
        self.normalize_dataset(dataset=self.dataset, minmax=minmax)

        self.switch_ONES_column(self.dataset)

        self.l_rate = 0.01
        self.n_epoch = 50


    @staticmethod
    def load_dataset(filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file, delimiter='\t')
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    @staticmethod
    def del_mark1(dataset):
        for row in dataset:
            mark = row.pop(3)

    @staticmethod
    def del_mark2(dataset):
        for row in dataset:
            mark = row.pop(2)

    @staticmethod
    def price_column(dataset):
        for row in dataset:
            price = row.pop(0)
            row.append(price)

    @staticmethod
    def str_column_to_float(dataset, column):
        for row in dataset:
            try:
                row[column] = float(row[column].strip())
            except:
                if row[column] == 'diesel':
                    row[column] = float(1)
                elif row[column] == 'benzyna':
                    row[column] = float(2)
                elif row[column] == 'gaz':
                    row[column] = float(3)
                else:
                    s = row[column]
                    q = ''.join(str(ord(c)) for c in s)
                    row[column] = float(q)

    @staticmethod
    def dataset_minmax(dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    @staticmethod
    def normalize_dataset(dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    @staticmethod
    def switch_ONES_column(dataset):
        for each in dataset:
            i = 0
            while i < 5:
                thing = each.pop(0)
                each.append(thing)
                i = i + 1

    @staticmethod
    def predict(row, coeffs):
        yhat = coeffs[0]
        for i in range(len(row) - 1):
            yhat += coeffs[i + 1] * row[i]
        return yhat

    def coeffs_sgd(self, train, l_rate, n_epochs):
        coeff = [0.0 for _ in range(len(train[0]))]
        for epoch in range(n_epochs):
            sum_error = 0
            for row in train:
                yhat = self.predict(row, coeff)
                error = yhat - row[-1]
                sum_error += error ** 2
                coeff[0] = coeff[0] - l_rate * error
                for i in range(len(row) - 1):
                    coeff[i + 1] = coeff[i + 1] - l_rate * error * row[i]
            print("Epoch= ", epoch, ", lrate=", l_rate, ", error=", sum_error)
        return coeff

    def linear_regression_sgd(self):
        predictions = []
        coef = self.coeffs_sgd(self.dataset, self.l_rate, self.n_epoch)
        print(coef)
        for row in self.dataset2:
            yhat = self.predict(row, coef)
            predictions.append(yhat)
        return predictions
