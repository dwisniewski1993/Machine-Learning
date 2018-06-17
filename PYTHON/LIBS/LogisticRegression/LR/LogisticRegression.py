import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


class LogisticsRegression:
    def __init__(self, trainfile):
        self.trainFile = trainfile

        trainDataFrame = pd.read_csv(self.trainFile, header=0)
        trainDataFrame = trainDataFrame.dropna()
        # trainDataFrame['education'].unique()
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.9y', 'Basic',
                                               trainDataFrame['education'])
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.6y', 'Basic',
                                               trainDataFrame['education'])
        trainDataFrame['education'] = np.where(trainDataFrame['education'] == 'basic.4y', 'Basic',
                                               trainDataFrame['education'])
        #print(trainDataFrame.groupby('y').mean())

        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        for var in cat_vars:
            cat_list = 'var'+'_'+var
            cat_list = pd.get_dummies(trainDataFrame[var], prefix=var)
            data = trainDataFrame.join(cat_list)
            trainDataFrame = data

        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        data_vars = trainDataFrame.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in cat_vars]
        finalData = trainDataFrame[to_keep]
        finalData.columns.values
        cols = ["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
                "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
                "day_of_week_wed", "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
        self.features = finalData[cols]
        self.labels = finalData['y']

    def __str__(self):
        print("Train File: {}".format(self.trainFile))

    def modelFiting(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.labels,
                                                                                test_size=0.3, random_state=0)
        self.lr = LogisticRegression()
        self.lr.fit(self.X_train, self.Y_train)

    def output(self):
        print("Accuracy: {:.2f}".format(self.lr.score(self.X_test, self.Y_test)))
