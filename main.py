import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FILENAME = "california_housing.csv"

# reading the dataset
lines = [line.rstrip().split(',') for line in open(FILENAME)]

dataset = {}
dataset['feature_names'] = lines[0]

# skipping the feature names
lines = lines[1:]

# removing ocean proximity to handle it separately
data = [line[:-1] for line in lines]

def float_or_null(x):
    a = None
    try:
        a = float(x)
    except:
        a = -1
    return a

# parsing and setting nonexistant values to -1
data = [[float_or_null(x) for x in line] for line in data]

# handling ocean proximity
data_proximity = [line[-1] for line in lines]
proximity_values = sorted(list(set(data_proximity)))

# expanding ocean proximity classification integer into sparse array
data_proximity = [[int(value == entry) for value in proximity_values] for entry in data_proximity]

# making median house value the target and separating it from the data
target_index = dataset['feature_names'].index('median_house_value')
clean_data = [line[:target_index] + line[(target_index + 1):] for line in data]
target = [line[target_index] for line in data]

# assembling the dataset
dataset['data'] = np.array([clean_data[i] + data_proximity[i] for i in range(len(data))])
dataset['target'] = np.array(target)

print(dataset['data'][100])
print(dataset['target'][100])

# lines = [[int(x) for x in line] for line in lines]

# dataset = {}
# dataset['data'] = np.array([line[:-1] for line in lines])
# dataset['target'] = np.array([line[-1] for line in lines])

# # splitting the dataset into training and testing parts

# X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# # training the logistic regression model

# logreg = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class='auto')
# logreg.fit(X_train, y_train)

# # showing the model score

# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))