import numpy as np
import sklearn

from sklearn.linear_model import Ridge, LinearRegression, Lasso
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

# parsing and setting nonexistant values to -1
def float_or_null(x):
    a = None
    try:
        a = float(x)
    except:
        a = -1
    return a

data = [[float_or_null(x) for x in line] for line in data]

# handling ocean proximity
data_proximity = [line[-1] for line in lines]
proximity_values = sorted(list(set(data_proximity)))

# expanding ocean proximity classification string into a sparse array
# that means we're separating one feature (ocean proximity) into multiple
# for example the string 'NEAR BAY' gets converted to a data point that looks like this - [0, 0, 0, 1, 0]
# why? because the string is a classification and not a continuous data point
data_proximity = [[int(value == entry) for value in proximity_values] for entry in data_proximity]

# making median house value the target and separating it from the data
target_index = dataset['feature_names'].index('median_house_value')
clean_data = [line[:target_index] + line[(target_index + 1):] for line in data]
target = [line[target_index] for line in data]

# assembling the dataset
dataset['data'] = np.array([clean_data[i] + data_proximity[i] for i in range(len(data))])
dataset['target'] = np.array(target)

# splitting the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# training the regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# showing the model score
print("linreg Training set score: {:.3f}".format(linreg.score(X_train, y_train)))
print("linreg Test set score: {:.3f}".format(linreg.score(X_test, y_test)))

# training the regression model
ridge = Ridge(alpha=100)
ridge.fit(X_train, y_train)

# showing the model score
print("ridge Training set score: {:.3f}".format(ridge.score(X_train, y_train)))
print("ridge Test set score: {:.3f}".format(ridge.score(X_test, y_test)))

# training the regression model
lasso = Lasso()
lasso.fit(X_train, y_train)

# showing the model score
print("lasso Training set score: {:.3f}".format(lasso.score(X_train, y_train)))
print("lasso Test set score: {:.3f}".format(lasso.score(X_test, y_test)))