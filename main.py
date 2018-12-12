import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

FILENAME = "california_housing.csv"

# reading the dataset
lines = [line.rstrip().split(',') for line in open(FILENAME)]

dataset = {}
dataset['feature_names'] = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
    "ocean_proximity_is_1h_ocean",
    "ocean_proximity_is_inland",
    "ocean_proximity_is_island",
    "ocean_proximity_is_near_bay",
    "ocean_proximity_is_near_ocean"]

# skipping the feature names
lines = lines[1:]

# removing ocean proximity to handle it separately
data = [line[:-1] for line in lines]

# parsing and setting nonexistent values to -1
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

# expanding ocean proximity category string into a sparse array

# that means we're separating one feature (ocean proximity) into multiple
# for example the string 'NEAR BAY' gets converted to a data point that looks like this - [0, 0, 0, 1, 0]
# why? because the string is a classification/category and not a continuous data point

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

# random forest regressor was used because I found it to
# perform the best out of the models I've checked
# linear models (linreg, ridge, lasso) peaked at about 60% accuracy
# gradient boosting regressor had around the same score (81%) but was slower to train

forest = RandomForestRegressor(n_estimators=15, random_state=0, max_features=10)
forest.fit(X_train, y_train)

# showing the model score
print("Training set score: {:.3f}".format(forest.score(X_train, y_train)))
print("Test set score: {:.3f}".format(forest.score(X_test, y_test)))

# calculating feature importances
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# printing feature importance ranking
print("Feature ranking:")

for f in range(dataset['data'].shape[1]):
    print("{0}. {1} - {2}".format(f + 1, dataset['feature_names'][indices[f]], importances[indices[f]]))

# plotting feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(dataset['data'].shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(dataset['data'].shape[1]), indices)
plt.xlim([-1, dataset['data'].shape[1]])
plt.show()