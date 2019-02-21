# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:54:25 2018

@author: fjehlik


Model Evaluation and Validation

In this project, machine learning concepts are applied on data collected for housing prices in the Boston, 
Massachusetts area to predict the selling price of a new home. The data is explored to obtain important features 
and descriptive statistics about the dataset. Next, the data is split into testing and training subsets, 
and a suitable performance metric for this problem determined. From this analysis an optimal model that best generalizes for unseen data
was selected. Also, a seperate script generates descriptive plots for analysis. 
From this the optimal model on a new sample and compare the predicted selling price is compared.

"""

# Import libraries necessary for this project
import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import ShuffleSplit, train_test_split

# Load the Boston housing dataset. Add a line of code to point to director where dataset is stored. 
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))


"""CALCULATE STATISTICS FOR HOUSING PRICES:
    This section calculates some base statistics for the Boston housing price dataset
"""
#calculate statistics for the boston housing prices
minimum_price = min(prices)

# Maximum price of the data
maximum_price = max(prices)

# Mean price of the data
mean_price = np.median(prices)

# Median price of the data
median_price = np.mean(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.0f}".format(minimum_price)) 
print("Maximum price: ${:,.0f}".format(maximum_price))
print("Mean price: ${:,.0f}".format(mean_price))
print("Median price ${:,.0f}".format(median_price))
print("Standard deviation of prices: ${:,.0f}".format(std_price))

"""DEFINE PERFORMACE METRIC:
    This section of code uses r^2 to fit the data and see if it is a good fit to the developed model"""

# Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    performance_score=r2_score(y_true,y_predict)
    
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = performance_score
    
    # Return the score
    return score

"""SHUFFLE AND SPLIT DATA TO DEVELOP MODEL:
        Test portion needs to be removed from dataset to fit the model. 
        Don't ever use test data for model development"""

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state=20)

"""FITTING THE MODEL:
    This section fits the model """
    # Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#clf = DecisionTreeClassifier(random_state=42)
clf = DecisionTreeRegressor(random_state=42)

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], test_size=0.10, random_state=0)
    
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
   
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

