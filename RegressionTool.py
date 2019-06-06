'''
Author: Gray Meeks
Project: Senior Seminar
File: RegressionTool.py

All of the programming and logic for this program
(aside from external library calls) was written
on my own.
'''
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
import math
import copy
import statistics
import numpy as np

# Returns a score value for a given
# classifier, x dataset, and y dataset.
def get_clf_score(clf, xs, ys):
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xs, ys, test_size=0.10)
        clf.fit(xtrain, ytrain)
        score = clf.score(xtest, ytest)
        if (score < 0):
            score = 0
        return score

# Returns the percent error of a a given
# estimated value and actual value.
def get_percent_error(estimated_val, actual_val):
    x = (actual_val - estimated_val) / actual_val
    y = abs(x)
    return y

# Returns the accuracy of a clf given the
# dataset as well as a specific % error
# to measure by.
def get_accuracy(clf, xs, ys, percent_error, trace=False):
    num_samples = len(xs)
    num_correct = 0
    for i in range(0, num_samples):
        curr_X = xs[i]
        curr_X = np.array(curr_X).reshape((1, -1))
        curr_y = ys[i]
        curr_estimation = clf.predict(curr_X)
        pct_error = get_percent_error(curr_estimation, curr_y)
        if (pct_error < percent_error):
            num_correct += 1
            if (trace):
                print(str(curr_y) + '\t\t' + str(curr_estimation))
    return num_correct / num_samples

# Fetches mutual information values for each feature
# and removes features that do not make the cut. An
# important exception is that if NONE of the features
# make the cut, the original x dataset is returned.
def mutual_information_filter(xs, ys, xnames, cutoff):
    mutual_information = feature_selection.mutual_info_regression(xs, ys)
    new_xs = xs
    new_features = xnames
    i = len(xs[0]) - 1
    while (i > -1):
        if (mutual_information[i] < cutoff):
            new_xs = np.delete(new_xs, i, axis=1)
            new_features = np.delete(new_features, i, axis=0)
        i -= 1
    if (len(new_xs[0]) > 0):
        return new_xs, new_features
    else:
        return xs, xnames

# Uses recursive feature elimination to determine
# the topmost features in a given x dataset.
# The cutoff indicates how many features you
# want to hold on to:
#
#   cutoff=.5 --> keep half of features
#   cutoff=.666 --> keep 2/3rds of features
#   cutoff=.333 --> keep 1/3rd of features
def recursive_feature_elimination(xs, ys, xnames, cutoff):
    estimator = ensemble.RandomForestRegressor()
    selector = feature_selection.RFE(estimator, math.floor(len(xs[0]) * cutoff), step=1)
    selector.fit(xs, ys)
    bool_arr = selector.support_
    new_xs = xs
    new_features = xnames
    i = len(xs[0])- 1
    while (i > -1):
        if (not bool_arr[i]):
            new_xs = np.delete(new_xs, i, axis=1)
            new_features = np.delete(new_features, i, axis=0)
        i -= 1
    if (len(new_xs[0]) > 0):
        return new_xs, new_features
    else:
        return xs, xnames
    
# Returns a refined dataset xs,ys
# based on filtering out features
def feature_select(xs, ys, xnames):

    # Feature select 1: recursive feature selection
    try:
        curr_xs, curr_xnames = recursive_feature_elimination(xs, ys, xnames, .90)
    except:
        curr_xs = xs
        curr_xnames = xnames
       
    # Feature select 2: mutual information
    new_xs, new_xnames = mutual_information_filter(curr_xs, ys, curr_xnames, .05)
    
    return new_xs, ys, new_xnames

class RegressionTool:

    '''
            ******ATTRIBUTES******* 

            xs:                 list of input values for regression dataset
            orig_features:      string list that identifies what each column stands for specifically wrt xs
            revised_xs:         feature-selected version of xs
            revised_features:   string list that identifies what each column stands for specifically wrt revised_xs
            ys:                 output values of the regression dataset
            label:              string that identifies what the output values represent
            classifier_list:    List of tuples that store sklearn classifiers

    '''

    def __init__(self, xs, xnames, ys, yname):
        # Error checking
        if (len(xs) != len(ys)):
            raise RuntimeError("xs and ys must be of the same length")
        if (len(xs) < 25):
            raise RuntimeError("must have at least 25 samples")
        x_size = len(xs[0])
        for x in xs:
            if (len(x) != x_size):
                raise RuntimeError("all xs must be the same length")
        if (len(xnames) != len(xs[0])):
            raise RuntimeError('Invalid xnames parameter')
            
        # ***************************
        # Initialize class attributes
        # ***************************

        # Dataset attributes
        refined_x, refined_y, refined_xnames = feature_select(xs, ys, xnames)
        self.xs = xs
        self.orig_features = xnames
        self.revised_xs = refined_x
        self.revised_features = refined_xnames
        self.ys = ys
        self.label = yname

        # List of all classifiers to be used
        self.classifier_list = [
            ('Random Forest', ensemble.RandomForestRegressor()),
            ('Support Vector Machine', svm.SVR()),
            ('K Nearest Neighbors', neighbors.KNeighborsRegressor()),
            ('Ridge Regression', linear_model.Ridge()),
            ('Lasso Regression', linear_model.Lasso()),
            ('Elastic Net', linear_model.ElasticNet()),
            ('Extra Trees', ensemble.ExtraTreesRegressor())
            ]
        return

    # Fits various classifiers to self's original,
    # unchanged dataset. fit_default returns the
    # following:
    #       1) Classifier name (string)
    #       2) Classifier score (float)
    #       3) Regression classifier (sklearn clf)
    def fit_default(self, iterations=5, debug=False):
        max_score = 0
        clf_name = 'None'
        best_clf = None
        for each in self.classifier_list:
            score_cum_sum = 0
            for i in range(0, iterations):
                clf = copy.deepcopy(each[1])
                score = get_clf_score(clf, self.xs, self.ys)
                score_cum_sum += score
            score_avg = score_cum_sum / iterations
            if (debug):
                print(score_avg)
            if (score_avg > max_score):
                max_score = score_avg
                clf_name = each[0]
                best_clf = each[1]
                best_clf.fit(self.xs, self.ys)
        return clf_name, max_score, best_clf


    # Fits various classifiers to self's feature-
    # selected dataset. fit_revised returns the
    # following:
    #       1) Classifier name (string)
    #       2) Classifier score (float)
    #       3) Regression classifier (sklearn clf)
    def fit_revised(self, iterations=5, debug=False):
        max_score = 0
        clf_name = 'None'
        best_clf = None
        for each in self.classifier_list:
            score_cum_sum = 0
            for i in range(0, iterations):
                clf = copy.deepcopy(each[1])
                score = get_clf_score(clf, self.revised_xs, self.ys)
                score_cum_sum += score
            score_avg = score_cum_sum / iterations
            if (debug):
                print(score_avg)
            if (score_avg > max_score):
                max_score = score_avg
                clf_name = each[0]
                best_clf = each[1]
                best_clf.fit(self.revised_xs, self.ys)
        return clf_name, max_score, best_clf

    '''
    Deprecated (used "see all" clf instead of traintestsplit sampler)
    def graph(self, clf, revised=True):
        if (revised):
            X = self.revised_xs
        else:
            X = self.xs
        print(X)
        xaxis = []
        y_hats = []
        for i in range(0, len(self.ys)):
            y_hat = clf.predict([X[i]])
            y_hats.append(y_hat)
            xaxis.append(i)
        plt.plot(xaxis, self.ys, 'o', color='black')
        plt.plot(xaxis, y_hats, 'x', color='blue')
        plt.show()
        return
    '''

    # Fits given clf to a train test split
    # sample and, for each label index,
    # graphs the actual label value as
    # well as the estimated label value.
    # Must specify if the given classifier
    # is feature-selected or original.
    def graph(self, clf, is_revised):
        if (is_revised):
            X = self.revised_xs
        else:
            X = self.xs
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, self.ys, test_size=0.10)
        clf.fit(xtrain, ytrain)
        xaxis =[]
        y_hats = []
        for i in range(0, len(xtest)):
            y_hat = clf.predict([xtest[i]])
            y_hats.append(y_hat)
            xaxis.append(i)
        actual = plt.plot(xaxis, ytest, 'o', color='black', label='y-actual')
        estimated = plt.plot(xaxis, y_hats, 'x', color='blue', label='y-estimated')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        plt.show()
        return

    # Calculates a list of percent errors s.t. each
    # percent error corresponds to a given y/yhat
    # tuple. The following two values returned
    # regarding the list:
    #       1) Average percent error
    #       2) Standard deviation of percent error
    def get_error_info(self, clf, is_revised):
        if (is_revised):
            X = self.revised_xs
        else:
            X = self.xs
        percent_errors = []
        for i in range(0, len(self.ys)):
            y_hat = clf.predict([X[i]])
            curr = get_percent_error(y_hat[0], self.ys[i])
            percent_errors.append(curr)
        avg = statistics.mean(percent_errors)
        stdev = statistics.stdev(percent_errors)
        return avg, stdev
