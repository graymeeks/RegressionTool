'''
Author: Gray Meeks
Project: Senior Seminar
File: EnergyDriver.py

All of the programming and logic for this program
(aside from external library calls) was written
on my own.
'''
from sklearn import datasets
import RegressionTool
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Load dataset and print information about it to screen
data = np.genfromtxt('energy.csv', delimiter=',')
print('Dataset: factory energy')
print('Label: energy output')

# Select features and label
xs = data[:,[0, 1, 2, 3]]
ys = data[:, 4]
xnames= ['temp', 'ambient presssure', 'relative humidity', 'exhaust vacuum']
yname = 'energy output'

# ************************
# Instantiate and use tool
# ************************

tool = RegressionTool.RegressionTool(xs, xnames, ys, yname)

'''
# Part1: all features
name1, score1, clf1 = tool.fit_default(debug=False, iterations=1)
avg1, stdev1 = tool.get_error_info(clf1, is_revised=False)
print('\n\nOriginal xs:\n')
print('Best classifier: {}'.format(name1))
print('Score: {}'.format(score1))
print('Average % error: {}'.format(avg1))
print('Features: {}'.format(tool.orig_features))
tool.graph(clf1, is_revised=False)
'''

# Part2: select features
tool = RegressionTool.RegressionTool(xs, xnames, ys, yname)
name2, score2, clf2 = tool.fit_revised(debug=False, iterations=15)
avg2, stdev2 = tool.get_error_info(clf2, is_revised=True)
print('\n\nFeature-selected xs:\n')
print('Best classifier: {}'.format(name2))
print('Score: {}'.format(score2))
print('Average % error: {}'.format(avg2))
print('Features: {}'.format(tool.revised_features))
tool.graph(clf2, is_revised=True)
