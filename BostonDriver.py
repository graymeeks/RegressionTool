'''
Author: Gray Meeks
Project: Senior Seminar
File: BostonDriver.py

All of the programming and logic for this program
(aside from external library calls) was written
on my own.
'''
from sklearn import datasets
import RegressionTool
import warnings
warnings.filterwarnings("ignore")

# Loads boston housing dataset utilizing
# call from sklearn library
def grab_data():
    my_dataset = datasets.load_boston()
    xs = my_dataset.data
    ys = my_dataset.target
    return xs, ys

# Initialize data
xs, ys = grab_data()
xnames = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSAT'
    ]
yname = 'MEDV'

print('Dataset: boston')
print('Label: median home value')

tool = RegressionTool.RegressionTool(xs, xnames, ys, yname)

# Part1: all features
name1, score1, clf1 = tool.fit_default(debug=False, iterations=10)
avg1, stdev1 = tool.get_error_info(clf1, is_revised=False)
print('\n\nOriginal xs:\n')
print('Best classifier: {}'.format(name1))
print('Score: {}'.format(score1))
print('Average % error: {}'.format(avg1))
print('Features: {}'.format(tool.orig_features))
tool.graph(clf1, is_revised=False)

# Part2: select features
tool = RegressionTool.RegressionTool(xs, xnames, ys, yname)
name2, score2, clf2 = tool.fit_revised(debug=False, iterations=10)
avg2, stdev2 = tool.get_error_info(clf2, is_revised=True)
print('\n\nFeature-selected xs:\n')
print('Best classifier: {}'.format(name2))
print('Score: {}'.format(score2))
print('Average % error: {}'.format(avg2))
print('Features: {}'.format(tool.revised_features))
tool.graph(clf2, is_revised=True)
