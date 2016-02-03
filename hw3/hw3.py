#!/usr/bin/env python3

#
# Santiago Larrain
# slarrain@uchicago.edu
#

import copy
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import genfromtxt
from statistics import mean

from collections import Counter
#%matplotlib inline
from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

mpl.rc('figure', figsize=[8,8])  #set the default figure size

class NaiveBayes():

    def __init__ (self):
        self.X = []
        self.y = []
        self.classes = {}
        self.f = {}
        self.prob = {}

    def fit(self, X, y):
        self.X = X
        self.y = y
        freq = dict(Counter(y))
        # Double check which way is better
        #for key in freq:
            #freq[key] /= len(y)
        self.classes = freq

        #Count the number of times a f belongs to a class for a and b
        for i in range(len(X)): #long list
            for j in range(len(X[i])):  #only 2 in this case
                if j not in self.f.keys(): # The dictionary will start with a, b = {0, 1}
                    self.f[j] = {}
                if X[i][j] not in self.f[j].keys():
                    self.f[j][X[i][j]] = {0:0, 1:0} #Hardcode :(
                #print (self.f[j][X[i][j]])          # class_i : n_times
                self.f[j][X[i][j]][y[i]] += 1       # class_i = n_times + 1

        #Now compute the probability
        self.prob = copy.deepcopy(self.f)
        for ab in self.prob:
            for f in self.prob[ab]:
                for clas in self.prob[ab][f]:
                    self.prob[ab][f][clas] /= self.classes[clas]

    def get_class(self, ab):
        probs = {}
        for c in self.classes.keys():
            p=self.classes[c]/len(self.y)
            for i in range(len(ab)):
                if ab[i] not in self.prob[i].keys():
                    m = 0.0
                else:
                    m = self.prob[i][ab[i]][c]
                p *=  m #P(a/b|c)
            probs[c] = p
        return max(probs, key=probs.get)    #returns the argmax

    def predict(self, array):
        results = np.array([], dtype=int)
        n=0   #Empty array
        for ab in array:
            results = np.append(results, self.get_class(ab))
            n+=1
            #print (n)
        return results


def run_model(X, y):
    kf = KFold(len(y), n_folds=5)
    lg_results = []
    nb_results = []
    for train_index, test_index in kf:
        lg = LogisticRegression()
        nb = NaiveBayes()
        lg.fit(X[train_index], y[train_index])
        nb.fit(X[train_index], y[train_index])
        lg_pred = lg.predict(X[test_index])
        nb_pred = nb.predict(X[test_index])
        lg_results.append(f1_score(y[test_index], lg_pred))
        nb_results.append(f1_score(y[test_index], nb_pred))
    print (lg_results, nb_results)
    return mean(lg_results), mean(nb_results)

def run_file():

    for i in range(1, 4):
        D = genfromtxt('dataset'+str(i)+'.csv', delimiter=',')
        X = D[:, 0:2]
        y = D[:, 2]
        lgr, nbr = run_model (X, y)
        print ("For the dataset"+str(i)+" the results for Logarithmic Regression are "+
                str(lgr)+" and for Naive Bayes are "+str(nbr))





# This helper function will help you plot the decision boundary.
# It also shows the dataset.  If you add functions to your Naive
# Bayes classifier similar predict and fit in
# sklearn.linear_model.LinearRegression (see the documentation)
# use should be able to use this without any modification.  See
# the example with LinearRegression below.

def plot_decision_boundary(predict_func, X, train_func=None, y=None):
    '''Plot decision boundary for predict_func on data X.

    If train_func and y are provided, first train the classifier.'''
    print (1)
    if train_func and y is not None:
        print (2)
        train_func(X, y)
        print (3)

    # Plot decision boundary

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))

    Z = predict_func(np.c_[np.around(xx.ravel()),np.around(yy.ravel())])
    Z = np.asarray(Z).reshape(xx.shape)

    _ = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)

    # Plot data points

    # Add small random noise so points with the same values
    # can be see differently
    X0 = X[:,0] + np.random.random(size=len(X))/5
    X1 = X[:,1] + np.random.random(size=len(X))/5

    # Plot the two classes separately with different colors/markers
    X0_class0 = X0[y==0]
    X1_class0 = X1[y==0]
    _ = plt.scatter(X0_class0, X1_class0, marker='.', c='green')

    X0_class1 = X0[y==1]
    X1_class1 = X1[y==1]
    _ = plt.scatter(X0_class1, X1_class1, marker='+', c='red')
    plt.savefig('plot.png')

if __name__ == "__main__":

    from numpy import genfromtxt
    from sklearn.linear_model import LogisticRegression


    #lr_clf = LogisticRegression()
    lr_clf = hw3.NaiveBayes()
    D = genfromtxt('dataset1.csv', delimiter=',')

    X = D[:, 0:2]
    y = D[:, 2]

    hw3.plot_decision_boundary(lr_clf.predict, X, lr_clf.fit, y)


    from numpy import genfromtxt
    from sklearn.metrics import f1_score
    import hw3
    clf = hw3.NaiveBayes()
    clf = LogisticRegression()
    D = genfromtxt('dataset3.csv', delimiter=',', dtype=int)
    D1 = D[:150]
    D2 = D[150:]
    X = D1[:, 0:2]
    y = D1[:, 2]
    clf.fit(X, y)
    X2 = D2[:, 0:2]
    y2 = D2[:, 2]
    pred = clf.predict(X2)
    f1_score(y2, pred)

