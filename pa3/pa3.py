#
# Santiago Larrain
# slarrain@uchicago.edu
#

import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
import re
import pylab as pl
import time
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, accuracy_score, recall_score,roc_auc_score,f1_score,precision_recall_curve
import matplotlib.pyplot as plt
import pickle

# 1: Read the Data and Fix Columns names
def read_data (name, index_row):
    df = pd.read_csv(name).set_index(index_row)
    df.index.names = ['index']
    return df

def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for
    each variable is null and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def fix_columns_names(df):
    df.columns = [camel_to_snake(col) for col in df.columns]

# 2: Split the data into training and testing subsets
# Deprecated
def split_train_test(df):
    train_index, test_index = train_test_split(df.index) #Default value
    df_train = df.ix[train_index].copy()
    df_test = df.ix[test_index].copy()
    return df_train, df_test

# 3: Fill out the missing values
def fill_median(df_train, df_test):
    median = df_train.median()
    df_train.fillna(median, inplace=True)
    df_test.fillna(median, inplace=True)
    '''
    median = {}
    for value in df[col].unique():
        p = df[col] == value
        median[value] = df[p].median()
        df.ix[p] = df[p].fillna(median[value])
    for value in test[col].unique():
        p = test[col] == value
        try:
            test.ix[p] = test[p].fillna(median[value])
        except:
            import pdb; pdb.set_trace()
    '''

# 4: Normalize the Data
def normalize(df_train, df_test):

    mu = df_train.mean(axis=0)
    std = df_train.std(axis=0)
    df_train = (df_train-mu)/std

    mu2 = df_test.mean(axis=0)
    std2 = df_test.std(axis=0)
    df_test = (df_test-mu2)/std2



# 4 a: K
def knn(df, k, dep_var, features, test):
    best = []
    print 'KNN'
    for k in range(2, 20):
        start_time = time.time()
        clf = KNeighborsClassifier (n_neighbors=k)
        clf.fit(df[features], df[dep_var])
        score = clf.score(test[features], test[dep_var])
        print k, clf.score(df[features], df[dep_var]), clf.score(test[features], test[dep_var])
        end_time = time.time()
        print 'Time: ', end_time-start_time
        tm = end_time-start_time
        best.append([score, (k, tm), clf])
        #print 'Predict Prob === ', clf.predict_proba(test[features])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'k': best[1][0], 'time': best[1][1], 'clf': clf}


def log_reg(df, dep_var, features, test):
    print 'Logistic Regression'
    start_time = time.time()
    clf = LogisticRegression()
    clf.fit (df[features], df[dep_var])
    score = clf.score(test[features], test[dep_var])
    print score
    end_time = time.time()
    tm =end_time-start_time
    print 'Time: ', end_time-start_time
    #pred = clf.predict(test[features])
    #print classification_report(test[dep_var], pred)
    return {'score': score, 'time': tm, 'clf': clf}

def linear_svc (df, dep_var, features, test):

    print 'Linear SVC'
    best =[]
    param = [0.01, 0.1, 1.0, 10.0, 100.0]
    for x in param:
        start_time = time.time()
        clf = LinearSVC(C=x)
        clf.fit (df[features], df[dep_var])
        score = clf.score(test[features], test[dep_var])
        print x, score
        end_time = time.time()
        print 'Time: ', end_time-start_time
        tm = end_time-start_time
        best.append([score, (x, tm), clf])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'C': best[1][0], 'time': best[1][1], 'clf': clf}

def random_forest(df, dep_var, features, test):

    print 'Random Forest'
    best=[]
    for x in range(3, 30, 3):
        start_time = time.time()
        clf = RandomForestClassifier(n_estimators=x)
        clf.fit (df[features], df[dep_var])
        score = clf.score(test[features], test[dep_var])
        print x, score
        end_time = time.time()
        tm =end_time-start_time
        print 'Time: ', tm
        best.append([score, (x, tm), clf])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'n_estimators': best[1][0], 'time': best[1][1], 'clf': clf}

def dec_tree(df, dep_var, features, test):
    print 'Decision Tree'
    best=[]
    for depth in [1, 2, 3, 4, 5, None]:
        for x in range(1, 6):
            print 'Decission Tree'
            start_time = time.time()
            clf = DecisionTreeClassifier(max_features=x, max_depth=depth)
            clf.fit (df[features], df[dep_var])
            score =clf.score(test[features], test[dep_var])
            print 'depth: ', depth, ' Max_F: ', x, score
            end_time = time.time()
            tm =end_time-start_time
            print 'Time: ', tm
            best.append([score, (x, depth, tm), clf])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'max_features': best[1][0], 'max_depth': best[1][1] , 'time': best[1][2], 'clf': clf}


def boosting(df, dep_var, features, test):
    print 'Boosting'
    best = []
    for depth in [1, 2, 3, 4, 5]:
        for x in range(1, 6):
            for n_est in range(25, 200, 25):
                start_time = time.time()
                clf = GradientBoostingClassifier(max_features=x, max_depth=depth, n_estimators=n_est)
                clf.fit (df[features], df[dep_var])
                score = clf.score(test[features], test[dep_var])
                print 'depth: ', depth, ' Max_F: ', x, 'n estimators: ', n_est, score
                end_time = time.time()
                tm =end_time-start_time
                print 'Time: ', tm
                best.append([score, (x, depth, n_est, tm), clf])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'max_features': best[1][0], 'max_depth': best[1][1] , 'n_estimators': best[1][2], 'time': best[1][3], 'clf': clf}


def bagging(df, dep_var, features, test):
    print 'Bagging'
    best = []
    #best_maxfeautures = []
    #best_n_estimators = []
    for sample in [1, 2, 3, 4, 5]:
        for x in range(1, 6):
            for n_est in range(3, 21, 3):
                start_time = time.time()
                clf = BaggingClassifier(max_features=x, max_samples=sample, n_estimators=n_est)
                clf.fit (df[features], df[dep_var])
                score = clf.score(test[features], test[dep_var])
                print 'sample: ', sample, ' Max_F: ', x, 'n estimators: ', n_est, score
                end_time = time.time()
                tm =end_time-start_time
                print 'Time: ', tm
                best.append([score, (x, sample, n_est, tm), clf])
    best.sort(reverse=True)
    print best[0]
    best = best[0]
    return {'score': best[0], 'max_features': best[1][0], 'max_sample': best[1][1] , 'n_estimators': best[1][2], 'time': best[1][3], 'clf': clf}




def eval_metrics(dep_var, features, df_test, score):
    evaluation = {}
    # (accuracy, precision, recall, F1, area under curve, and precision-recall curves)
    plt.figure()

    for clas in score:
        predict = score[clas]['clf'].predict(df_test[features])
        if clas=='linear_svc':
            prob_predict = score[clas]['clf'].decision_function(df_test[features])
        else:
            prob_predict = score[clas]['clf'].predict_proba(df_test[features])
            prob_predict = prob_predict[:, 1]
        accuracy = accuracy_score(df_test[dep_var], predict)
        precision = precision_score(df_test[dep_var], predict)
        recall = recall_score(df_test[dep_var], predict)
        f1 = f1_score(df_test[dep_var], predict)
        prc_precision, prc_recall, prc_thresholds = precision_recall_curve(df_test[dep_var], prob_predict)
        auc = roc_auc_score (df_test[dep_var], prob_predict)

        plt.plot(prc_recall, prc_precision, label=clas)

        evaluation[clas] = {'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'f1':f1, 'AUC': auc}

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('precision_recall_curve.png')

    print evaluation
    return evaluation

def tocsv(dic, filename):
    df_dic = pd.DataFrame.from_dict(dic, orient='index')
    df_dic.to_csv(filename)

def process_and_save(score, em):

    for clas in score:
        del score[clas]['clf']

    tocsv(score, 'score.csv')
    tocsv(em, 'Eval_metrics.csv')


def run_models(df_train, dep_var, features, df_test):
    score = {}

    score['knn'] = knn(df_train, 3, dep_var, features, df_test)
    score['log_reg'] = log_reg(df_train, dep_var, features, df_test)
    score['linear_svc'] = linear_svc(df_train, dep_var, features, df_test)
    score['random_forest'] = random_forest (df_train, dep_var, features, df_test)
    score['decision_tree'] = dec_tree (df_train, dep_var, features, df_test)

    score['boosting'] = boosting (df_train, dep_var, features, df_test)

    score['bagging'] = bagging(df_train, dep_var, features, df_test)
    print '#################################'
    print score
    with open('pickle_score_2.txt', 'w') as f:
        pickle.dump(score, f)
    em = eval_metrics(dep_var, features, df_test, score)
    '''
    for clas in score:
        for metric in score[clas]:
            if metric != 'clf':
                em[clas][metric] = score[clas][metric]
    '''

    #process_and_save(score, em)

    return score, em



def cros_v(df):

    features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']

    dep_var = 'serious_dlqin2yrs'
    n_folds = 4
    kf = KFold(len(df[dep_var]), n_folds)
    evaluation = {}
    score = {}
    for train_index, test_index in kf:

        df_train = df.ix[train_index]
        df_test = df.ix[test_index]

        fill_median(df_train, df_test)
        normalize(df_train, df_test)

        score, eval_m = run_models(df_train, dep_var, features, df_test)
        if len(evaluation) == 0:
            evaluation = eval_m
        for clas in eval_m:
            for metric in eval_m[clas]:
                evaluation[clas][metric] = evaluation[clas].get(metric, 0)+ eval_m[clas][metric]/n_folds
        
    process_and_save(score, eval_m)


def do():
    name = 'cs-training.csv'
    index_row = 'Unnamed: 0'
    df = read_data(name, index_row)
    fix_columns_names(df)
    
    cros_v(df)

    '''
    df_train, df_test = split_train_test(df)


    fill_median(df_train, df_test)
    normalize(df_train, df_test)

    features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
            'monthly_income', 'age', 'number_of_times90_days_late']

    dep_var = 'serious_dlqin2yrs'

    run_models(df_train, dep_var, features, df_test)
    '''
if __name__ == "__main__":
    do()
