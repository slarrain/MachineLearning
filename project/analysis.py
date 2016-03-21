#
# Natnaell Mammo
# Santiago Larrain
#

import urllib
import pandas as pd
import sys
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from time import time
from sklearn.metrics import recall_score,precision_score,accuracy_score
from scipy import sparse

judges = ['AScalia', 'AMKennedy', 'CThomas','RBGinsburg', 'SGBreyer', 'JGRoberts', 'SAAlito', 'SSotomayor','EKagan']
train_cols = ['issue','issueArea','lcDispositionDirection','petitioner','respondent']

def read():
    facts = pd.read_table('cases_55-15.csv',sep='|')
    df_justice_all = pd.read_csv('SCDB_2015_02_justiceCentered_LegalProvision.csv', encoding="ISO-8859-1")
    return df_justice_all, facts

def tfvecto(list_of_facts, x=1):
    tfvec = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('english'), ngram_range=(1, x))
    X = tfvec.fit_transform(list_of_facts)
    return X

def run(n=0, bigram=False):
    df_justice_all, facts = read()
    df = pd.merge(df_justice_all,facts,left_on='docket',right_on='docket_number')
    cur_df = df[df['justiceName'].isin(judges)] #DF with only the current judges + Scalia
    cur_df = cur_df[cur_df.direction.notnull()] #DF where each Judge has a direction.
                                                # Can't train otherwise
    cur_df.reset_index(inplace=True)
    #bigram=False
    if n>1:
        bigram = True
    if bigram:
        X = tfvecto(cur_df.facts_of_the_case, 2)
    else:
        X = tfvecto(cur_df.facts_of_the_case)

    if (n==1 or n==2):
        jn = pd.get_dummies(cur_df.justiceName, sparse=True)
        A = sparse.coo_matrix(jn)
        Q = sparse.hstack([X, A])
        W = Q.tocsr()
        cross_v(X, cur_df)
        return

    # Facts + variables case

    X_array = X.toarray()
    facts_df = pd.DataFrame(X_array)

    #df = pd.concat([cur_df[train_cols+['direction', 'justiceName']], facts_df], axis=1)
    df = pd.concat([cur_df[['direction', 'justiceName']], facts_df], axis=1)
    # print ('Len with nulls: ',len(df))
    df = df[df.notnull().all(axis=1)].reset_index()
    # print ('Len with no Nulls: ', len(df))
    # df = df[df.notnull().all(axis=1)==False].reset_index()
    x = df.ix[:,df.columns != 'direction']
    # x = df.ix[:, [col for col in df.columns if col not in train_cols+['direction']]]
    # x = df.ix[:, train_cols+['justiceName']]

    x = pd.get_dummies(x)
    #x = x.ix[:, x.columns != 'justiceName']
    #rv = cross_v(x, df)
    #return rv
    cross_v(x, df)

def cross_v(X, cur_df):

    cur_df['predicted'] = 0
    kf = cross_validation.KFold(len(cur_df), n_folds=5, shuffle = True)
    scores = 0
    baselines = 0
    for train, test in kf:
        score, base, pred = model(X, cur_df, train, test)
        # if scores:
        #     for i in range(len(score)):
        #         scores[i] += score[i]
        # else:
        #     scores = score
        scores += score
        baselines += base
        cur_df.loc[test,['predicted']] = pred
        #cur_df.predicted[test] = pred
    # for score in socores:
    #     s = score/5
    #     print ('Score of CLF = ', s)
    scores = scores/5
    baselines = baselines/5
    print ('Score of CLF = ', scores)
    print ("Baseline of = ", baselines)
    rv_cols = ['predicted', 'direction', 'justiceName', 'issueArea']
    #return cur_df.loc[:,rv_cols]




def model (X, cur_df, train, test):

    model_RF = RandomForestClassifier(n_jobs=-2, n_estimators=80, max_features=None)
    # model_KN = KNeighborsClassifier(n_jobs=-2, n_neighbors=10)
    # model_SVC = LinearSVC()
    # model_Logistic = LogisticRegression()
    modelo = model_RF
    t0 = time()
    print ('Fitting model...')

    '''
    Case for DF
    '''
    if type(X)==pd.core.frame.DataFrame:
        modelo.fit(X.ix[train, :], cur_df.direction[train].astype(int))
        y = modelo.predict(X.ix[test,:])
        score = modelo.score(X.ix[test, :], cur_df.direction[test])
        baseline = accuracy_score(cur_df.direction[test], [cur_df.direction[test].value_counts().index[0]]*len(cur_df.direction[test]))

    else:
        '''
        Case for sparse Matrix
        '''
        modelo.fit(X[train], cur_df.direction[train].astype(int))
        y = modelo.predict(X[test])
        score = modelo.score(X[test], cur_df.direction[test])
        baseline = accuracy_score(cur_df.direction[test], [cur_df.direction[test].value_counts().index[0]]*len(cur_df.direction[test]))

    print ('Score: ', score)
    print ('Baseline: ', baseline)
    print("Time: %0.3fs" % (time() - t0))
    print ('---------------')

    return score, baseline, y
if __name__ == '__main__':
    n = 0
    # Cases:
    # No arg = n=0 = DF (facts+cols) unigram
    # n=1 = Sparse Matrix Unigram
    # n=2 Sparse Matrix bigram
    # n>3 DF Bigram

    if len(sys.argv) == 2:
        n = sys.argv[1]
        # n=1 is use the Sparse Matrix only. No n is use the DataFrame
    rv = run(int(n))
