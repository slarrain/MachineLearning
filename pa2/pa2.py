#
# Santiago Larrain
# slarrain@uchicago.edu
# CAPP30254
#

import pandas as pd
import re
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


def read_data (name, index_row):
    df = pd.read_csv(name).set_index(index_row)
    df.index.names = ['index']
    return df

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


def summary(df, interesting_stats):

    summary = df.describe().T

    #Change 50% to 'median'
    #summary.rename(columns={'50%':'median'})

    for col in summary.columns:
        if col not in interesting_stats:
            del summary[col]

    # mode not in describe
    m = 'mode'
    if m in interesting_stats:
        summary[m] = df.mode().T

    return summary


def histogram(df, limits):

    for col in df.columns:
        pl.figure()
        pl.xlabel(col)
        pl.ylabel('Persons')
        title = col+' histogram'
        pl.title(title)
        df[col].hist(bins=np.linspace(0, int(limits[col]), 10))
        name = col+'.png'
        pl.savefig(name)


def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for
    each variable is null and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)

def fill_simple(df, column):

    # I'm assuming that non responants had no dependents
    df[column] = df[column].fillna(0)

    # Alternative method with class-conditional imputation
    '''
    for value in df[column].unique():
        p = df[col] == value
        df.ix[p] = df[p].fillna(df[p].mean())
    '''

def fill_income(df):

    income_imputer = KNeighborsRegressor(n_neighbors=2)
    df_w_monthly_income = df[df.monthly_income.isnull()==False].copy()
    df_w_null_monthly_income = df[df.monthly_income.isnull()==True].copy()
    cols = ['number_real_estate_loans_or_lines',
    'number_of_open_credit_lines_and_loans']
    income_imputer.fit(df_w_monthly_income[cols], df_w_monthly_income.
        monthly_income)
    new_values = income_imputer.predict(df_w_null_monthly_income[cols])
    df_w_null_monthly_income.loc[:, 'monthly_income'] = new_values
    df2 = df_w_monthly_income.append(df_w_null_monthly_income)
    return df2

def cap(x, cap):
    if x < cap:
        return x
    else:
        return cap

def discretize(df, column, newcolumn, bin, capsize):

    df[newcolumn] = df[column].apply(lambda x: cap(x, capsize))
    df[newcolumn] = pd.cut(df[newcolumn], bins=bin, labels=False)
    return df

def binary(df, column):

    a = pd.get_dummies(df[column])
    df2 = pd.concat([df, a], axis=1)
    return df2

def train (x_train, y_train):

    #reg = LinearRegression()
    reg = KNeighborsClassifier()
    reg.fit(x_train, y_train)

    return reg

def test(reg, x_test, y_test):

    #prediction = reg.predict(x_test)
    #import pdb; pdb.set_trace()
    return reg.score(x_test, y_test)

def cross_v(df, indep_var, dep_var):

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(df[
        indep_var], df[dep_var])
    return x_train, x_test, y_train, y_test

def prep_histo (df):

    limits = {}
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        limit = mean+2*std #With in 2 standard deviations
        limits [col] = limit

    # Not general. Delete afterwords
    limits['serious_dlqin2yrs'] = 3
    limits['revolving_utilization_of_unsecured_lines'] = 1
    return limits


def save_df_to_csv(df, filename):
    df.to_csv(filename)

def do():
    name = 'cs-training.csv'
    index_row = 'Unnamed: 0'
    df = read_data(name, index_row)
    fix_columns_names(df)

    interesting_stats = ['mean', '50%', 'std', 'count', 'mode']

    explore = summary(df, interesting_stats)
    save_df_to_csv(explore, 'exploratory_analysis.csv')

    #limits = prep_histo(df)
    #histogram(df, limits)

    column = 'number_of_dependents'
    fill_simple(df, column)
    df2 = fill_income(df)

    b = 'revolving_utilization_of_unsecured_lines'
    c = 'discrete_line'
    df3 = discretize(df2, b, c, 5, 1.0)
    df4 = binary(df3, c)

    indep_var = ['revolving_utilization_of_unsecured_lines',
                     'age', 'number_of_time30-59_days_past_due_not_worse',
                     'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans',
                     'number_of_times90_days_late', 'number_real_estate_loans_or_lines',
                     'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents',]
    dep_var = 'serious_dlqin2yrs'

    x_train, x_test, y_train, y_test = cross_v(df4, indep_var, dep_var)
    regres = train (x_train, y_train)

    score = str(test (regres, x_test, y_test))
    #with open ('score.txt', 'w') as f:
        #f.write(score)
    print score
if __name__ == "__main__":
    do()
