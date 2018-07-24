#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
01-first-naive-model.py
"""


"""
in this second part, we will implement our first logistic regression model.

We will first implement by hand a naive classifier, then a dummy classifier 
(who does the same job), and finally a basic logistic regression model.

rather than looking at the results of a regression we will implement a 
function that will test the model x times and that will average the results
 obtained

we will then implement a results manager that will be a dataframe
"""


# import

# from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import *

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss

# from first_tour import *


# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions

# @timer
def return_X_y(df) : 

    X = df.drop("target", axis=1)
    y = df.target

    return X, y  


# @timer
def split(X,y) : 

    func = train_test_split
    tup = train_test_split(X, y)
    
    return tup


# @timer
def naive_model(df=None) :

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    freq = y_test.value_counts() / len(y_test)
        
    y_pred = np.random.binomial(1, freq[1], len(y_test))
    y_pred = pd.Series(y_pred)

    acc = accuracy_score(y_test, y_pred).round(3)

    return acc, None 


# @timer
def dummy_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()
    
    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred).round(3)

    return acc, model


# @timer
def basic_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()
    
    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred).round(3)
    
    return acc, model


@timer
def model_accuracy_mean(model, nb=5, df=None) : 

    scores = pd.Series([model(df)[0] for i in range(nb)])

    info(type(scores))
    info(type(range(nb)))

    score = scores.mean().round(3)

    return score


##############################################################
##############################################################


COLUMNS = ["naive", "dummy", "basic", "features eng."]
MODELS = [naive_model, dummy_model, basic_model]


##############################################################
##############################################################


# @timer
def add_new_results(results=None, feat_com=None, n=5, 
                    models=MODELS, columns= COLUMNS, 
                    df=None) : 
    
    if not isinstance(results, pd.DataFrame) : 
        results = pd.DataFrame(columns=columns)

    new = [model_accuracy_mean(i, n, df) for i in models]
    info(new)

    if not feat_com : 
        feat_com = "No comment"

    new.append(feat_com)
    info(new)
    
    new = pd.Series(new, index=columns)
    info(new)
    
    results = results.append(new, ignore_index=True)
    info(results)

    return results


@timer
def first_approch_of_feat_eng(  drop_list,
                                results=None,
                                n=5, 
                                models=MODELS, columns= COLUMNS, 
                                df=None) : 
    
    if not isinstance(drop_list, list) : 
        raise TypeError

    if not isinstance(results, pd.DataFrame) : 
        results = pd.DataFrame(columns=columns)

    for i in drop_list : 

        df = first_tour()
        df = delete_outliers(df, i) 

        feat_com = "drop outliers > " + str(i)

        results = add_new_results(  results=results,
                                    feat_com=feat_com,
                                    n=n, 
                                    models=models, 
                                    columns=columns, 
                                    df=df)

    return results


@timer
def first_naive_model() :  

    results = pd.DataFrame(columns=COLUMNS)
    results = add_new_results(results, "without_any_feat_eng")

    results = first_approch_of_feat_eng([1.5, 2.0, 2.5, 3.0, 3.5])
    
    return results


