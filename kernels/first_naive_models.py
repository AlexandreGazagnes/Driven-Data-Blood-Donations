#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
01-first-naive-models.py
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


from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss

from first_dataset_tour import *


# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions

# @timer
def return_X_y(df) : 

    if "target" in df.columns : 

        X = df.drop("target", axis=1)
        y = df.target

        return X, y  

    else : 
        return df


# @timer
def split(X,y, size=0.33) : 

    return train_test_split(X, y, test_size=size)


# @timer
def naive_model(param=None, df=None) :

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test  = split(X,y)

    freq = y_test.value_counts() / len(y_test)
        
    y_pred = np.random.binomial(1, freq[1], len(y_test))
    y_pred = pd.Series(y_pred)

    lolo = log_loss(y_test, y_pred).round(3)

    return lolo, None 


# @timer
def dummy_model(param=None, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)
    
    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test = split(X,y)

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 

    lolo = log_loss(y_test, y_pred).round(3)

    return lolo, model


# @timer
def basic_model(param=None, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)
    
    X,y     = return_X_y(df)

    X_train, X_test, y_train, y_test = split(X,y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    lolo = log_loss(y_test, y_pred).round(3)
    
    return lolo, model


# @timer
def model_accuracy_mean(model=None, nb=5, df=None) : 
    
    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = basic_model

    scores = pd.Series([model(df)[0] for i in range(nb)])

    info(type(scores))
    info(type(range(nb)))

    score = scores.mean().round(3)

    return score


# ##############################################################
# ##############################################################


# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# ##############################################################
# ##############################################################


# # @timer
# def add_new_results(results=None, feat_com=None, n=5, 
#                     models=MODELS, columns= COLUMNS, 
#                     df=None) : 
    
#     if not isinstance(results, pd.DataFrame) : 
#         results = pd.DataFrame(columns=columns)

#     new = [model_accuracy_mean(i, n, df) for i in models]
#     info(new)

#     if not feat_com : 
#         feat_com = "No comment"

#     new.append(feat_com)
#     info(new)
    
#     new = pd.Series(new, index=columns)
#     info(new)
    
#     results = results.append(new, ignore_index=True)
#     info(results)

#     return results


# @timer
# def first_approch_of_feat_eng(  drop_list,
#                                 results=None,
#                                 n=5, 
#                                 models=MODELS, columns= COLUMNS, 
#                                 df=None) : 
    
#     if not isinstance(drop_list, list) : 
#         raise TypeError

#     if not isinstance(results, pd.DataFrame) : 
#         results = pd.DataFrame(columns=columns)

#     for i in drop_list : 

#         df = build_df(DATA, TRAIN_FILE)
#         df = delete_outliers(df, i) 

#         feat_com = "drop outliers > " + str(i)

#         results = add_new_results(  results=results,
#                                     feat_com=feat_com,
#                                     n=n, 
#                                     models=models, 
#                                     columns=columns, 
#                                     df=df)

#     return results


# @timer
# def first_naive_model() :  

#     results = pd.DataFrame(columns=COLUMNS)
#     results = add_new_results(results, "without_any_feat_eng")

#     results = first_approch_of_feat_eng([1.5, 2.0, 2.5, 3.0, 3.5])
    
#     return results



