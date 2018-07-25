#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02_starting_ML.py
"""


"""
blablabla
blablabla
blablabla
blablabla
"""



# import

import itertools as it
from collections import OrderedDict

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

from first_naive_models import *


# consts 

# COLUMNS   = ["naive", "dummy", "basic", "features eng."]
# MODELS    = [naive_model, dummy_model, basic_model]


# functions


def beeper(funct) : 

    def wrapper(*args, **kwargs) : 
        
        res = funct(*args, **kwargs)

        freq=440
        duration=3
        cmd = 'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq)
        os.system(cmd)

        return res

    return wrapper


# @timer
@caller
def run_GSCV(   model, params=None, 
                df=None,
                cv=5, n_jobs=6, scoring="accuracy",
                verbose=1, return_train_score=False) : 

    model = model()


    if not isinstance(df, pd.DataFrame): 
        df = build_df()

    if not params : 
        params = dict()
     
    X,y         = return_X_y(df)
        
    X_train, X_test, y_train, y_test = split(X,y)


    try : 

        grid        = GridSearchCV( estimator=model, 
                                    param_grid=params,  
                                    cv=cv, 
                                    n_jobs=n_jobs,
                                    scoring=scoring, 
                                    verbose=verbose,
                                    return_train_score=return_train_score)


        grid.fit(X_train, y_train)

        info(grid.best_estimator_)
        info(grid.best_score_)
        info(grid.best_params_)

        try : 
            y_pred = grid.predict_proba(X_test)
            y_pred = y_pred[:,1]
        except : 
            y_pred = grid.predict(X_test)
        
        lolo = log_loss(y_test, y_pred).round(3)
        info(lolo)

        return lolo, grid

    except : 

        return 100, model 


#############################################################
#############################################################


COLUMNS = [     "dummy",    "basic", 
                "LR",       "RC",
                "SVC",      "Nu",
                "KNN",
                "DT", 
                "RF", 
                "Ada", 
                "Per",      "MLP",

                "features eng."]

MODELS = [      dummy_model, basic_model, 
                LogisticRegression, RidgeClassifier,
                LinearSVC, NuSVC,
                KNeighborsClassifier,
                DecisionTreeClassifier
                RandomForestClassifier,
                AdaBoostClassifier,
                Perceptron, MLPClassifier]


#############################################################
##############################################################


@timer
def benchmark_various_models_once(n=10, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df()

    results     = [model_accuracy_mean(m, n, df) for m in MODELS]
    results     = pd.Series(results, index=COLUMNS[:-1])

    info(results)

    return results


@timer
def benchmark_various_grid_multi(N=10, n=10, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df()

    benchmark   = benchmark_various_models_once
    series      = [benchmark(n, df) for i in range(N)]
    results     = pd.DataFrame(series, columns=COLUMNS[:-1])

    info(results)

    info(results.describe())
    info((results.T).describe())

    return results


def combine_param_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


def transform_df(Tool, df=None, **kwargs)

    if not isinstance(df, pd.DataFrame) : 
        df = first_tour()

    X = df.drop("target", axis=1)
    y = df.target

    tool = Tool(**kwargs)
    _X = tool.fit_transform(X)
    _X = pd.DataFrame(_X, columns=X.columns, index=X.index)
    _X["target"] = y

    return _X


def rescale(df=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = first_tour()

    return transform_df(StandardScaler, df)



def normalize(df=None, norm=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = first_tour()

    if not norm : 
        norm='l2'

    return transform_df(StandardScaler, df, norm=norm) 
    

def min_max(df=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = first_tour()

    return transform_df(MinMaxScaler, df)



def benchmark_various_outliers_once(df=None) :

    k_list = np.linspace(0.3, 5, 0.1) 
    
    pass


def benchmark_various_outliers_multi(df=None) : 
    
    pass



def benchmark_various_transform_once(df=None) : 

    transform_list = [None, normalize, rescale, min_max]
    
    pass


def benchmark_various_transform_once(df=None) : 
    
    pass


def benchmark_various_grid_once(df=None) : 

    cv = [3,5,10]
    test_size = [0.4, 0.33, 0.25]
    scoring = ["accuracy", "neg_loss_loss"]
    
    pass

def benchmark_various_grid_multi(df=None) : 
    
    pass