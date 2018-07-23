#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02-improving-naive-model.py
"""


"""

"""



# import

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

from first_naive_model import *

# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions

def run_grid(   model, params, 
                df=None, 
                outliers=None, regularize=None,
                cv=10, n_jobs=6, scoring="accuracy",
                verbose=0) : 

    comment=str()

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    if outliers : 
        try : 
            outliers = float(outliers)
        except :
            raise ValueError("outliers params must be a float")

        if not (1.4<= outliers <= 3.5)  :
            raise ValueError("outliers must be 1.4<= outliers <= 3.5")

    if outliers : 
        comment+="outliers k= "+ str(outliers)
        df = delete_outliers(df, outliers)
     
    X,y = return_X_y(df)

    if regularize : 
        raise ValueError("regularize funct not implemented yet")
        X = regularize(X)

    t = split(X,y)

    grid        = GridSearchCV( estimator=model, 
                                param_grid=params,  
                                cv=cv, 
                                n_jobs=n_jobs,
                                scoring=scoring, 
                                verbose=verbose)

    X_train, X_test, y_train, y_test = t 
    grid.fit(X_train, y_train)

    info(grid.best_estimator_)
    info(grid.best_score_)
    info(grid.best_params_)

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred).round(3)
    info(acc)

    return acc, grid


def grid_LogisticRegression(df=None, param=None) : 

    default_params  = { "penalty":["l2"],
                        "dual":[False],
                        "tol":[0.0001],
                        "C":[1.0],
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["liblinear"],
                        "max_iter":[100],
                        "multi_class":["ovr"],
                        "warm_start":[False],   }

    none_params     = {}

    params          = { "penalty":["l2"],
                        "dual":[True],
                        "tol":np.logspace(-6, 2, 9),
                        "C":np.logspace(-4, 2, 7),
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "loss":["hinge","squared_hinge"],
                        "solver":["liblinear"],
                        "max_iter":np.logspace(3, 5, 3),
                        "multi_class":["ovr","crammer_singer" ],
                        "warm_start":[False, True],   }

    if not param :  param = none_params
    else  :         param = params

    model           = LogisticRegression()
    acc, grid       = run_grid(model, param, None)

    return grid


def grid_RidgeClassifier(df=None, param=None):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    model           = RidgeClassifier()
    acc, grid       = run_grid(model, param, None)

    return grid



def grid_LinearSVC(df=None, param=None):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    model   = LinearSVC()
    grid    = run_grid(model, param, None)

    return grid


def grid_NuSVC(df=None, param=None):

    # default_params  = {}

    # none_params     = {}

    # params          = { "nu":[0.5],
    #                     "kernel":["rbf"],
    #                     "degree":[3],
    #                     "gamma":["auto"],
    #                     "coef0":[0.0],
    #                     "shrinking":[True],
    #                     "probability":[False],
    #                     "tol":[0.001],
    #                     "_size":[200],
    #                     "class_weight":[None],
    #                     "max_iter":[-1],
    #                     "decision_function_shape":["ovr"]}

    # if not param :    param = none_params
    # else  :       param = params

    # model   = NuSVC()
    # grid    = run_grid(model, param, None)

    # return grid
    return -1.0

def grid_KNeighborsClassifier(df=None, param=None):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    model   = KNeighborsClassifier()
    grid    = run_grid(model, param, None)

    return grid


def grid_RandomForestClassifier(df=None, param=None):

    default_params  = { "n_estimators" : [10],
                        "criterion" : ["gini"], # or    "entropy"
                        "max_features" : ["auto"], # "auto" eq sqrt or"log2", 
                        "max_depth" : [None], 
                        "min_samples_split" : [2],
                        "min_samples_leaf" : [1],
                        "min_weight_fraction_leaf" : [0.0],
                        "max_leaf_nodes" : [None],
                        "min_impurity_decrease" : [0.0],
                        "min_impurity_split" : [None],
                        "bootstrap" : [True],
                        "oob_score" : [False],
                        "n_jobs" : [1],
                        "random_state" : [None],
                        "verbose" : [0],
                        "warm_start" : [False],
                        "class_weight" : [None]     }

    none_params     = {}

    params          = { "n_estimators" : [100],
                        "criterion" : ["gini"],
                        "max_depth" : [None],
                        "min_samples_split" : [2],
                        "min_samples_leaf" : [1],
                        "min_weight_fraction_leaf" : [0.0],
                        "max_features" : [None],
                        "max_leaf_nodes" : [None],
                        "min_impurity_decrease" : [0.0],
                        "min_impurity_split" : [None],
                        "bootstrap" : [True],
                        "oob_score" : [True],
                        "warm_start" : [True],  }

    if not param :  param = none_params
    else  :         param = params

    model           = RandomForestClassifier()
    acc, grid       = run_grid(model, param, None)
    
    return grid


def grid_AdaBoostClassifier(df=None, param=None):

    default_params  = {}

    none_params     = {}

    if not param :  param = none_params
    else  :         param = params

    model           = AdaBoostClassifier()
    acc, grid       = run_grid(model, param, None)
    
    return grid


def grid_Perceptron(df=None, param=None):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    model           = Perceptron()
    acc, grid       = run_grid(model, param, None)
    
    return grid



def grid_MLPClassifier(df=None, param=None):

    default_params  = {}

    none_params     = {}

    params          = { "hidden_layer_sizes": [(4,4,4)],
                        "alpha": np.logspace(-5, 1, 20),
                        "max_iter": [200, 500],
                        "activation": ["logistic", "tanh", "relu"],
                        "solver": ["lbfgs", "sgd", "adam"],
                        "warm_start": [True],
                        "tol": np.logspace(-6, -2, 10)}
        
# "activation": ["identity", "logistic", "tanh", "relu"],   
# "solver": ["lbfgs", "sgd", "adam"],   
# "alpha": np.logspace(-4, 2, 9), 

    params          = { "hidden_layer_sizes": [(3,3,3), (3,3), (4,4), (5,5), (5,5,5), (4,4,4)],   
                        "activation": ["identity", "logistic", "tanh", "relu"],   
                        "solver": ["lbfgs", "sgd", "adam"],   
                        "alpha": np.logspace(-4, 2, 9),   
                        "batch_size": ["auto"],   
                        "learning_rate": ["constant", "invscaling", "adaptive"],   
                        "learning_rate_init": [0.001],   
                        "power_t": [0.5],   
                        "max_iter": [200, 1000, 5000],   
                        "shuffle": [True],      
                        "tol": [0.0001],      
                        "warm_start": [False, True],   
                        "momentum": [0.9],   
                        "nesterovs_momentum": [True],   
                        "early_stopping": [False],   
                        "validation_fraction": [0.1],   
                        "beta_1": [0.9],   
                        "beta_2": [0.999],   
                        "epsilon": [1e-08]}

    if not param :  param = none_params
    else  :         param = params

    model           = MLPClassifier()
    acc, grid       = run_grid(model, param, None)

    return acc, grid


def beep(freq=440, duration=3) : 

    cmd = 'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq)
    os.system(cmd)


#############################################################
#############################################################


COLUMNS = [     "dummy", 
                "basic", 

                "gridLR",
                "gridRC",

                "gridSVC",
                "gridNu",

                "gridKNN",

                "gridRF",

                "gridAda", 

                "gridPer",
                "gridMLP",

                "features eng."]

MODELS = [      dummy_model, 
                basic_model, 

                grid_LogisticRegression,
                grid_RidgeClassifier,

                grid_LinearSVC, 
                grid_NuSVC,

                grid_KNeighborsClassifier,

                grid_RandomForestClassifier,

                grid_AdaBoostClassifier,

                grid_Perceptron,
                grid_MLPClassifier]


#############################################################
##############################################################


def benchmark_various_grid_once(col, mod, n=10) : 

    results     = [model_accuracy_mean(m, n, None) for m in MODELS]
    results     = pd.Series(results, index=col[:-1])

    return results


def benchmark_various_grid_multi(col, mod, N=10, n=10) : 

    benchmark   = benchmark_various_grid_once
    series      = [benchmark(col, mod, n) for i in range(N)]
    results     = pd.DataFrame(series, columns=col[:-1])

    return results





# results = first_approch_of_feat_eng(results, [1.5, 2.0, 2.5, 3.0, 3.5])
    


