#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02-improving-naive-model.py
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


# @decor_run_GSCV
# @timer
@caller
def run_GSCV(   model, params, 
                df=None, 
                outliers=None, regularize=None,
                cv=10, n_jobs=6, scoring="neg_log_loss",
                verbose=0) : 

    model = model()

    comment=str()

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    # if outliers : 
    #     try : 
    #         outliers = float(outliers)
    #     except :
    #         raise ValueError("outliers params must be a float")

    #     if not (1.4<= outliers <= 3.5)  :
    #         raise ValueError("outliers must be 1.4<= outliers <= 3.5")

    # if outliers : 
    #     comment+="outliers k= "+ str(outliers)
    #     df = delete_outliers(df, outliers)
     
    X,y         = return_X_y(df)

    # if regularize : 
    #     raise ValueError("regularize funct not implemented yet")
    #     X = regularize(X)

    t           = split(X,y)

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
    lolo = log_loss(y_test, y_pred).round(3)
    info(lolo)

    return lolo, grid


# @decor_grid
# @timer
def grid_LogisticRegression(df=None, param=None,
                            model=LogisticRegression) : 

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

    all_params      = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001],  # consider also np.logspace(-6, 2, 9)
                        "C":[1.0],      # consider also np.logspace(-4, 2, 7)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[False, True],   }

    none_params     = {}

    best_params_1   = { "C" :[1],
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :[100],
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga"],
                        "tol":[0.0001],
                        "warm_start" :[True]     }

    best_params_2   = { "C" :np.logspace(-4, 2, 7),
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :np.logspace(3, 5, 3),
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga"],
                        "tol":np.logspace(-6, 2, 9),
                        "warm_start" :[True]     }


    if not param :  param = none_params
    else  :         param = best_params_2

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


# @decor_grid
# @timer
def grid_RidgeClassifier(   df=None, param=None,
                            model=RidgeClassifier):

    default_params  = { "alpha":[1.0],
                        "fit_intercept":[True],
                        "normalize":[False],
                        "copy_X":[True],
                        "max_iter":[None],
                        "tol":[0.001],
                        "class_weight":[None],
                        "solver":["auto"],
                        "random_state":[None]       }

    all_params       = { "alpha":[1.0], # consider np.logspace(-6,2, 9)
                        "fit_intercept":[True],
                        "normalize":[False, True],
                        "copy_X":[True],
                        "max_iter":[None], # consider also np.logspace(3, 5, 3)
                        "tol":[0.001],   # consider also np.logspace(-6, 2, 9)
                        "class_weight":[None],
                        "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                        "random_state":[None]       }





    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params


    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


# @decor_grid
# @timer
def grid_LinearSVC(     df=None, param=None,
                        model=LinearSVC):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


# @decor_grid
# @timer
def grid_NuSVC(         df=None,  param=None,
                        model=NuSVC):

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

    # lolo, grid    = run_GSCV(model, param, None)

    # return lolo, grid
    return -1.0, None


# @decor_grid
# @timer
def grid_KNeighborsClassifier(  df=None, param=None,
                                model=KNeighborsClassifier):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


# @decor_grid
# @timer
def grid_RandomForestClassifier(df=None, param=None, 
                                model=RandomForestClassifier):

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

    lolo, grid       = run_GSCV(model, param, None)
    
    return lolo, grid


# @decor_grid
# @timer
def grid_AdaBoostClassifier(df=None, param=None,
                            model=AdaBoostClassifier):

    default_params  = { "base_estimator" : [None],
                        "n_estimators" : [50],
                        "learning_rate" : [1.0],
                        "algorithm" : ["SAMME.R"],
                        "random_state" : [None] }


    # m1 = DecisionTreeClassifier()
    # m2 = LogisticRegression()
    # m3 = RandomForestClassifier()

    all_params      =  { "base_estimator" : [DecisionTreeClassifier(), LogisticRegression(),RandomForestClassifier() ],
                        "n_estimators" : [50],  # consider np.arange(0.099, 201, 23)
                        "learning_rate" : [1.0],  # np.logspace(-3, 2, 6)
                        "algorithm" : ["SAMME", "SAMME.R"],
                        "random_state" : [None] }


    best_params      =  { "base_estimator" : [DecisionTreeClassifier(), LogisticRegression(), RidgeClassifier(), RandomForestClassifier() ],
                        "n_estimators" : [int(i) for i in np.arange(1, 201, 23)],
                        "learning_rate" : np.logspace(-3, 2, 6),
                        "algorithm" : ["SAMME", "SAMME.R"],
                        "random_state" : [None] }



    none_params     = {}

    if not param :  param = none_params
    else  :         param = params

    lolo, grid       = run_GSCV(model, param, None)
    
    return lolo, grid


# @decor_grid
# @timer
def grid_Perceptron(    df= None, param=None,
                        model=Perceptron):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    lolo, grid       = run_GSCV(model, param, None)
    
    return lolo, grid


# @decor_grid
# @timer
def grid_MLPClassifier( df=None, param=None,
                        model=MLPClassifier):

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

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


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


@timer
def benchmark_various_grid_once(col, mod, n=10) : 

    results     = [model_accuracy_mean(m, n, None) for m in MODELS]
    results     = pd.Series(results, index=col[:-1])

    info(results)

    return results


@timer
def benchmark_various_grid_multi(col, mod, N=10, n=10) : 

    benchmark   = benchmark_various_grid_once
    series      = [benchmark(col, mod, n) for i in range(N)]
    results     = pd.DataFrame(series, columns=col[:-1])

    info(results)

    info(results.describe())
    info((results.T).describe())

    return results


def combine_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


@timer
def benchmark_various_params_once(model, params, n) : 

    param_dict = combine_dict(params)

    results = list()

    columns = list(params.keys())
    columns.append("lolo")

    for param in param_dict : 

        info("testing param : " + str(param))

        try : 
            lolos = [run_GSCV(model, param)[0] for i in range(n)]
            lolos = np.array(lolos)
            lolo = lolos.mean().round(3)
            # grid_param = grid.get_params()

            info("done")

        except Exception as e : 

            lolo, grid_param = -100.0, "invalid dict of args"
            info("invalid params")
            info(str(param))
            info(e)

        serie = {i: j[0] for i,j in param.items()}
        serie["lolo"] = lolo

        results.append(pd.Series(serie))

    results = pd.DataFrame(results, columns =columns.append("lolo") )

    # clean nas in lolo
    mask = results.lolo != -100.0
    results = results[mask]

    results.sort_values(by="lolo", ascending=False, inplace=True)

    return results


def graph_benchmark_various_params_once(model, params, n_list=[1, 3, 5,]) : # 10, 20, 50

    color_list = ["red", "green", "blue", "grey", "black", "yellow"]

    meta_results = list()

    for n, c in zip(n_list, color_list) : 
        results = benchmark_various_params_once(model, params, n)

        m, s = results.lolo.mean(), results.lolo.std()

        plt.scatter(results.index, results.lolo, c=c, marker=".")
        plt.plot(results.index, [m for i in results.index ], c=c)
        plt.plot(results.index, [m+s for i in results.index ], c=c,linestyle="dashed")
        plt.plot(results.index, [m-s for i in results.index ], c=c, linestyle="dashed")
        txt = "n={}, m={}, s={}".format(n,m,s)
        # plt.label(txt)

        meta_results.append((n, results))

    plt.title("result mean, std and val regarding numb of exp")
    plt.show()

    return meta_results


# def decor_run_GSCV(funct, *args, **kwargs) : 

#     def wrapper(funct)

#         return funct 

#     return wrapper


# def decor_grid(funct, *args, **kwargs) : 

#     def wrapper(funct)

#         return funct 

#     return wrapper


# def decor_run_GSCV_constructor() : 

#     #####

#     return decor_run_GSCV 


# def decor_grid_constructor() : 

#     #####

#     return decor_grid 


###########################################################
###########################################################

# rearding accuracy

# first Adaboost 
# second gridLr et gridRC
# why not RF
# then perc and NLT


###########################################################
###########################################################
